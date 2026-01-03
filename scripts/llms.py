#!/usr/bin/env python
"""
Trainer-based fine-tuning for decoder-only LLMs adapted to sequence classification
(e.g., MistralForSequenceClassification via AutoModelForSequenceClassification).

- Loads `prepared.parquet` + `label2id.json` (même format que BERT).
- Hugging Face `Trainer` + `DataCollatorWithPadding`.
- Causal-decoder best practices: left padding, explicit PAD token.
- Optional 4-bit/8-bit quantization (BitsAndBytes) + optional LoRA (PEFT) for QLoRA.
- Small-test helpers: subset, fast_dev_run, limit batches via `max_steps`, tqdm (Trainer gère), seed.
- Checkpoints: **best-only** (save_strategy=epoch, save_total_limit=1, load_best_model_at_end=True) sous `<run_dir>/`.

Exemple rapide (TinyLlama pour smoke test):
  python LLMs/scripts/train.py \
    --data_path ./runs/biorc/prepared/prepared.parquet \
    --label_map ./runs/biorc/prepared/label2id.json \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --output_dir ./runs/biorc/tinyllama-seqcls/seed-1 \
    --epochs 1 --batch_size 8 --max_len 256 --fast_dev_run

Puis tu peux passer sur un Mistral public (ex: `mistralai/Mistral-7B-Instruct-v0.1`) si tu as le GPU et/ou QLoRA.
"""
from __future__ import annotations



import datasets, os
root_path = os.environ['DSDIR'] + '/HuggingFace_Models'


import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Optional deps
try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

try:
    from peft import (
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )  # type: ignore
except Exception:  # pragma: no cover
    LoraConfig = TaskType = prepare_model_for_kbit_training = None  # type: ignore


try:
    import idr_torch
    on_jean_zay = True
except ImportError:
    on_jean_zay = False


# Configuration pour Jean Zay (si détecté)
if on_jean_zay:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Initialisation des variables d'environnement
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=idr_torch.size,
        rank=idr_torch.rank
    )

    torch.cuda.set_device(idr_torch.local_rank)
    device = torch.device("cuda")
else:
    # Configuration pour l'exécution en local
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # (on évite get_device() non défini)

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s: List[float] = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def build_tokenizer(model_name: str):

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure a PAD token exists and use left padding for causal decoders.
    if tok.pad_token is None:
        # prefer unk as pad; otherwise add a new token
        if tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.padding_side = "left"  # causal decoders: classify on last non-pad token
    return tok


def get_quant_config(args) -> Optional[BitsAndBytesConfig]:
    if BitsAndBytesConfig is None:
        return None
    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if args.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def add_lora_like_doc(model, args):
    """
    Implémentation qui suit la doc PEFT :
      - construction LoraConfig(...)
      - model.add_adapter(lora_config, adapter_name="lora_seqcls")
      - model.set_adapter("lora_seqcls")
    """
    if LoraConfig is None:
        return model

    # Cibles typiques pour Mistral/LLaMA; l'utilisateur peut surcharger via --lora_target
    target = [t.strip() for t in args.lora_target.split(",") if t.strip()] if args.lora_target else [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # Comme dans la doc : inference_mode=False pour l'entraînement
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,        # DOC: choisir le TaskType adapté (CAUSAL_LM dans la doc, ici SEQ_CLS)
        inference_mode=False,              # DOC: set to False for training
        r=args.lora_r,                     # DOC: r=8 (par défaut 8)
        lora_alpha=args.lora_alpha,        # DOC: scaling factor
        lora_dropout=args.lora_dropout,    # DOC: dropout
        bias="none",
        target_modules=target,
        # DOC: pour entraîner aussi la tête de sortie (analogue lm_head); pour seq cls c'est "score"
        modules_to_save=["score"],
    )

    # DOC: add_adapter()
    model.add_adapter(lora_config, adapter_name="lora_seqcls")
    # DOC: set_adapter() si plusieurs adapters; ici on force l'usage de celui qu'on vient d'ajouter
    model.set_adapter("lora_seqcls")

    try:
        # méthode pratique de PEFT si présente; sinon pas grave
        model.print_trainable_parameters()
    except Exception:
        pass

    return model


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def main() -> int:
    p = argparse.ArgumentParser()
    # NEW: structure runs/<task>/<exp>/seed-<seed> -------------------------
    p.add_argument("--task_name", type=str, required=True, help="ex: biorc, scotus-steps, ...")  # NEW
    p.add_argument("--exp_name", type=str, required=True, help="nom d'expérience défini par toi")  # NEW
    # Data
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--label_map", type=Path, required=True)
    # Model
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--output_dir", type=Path, required=False, default=None)  # sera écrasé par la structure NEW
    # Train basics
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # Quantization / LoRA
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target", type=str, default="", help="comma list of module name substrings")
    # Small tests
    p.add_argument("--subset_train", type=int, default=0)
    p.add_argument("--subset_dev", type=int, default=0)
    p.add_argument("--fast_dev_run", action="store_true")
    # Trainer extras
    p.add_argument("--report_to", type=str, default="none", help="none|tensorboard|wandb")
    p.add_argument("--eval_metric", type=str, default="macro_f1", help="macro_f1|accuracy")
    # NEW
    p.add_argument("--infer", action="store_true", help="Run inference on train/dev/test after training")


    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # NEW: impose la structure de sortie (et crée les dossiers) ------------
    base_dir = Path(args.output_dir) / args.task_name / args.exp_name / f"seed-{args.seed}"
    tb_dir = base_dir / "tb_logs"
    (base_dir / "checkpoints" / "best").mkdir(parents=True, exist_ok=True)
    (base_dir / "inference").mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = base_dir  # on réutilise output_dir partout en-dessous

    # Data
    df = pd.read_parquet(args.data_path)
    with open(args.label_map, "r", encoding="utf-8") as f:
        label2id: Dict[str, int] = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}
    n_labels = len(label2id)

    train_df = df[df.split == "train"].copy()
    dev_df = df[df.split == "dev"].copy()
    if args.fast_dev_run:
        args.subset_train = args.subset_train or 64
        args.subset_dev = args.subset_dev or 64
    if args.subset_train and args.subset_train > 0:
        train_df = train_df.head(args.subset_train)
    if args.subset_dev and args.subset_dev > 0:
        dev_df = dev_df.head(args.subset_dev)

    model_dir = Path(root_path) / args.model_name
    if not model_dir.exists():
        model_dir = Path("./models") / args.model_name

    tok = build_tokenizer(str(model_dir))

    # tok = build_tokenizer(root_path +'/'+ args.model_name)

    def _preprocess(batch):
        # Active padding+truncation ici pour avoir des longueurs homogènes.
        return tok(batch["text"], padding=True, truncation=True, max_length=args.max_len)

    # Build HF datasets
    train_hf = HFDataset.from_pandas(train_df.reset_index(drop=True))
    dev_hf = HFDataset.from_pandas(dev_df.reset_index(drop=True))
    # map to tokenized
    train_hf = train_hf.map(_preprocess, batched=True, remove_columns=[c for c in train_hf.column_names if c not in {"text", "label"}])
    dev_hf = dev_hf.map(_preprocess, batched=True, remove_columns=[c for c in dev_hf.column_names if c not in {"text", "label"}])
    # rename label -> labels and to ids
    def _lab(batch):
        labs = [int(label2id[str(x)]) for x in batch["label"]]
        return {"labels": labs}
    train_hf = train_hf.map(_lab, batched=True)
    dev_hf = dev_hf.map(_lab, batched=True)

    from datasets import Value

    # On supprime la colonne 'label' source pour éviter que le collator la voie (et plante)
    train_hf = train_hf.remove_columns([c for c in train_hf.column_names if c == "label"])
    dev_hf = dev_hf.remove_columns([c for c in dev_hf.column_names if c == "label"])

    # On force le type de 'labels' à int64
    train_hf = train_hf.cast_column("labels", Value("int64"))
    dev_hf = dev_hf.cast_column("labels", Value("int64"))

    # On verrouille les colonnes utiles au Trainer
    needed_cols = [c for c in ["input_ids", "attention_mask", "labels"] if c in train_hf.column_names]
    train_hf = train_hf.with_format("torch", columns=needed_cols)
    dev_hf = dev_hf.with_format("torch", columns=needed_cols)

    train_hf.set_format(type="torch")
    dev_hf.set_format(type="torch")

    data_collator = DataCollatorWithPadding(tok, padding="longest")

    # Model
    quant_cfg = get_quant_config(args)
    config = AutoConfig.from_pretrained(
        str(model_dir),
        num_labels=n_labels,
        id2label={i: l for i, l in id2label.items()},
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        config=config,
        quantization_config=quant_cfg if quant_cfg is not None else None,
        torch_dtype=torch.bfloat16 if args.load_in_4bit or args.load_in_8bit else None,
        device_map="auto" if (args.load_in_4bit or args.load_in_8bit) else None,
    )

    # pad token id
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    # resize if we added a pad token
    if len(tok) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))

    # === DOC-COMPLIANT LoRA FLOW ===
    # Si modèle quantifié => on DOIT avoir un adapter LoRA
    if (args.load_in_4bit or args.load_in_8bit) and not args.use_lora:
        logging.warning("Quantized model detected; attaching a LoRA adapter as in the PEFT docs.")
        args.use_lora = True

    # Préparation k-bit AVANT d'ajouter l'adapter (pratique QLoRA courante)
    if (args.load_in_4bit or args.load_in_8bit) and prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        # Assurer que les embeddings propagent le gradient si nécessaire
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            emb = model.get_input_embeddings()
            def _hook(_, __, out):
                out.requires_grad_(True)
            emb.register_forward_hook(_hook)

    # Ajout adapter exactement comme la doc
    if args.use_lora:
        model = add_lora_like_doc(model, args)

    # Metrics
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):  # Trainer may return (logits, ...)
            preds = preds[0]
        y_pred = np.argmax(preds, axis=-1)
        y_true = eval_pred.label_ids
        acc = accuracy_score(y_true, y_pred)
        f1 = macro_f1(y_true, y_pred, n_labels)
        return {"accuracy": acc, "macro_f1": f1}

    # TrainingArguments
    metric_for_best = args.eval_metric
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        logging_dir=str(tb_dir),  # NEW: tb_logs sous seed-<seed>
        run_name=f"{args.task_name}-{args.exp_name}-seed{args.seed}",  # utile pour TB/W&B

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        # IMPORTANT : on évalue pour pouvoir sélectionner le best
        eval_strategy="epoch",
        # IMPORTANT : pas de checkpoints automatiques -> on sauve le best via notre callback
        save_strategy="no",
        load_best_model_at_end=False,  # <-- on laisse tomber le reload auto qui cherche pytorch_model.bin
        # pas de save_total_limit nécessaire du coup

        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        report_to=None if args.report_to == "none" else args.report_to,
        bf16=True,
        include_inputs_for_metrics=False,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,

        save_safetensors=False,  # garde la compat avec PEFT + évite soucis de shared tensors
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=dev_hf,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    best_dir = Path(args.output_dir) / "checkpoints" / "best"
    trainer.add_callback(
        SaveBestAdapterCallback(
            metric_name=metric_for_best,  # "macro_f1" ou "accuracy"
            greater_is_better=True,  # cohérent avec ton param
            best_dir=best_dir
        )
    )

    if args.fast_dev_run:
        # Limit to 1 epoch, a couple of steps
        trainer.args.num_train_epochs = 1
        trainer.args.max_steps = 5

    trainer.train(resume_from_checkpoint=None)

    # S’assure que le best existe (utile si fast_dev_run sans eval)
    best_dir.mkdir(parents=True, exist_ok=True)
    if not any(best_dir.iterdir()):
        # pas d'éval => on sauve le modèle courant
        trainer.model.save_pretrained(best_dir, safe_serialization=False)
        tok.save_pretrained(best_dir)
    print(f"Best checkpoint saved to: {best_dir}")

    # Write meta
    meta = {
        "task_name": args.task_name,  # NEW (cohérent runs/)
        "exp_name": args.exp_name,  # NEW
        "model_name": args.model_name,
        "n_labels": int(n_labels),
        "epochs": int(trainer.args.num_train_epochs),
        "batch_size": int(args.batch_size),
        "max_len": int(args.max_len),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "seed": int(args.seed),
        "load_in_4bit": bool(args.load_in_4bit),
        "load_in_8bit": bool(args.load_in_8bit),
        "use_lora": bool(args.use_lora),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target": [t.strip() for t in args.lora_target.split(",") if t.strip()] if args.lora_target else [],
        "best_checkpoint_dir": str(best_dir),
        "tb_log_dir": str(tb_dir),  # NEW
    }
    with open(Path(args.output_dir) / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ========== INFERENCE (même instance Trainer/Model) ========== #
    if args.infer:
        print("Running inference on train/dev/test (same model instance)...")
        labels_order = [id2label[i] for i in range(len(id2label))]

        out_dir = Path(args.output_dir) / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Utilise exactement la même tokenisation que pour train/dev
        def _build_split_ds(split_df: pd.DataFrame):
            ds = HFDataset.from_pandas(split_df.reset_index(drop=True))
            # garder text/label/sent_id/doc_id/split si présents
            keep = [c for c in ["text", "label", "sent_id", "doc_id", "split"] if c in ds.column_names]
            ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
            ds = ds.map(_preprocess, batched=True)
            has_label = "label" in ds.column_names
            if has_label:
                ds = ds.map(_lab, batched=True).remove_columns(["label"])
                # labels int64 pour compat
                from datasets import Value
                ds = ds.cast_column("labels", Value("int64"))
                cols = ["input_ids", "attention_mask", "labels"]
            else:
                cols = ["input_ids", "attention_mask"]
            cols = [c for c in cols if c in ds.column_names]
            ds = ds.with_format("torch", columns=cols)
            return ds, has_label

        # Préparer les splits (train/dev existent déjà; test depuis df)
        splits = [
            ("train", train_df),
            ("dev", dev_df),
            ("test", df[df.split == "test"].copy()),
        ]

        results_paths: Dict[str, str] = {}
        for split_name, split_df in splits:
            if split_df is None or split_df.empty:
                continue

            ds, has_label = _build_split_ds(split_df)
            # predict avec le même trainer/model
            pred_output = trainer.predict(ds)
            logits = pred_output.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            # numpy float32 pour stabilité JSON
            logits = np.asarray(logits, dtype=np.float32)
            probs = _softmax_np(logits.astype(np.float64)).astype(np.float32)
            pred_ids = probs.argmax(axis=1)

            # gold si dispo
            gold_ids = pred_output.label_ids if has_label and pred_output.label_ids is not None else None

            # Récup colonnes meta si présentes, sinon valeurs par défaut
            def _col_safe(col, default):
                return split_df[col].tolist() if col in split_df.columns else [default] * len(pred_ids)

            sent_ids = _col_safe("sent_id", "")
            doc_ids = _col_safe("doc_id", "")
            splits_c = _col_safe("split", split_name)
            texts = _col_safe("text", "")

            rows: List[Dict] = []
            for i in range(len(pred_ids)):
                true_id = int(gold_ids[i]) if gold_ids is not None else -1
                true_label = id2label.get(true_id, "") if true_id >= 0 else ""
                rows.append({
                    "sent_id": str(sent_ids[i]),
                    "doc_id": str(doc_ids[i]),
                    "split": str(splits_c[i]),
                    "text": str(texts[i]),
                    "logits": logits[i].astype(float).tolist(),
                    "probs": probs[i].astype(float).tolist(),
                    "labels": labels_order,
                    "pred_id": int(pred_ids[i]),
                    "pred_label": str(id2label[int(pred_ids[i])]),
                    "true_id": int(true_id),
                    "true_label": true_label,
                })

            out_path = out_dir / f"{split_name}.jsonl"
            _write_jsonl(out_path, rows)
            results_paths[split_name] = str(out_path)
            print(f"[infer] wrote {split_name}: {out_path}")

        # manifest (aligné esprit infer.py)
        manifest = {
            "splits": list(results_paths.keys()),
            "paths": results_paths,
            "label_map": str(args.label_map),
            "ckpt_dir": str(best_dir),  # on a entraîné en RAM, mais on pointe vers le best sauvegardé
            "checkpoint_type": "full",  # on a sauvé un full checkpoint pour ce run
            "base_model": args.model_name,
            "batch_size": int(args.batch_size),
            "max_len": int(args.max_len),
            "labels_order": labels_order,
            "quantization": {
                "load_in_4bit": bool(args.load_in_4bit),
                "load_in_8bit": bool(args.load_in_8bit),
                "device_map_auto": False,  # pas utilisé ici
            },
            "source": "train.py --infer (in-memory model)",
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[infer] manifest: {out_dir / 'manifest.json'}")



    print(f"Best checkpoint saved to: {best_dir}")
    return 0



from transformers import TrainerCallback
import math, os

class SaveBestAdapterCallback(TrainerCallback):
    def __init__(self, metric_name: str, greater_is_better: bool, best_dir: Path):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best = -math.inf if greater_is_better else math.inf
        self.best_dir = best_dir

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.metric_name not in metrics:
            return
        cur = metrics[self.metric_name]
        is_better = cur > self.best if self.greater_is_better else cur < self.best
        if is_better:
            self.best = cur
            self.best_dir.mkdir(parents=True, exist_ok=True)
            model = kwargs["model"]
            # Sauvegarder l’adapter LoRA (PEFT) proprement
            model.save_pretrained(self.best_dir, safe_serialization=False)
            kwargs["tokenizer"].save_pretrained(self.best_dir)
            # empêcher le Trainer d’écrire d’autres checkpoints
            control.should_save = False
            return control


if __name__ == "__main__":
    raise SystemExit(main())
