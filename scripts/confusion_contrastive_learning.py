#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tuning d'embeddings (MultipleNegativesRankingLoss) avec négatifs explicites par instance (colonnes à plat).
- Trainer officiel: SentenceTransformerTrainer
- Dataset HF avec colonnes: anchor, positive, negative_1, …, negative_K
- Détection Jean Zay (sélection GPU locale)
- Chargement modèle via DSDIR / chemin local / nom HF
- Éval macro-F1 / accuracy sur DEV & TEST à chaque epoch (callback) + logs TensorBoard

Entrées:
  confusion-aware-datasets/<task>/<exp>/{train,dev,test}.jsonl
  Colonnes requises: text, true_label  (+ optionnel: confusion_labels)

Sortie:
  runs/<task>/<exp>/seed-<seed>/checkpoints/trainer/
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    util,
)
from transformers import TrainerCallback
import torch
import os

# ---------- Métriques & éval retrieval ----------
def macro_f1(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    if not labels: return float("nan")
    f1s = []
    for c in labels:
        tp = sum((yt==c and yp==c) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt!=c and yp==c) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt==c and yp!=c) for yt, yp in zip(y_true, y_pred))
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        f1s.append(f1)
    return float(np.mean(f1s))

def eval_retrieval(model, df, label_list, label_template, batch_size):
    if df is None or df.empty or not label_list:
        return {"acc": float("nan"), "macro_f1": float("nan"), "n": 0}
    sents = df["text"].astype(str).tolist()
    y_true = df["true_label"].astype(str).tolist()
    sent_emb = model.encode(sents, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    lab_texts = [label_template.replace("{label}", str(l)) for l in label_list]
    lab_emb = model.encode(lab_texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    sims = model.similarity(sent_emb, lab_emb) if hasattr(model, "similarity") else util.cos_sim(sent_emb, lab_emb)
    pred_idx = torch.argmax(sims, dim=1).cpu().numpy().tolist()
    y_pred = [label_list[i] for i in pred_idx]
    acc = float(np.mean([int(a==b) for a,b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred)
    return {"acc": acc, "macro_f1": mf1, "n": len(y_true)}

class RetrievalEvalCallback(TrainerCallback):
    """Évalue DEV & TEST en fin d’epoch (macro-F1 + acc), print & log TensorBoard."""
    def __init__(self, df_dev, df_test, label_list, label_template, batch_size, is_main=True):
        self.df_dev = df_dev
        self.df_test = df_test
        self.label_list = label_list
        self.label_template = label_template
        self.batch_size = batch_size
        self.is_main = is_main
    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return control
        ep = int(state.epoch) if state.epoch is not None else -1
        if self.df_dev is not None and not self.df_dev.empty:
            m_dev = eval_retrieval(trainer.model, self.df_dev, self.label_list, self.label_template, self.batch_size)
            trainer.log({"epoch": ep, "dev_macro_f1": m_dev["macro_f1"], "dev_acc": m_dev["acc"], "dev_n": m_dev["n"]})
            if self.is_main:
                print(f"[Epoch {ep}] DEV:  macro_f1={m_dev['macro_f1']:.4f}  acc={m_dev['acc']:.4f}  n={m_dev['n']}")
        if self.df_test is not None and not self.df_test.empty:
            m_test = eval_retrieval(trainer.model, self.df_test, self.label_list, self.label_template, self.batch_size)
            trainer.log({"epoch": ep, "test_macro_f1": m_test["macro_f1"], "test_acc": m_test["acc"], "test_n": m_test["n"]})
            if self.is_main:
                print(f"[Epoch {ep}] TEST: macro_f1={m_test['macro_f1']:.4f}  acc={m_test['acc']:.4f}  n={m_test['n']}")
        return control

# ---------- Jean Zay (sélection GPU local si présent) ----------
try:
    import idr_torch  # dispo sur Jean Zay
    torch.cuda.set_device(idr_torch.local_rank)
    device_info = f"Jean Zay: rank={idr_torch.rank} local_rank={idr_torch.local_rank}"
    is_main = (idr_torch.rank == 0)
except Exception:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device_info = "Local: cuda:0"
    else:
        device_info = "CPU"
    is_main = True

# ---------- Utils ----------
def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return pd.DataFrame(rows)

# --- Négatifs: distribution rivale depuis confusion_labels (fallback uniforme) ---
def build_rival_probs_from_df(df: pd.DataFrame, all_labels):
    label_set = set(map(str, all_labels))
    reps = (df.dropna(subset=["true_label"])
              .groupby("true_label", as_index=False)
              .first()[["true_label","confusion_labels"]])
    mapping = {}
    for _, row in reps.iterrows():
        y = str(row["true_label"])
        dist = []
        cl = row.get("confusion_labels", None)
        if isinstance(cl, list) and cl:
            for it in cl:
                try:
                    r = str(it["label"]); p = float(it["percentage"])
                except Exception:
                    continue
                if p <= 0 or r == y or r not in label_set:
                    continue
                dist.append((r, p))
        if dist:
            s = sum(p for _, p in dist)
            mapping[y] = [(r, p/s) for r, p in dist] if s>0 else []
    for y in map(str, all_labels):
        if y not in mapping:
            others = [str(l) for l in all_labels if str(l) != y]
            mapping[y] = [(l, 1.0/len(others)) for l in others] if others else []
    return mapping

def sample_negative_label(y, rival_probs, rng):
    cand = rival_probs.get(str(y), [])
    if not cand: return None
    labels, probs = zip(*cand)
    return rng.choices(labels, weights=probs, k=1)[0]

# =========================
# NEW: helpers d'inférence
# =========================
def save_label_embeddings(model, label_list, label_template, batch_size, infer_dir: Path):
    lab_texts = [label_template.replace("{label}", str(l)) for l in label_list]
    lab_emb = model.encode(lab_texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    np.save(infer_dir / "label_embeddings.npy", lab_emb.detach().cpu().numpy())
    with (infer_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": label_list, "label_template": label_template}, f, ensure_ascii=False, indent=2)

def save_split_embeddings(model, df: pd.DataFrame, split_name: str, batch_size: int, infer_dir: Path):
    if df is None or df.empty: return 0
    texts = df["text"].astype(str).tolist()
    embs = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    np.save(infer_dir / f"{split_name}_sentence_embeddings.npy", embs.detach().cpu().numpy())
    out_jsonl = infer_dir / f"{split_name}_texts.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({
                "doc_id": r.get("doc_id", None),
                "sentence_id": r.get("sentence_id", None),
                "text": str(r["text"]),
                "true_label": str(r["true_label"]),
            }, ensure_ascii=False) + "\n")
    return len(texts)

def run_inference_and_print(model, df_dev, df_test, label_list, label_template, batch_size, infer_dir: Path):
    infer_dir.mkdir(parents=True, exist_ok=True)
    save_label_embeddings(model, label_list, label_template, batch_size, infer_dir)
    n_dev  = save_split_embeddings(model, df_dev,  "dev",  batch_size, infer_dir)
    n_test = save_split_embeddings(model, df_test, "test", batch_size, infer_dir)
    if df_dev is not None and not df_dev.empty:
        m_dev = eval_retrieval(model, df_dev, label_list, label_template, batch_size)
        print(f"[INFER] DEV : macro_f1={m_dev['macro_f1']:.4f}  acc={m_dev['acc']:.4f}  n={m_dev['n']}  (embs sauvés: {n_dev})")
    if df_test is not None and not df_test.empty:
        m_test = eval_retrieval(model, df_test, label_list, label_template, batch_size)
        print(f"[INFER] TEST: macro_f1={m_test['macro_f1']:.4f}  acc={m_test['acc']:.4f}  n={m_test['n']}  (embs sauvés: {n_test})")

def main():
    ap = argparse.ArgumentParser()
    # chemins / config basiques
    ap.add_argument("--task_name", type=str, required=True)
    ap.add_argument("--exp_name", type=str, required=True)
    ap.add_argument("--data_root", type=Path, default=Path("confusion-aware-datasets"))

    # modèle & FT
    ap.add_argument("--model_name", type=str, default="microsoft/mpnet-base")
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--label_template", type=str, default="LABEL: {label}")
    ap.add_argument("--seed", type=int, default=42)

    # hyperparams → TrainingArguments
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--no_amp", action="store_true", help="désactive fp16 si True")
    ap.add_argument("--scale", type=float, default=20.0, help="température pour MultipleNegativesRankingLoss")

    # négatifs explicites
    ap.add_argument("--num_negatives_per_anchor", type=int, default=3,
                    help="nb de négatifs par ancre → crée les colonnes negative_1..negative_K")

    # petits tests
    ap.add_argument("--fast_dev_run", action="store_true", help="tronque train à 64 ex. pour un test rapide")
    ap.add_argument("--subset_train", type=int, default=0, help="garde N premières lignes de train (0=all)")

    # NEW: mode inférence
    ap.add_argument("--infer", action="store_true",
                    help="si présent: pas d'entraînement; charge --ckpt_dir (ou dossier de sortie) et fait l'inférence")
    ap.add_argument("--ckpt_dir", type=Path, default=None,
                    help="chemin du checkpoint à charger en mode --infer (par défaut: dossier de sortie)")

    args = ap.parse_args()
    set_seed_all(args.seed)
    rng = random.Random(args.seed)

    if args.exp_name.strip().lower() == "bge-m3":
        exp_dir_for_data = "bert-base-uncased"
        if is_main:
            print("[data] exp_name='bge-m3' → lecture des JSONL depuis exp='bert-base-uncased'")

        # I/O
        split_dir = args.data_root / args.task_name / exp_dir_for_data

    else:
        split_dir = args.data_root / args.task_name / args.exp_name

    train_path = split_dir / "train.jsonl"
    dev_path   = split_dir / "dev.jsonl"
    test_path  = split_dir / "test.jsonl"
    run_root = Path("finetuning-multiple-negatives") / args.task_name / args.exp_name / f"seed-{args.seed}"
    out_dir = run_root / "checkpoints" / "trainer"
    tb_dir  = run_root / "tb"
    # NEW: répertoire d'inférence
    infer_dir = run_root / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device_info}")
    print(f"[data] train={train_path}")

    # --- charge train ---
    if not train_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {train_path}")
    df = read_jsonl(train_path)
    if df.empty:
        raise RuntimeError("Train DataFrame vide.")

    # garde uniquement text / true_label (+ confusion_labels si présent)
    need = {"text", "true_label"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Colonnes manquantes dans train.jsonl: {missing}")
    df = df.dropna(subset=["text", "true_label"]).reset_index(drop=True)

    # petits tests
    if args.fast_dev_run and len(df) > 64:
        df = df.head(64)
        print("[fast_dev_run] train tronqué à 64 ex.")
    if args.subset_train and args.subset_train > 0:
        df = df.head(args.subset_train)
        print(f"[subset_train] {len(df)} lignes")

    if df.empty:
        raise RuntimeError("Train DataFrame vide après filtrage/troncature.")

    # --- Labels (ordre d'apparition train→dev→test) ---
    def add_labels_from(df_, dst):
        if df_ is None or df_.empty: return dst
        for l in df_["true_label"].astype(str).tolist():
            if l not in dst: dst.append(l)
        return dst

    label_list = []
    add_labels_from(df, label_list)

    # DFs dev/test pour macro-F1
    df_dev = read_jsonl(dev_path) if dev_path.exists() else None
    if df_dev is not None:
        if df_dev.empty or not all(c in df_dev.columns for c in ["text","true_label"]):
            df_dev = None; print("[dev] ignoré (fichier vide ou colonnes manquantes).")
        else:
            df_dev = df_dev.dropna(subset=["text","true_label"]).reset_index(drop=True)
            add_labels_from(df_dev, label_list)
            print(f"[dev]  {len(df_dev)} lignes.")
    df_test = read_jsonl(test_path) if test_path.exists() else None
    if df_test is not None:
        if df_test.empty or not all(c in df_test.columns for c in ["text","true_label"]):
            df_test = None; print("[test] ignoré (fichier vide ou colonnes manquantes).")
        else:
            df_test = df_test.dropna(subset=["text","true_label"]).reset_index(drop=True)
            add_labels_from(df_test, label_list)
            print(f"[test] {len(df_test)} lignes.")

    if len(label_list) < 2:
        raise RuntimeError("Au moins 2 labels sont requis pour construire des négatifs.")

    # --- Chargement modèle (priorité DSDIR → ./berts/models → nom HF) ---
    model_paths = []
    dsd = os.environ.get("DSDIR", None)
    if dsd is not None:
        model_paths.append(str(Path(dsd) / "HuggingFace_Models" / args.model_name))
    model_paths.append(str(Path("./berts/models") / args.model_name))
    model_paths.append(args.model_name)

    last_err = None
    model = None
    for p in model_paths:
        try:
            model = SentenceTransformer(p)
            print(f"[model] loaded from: {p}")
            break
        except Exception as e:
            last_err = e
            continue
    if model is None:
        raise RuntimeError(f"Impossible de charger le modèle depuis {model_paths}. Dernière erreur: {last_err}")

    # --- Compat shim: retirer 'task' si non supporté par tokenize() ---
    import inspect
    from types import MethodType
    try:
        sig = inspect.signature(model.tokenize)
        if "task" not in sig.parameters:
            _orig_tokenize = model.tokenize
            def _tokenize_no_task(self, texts, **kwargs):
                kwargs.pop("task", None)
                return _orig_tokenize(texts, **kwargs)
            model.tokenize = MethodType(_tokenize_no_task, model)
            if is_main:
                print("[compat] model.tokenize ne supporte pas 'task' → wrapper actif.")
    except Exception:
        pass
    try:
        first = model[0]
        if hasattr(first, "tokenize"):
            sig0 = inspect.signature(first.tokenize)
            if "task" not in sig0.parameters:
                _orig_first_tok = first.tokenize
                def _first_tokenize_no_task(texts, **kwargs):
                    kwargs.pop("task", None)
                    return _orig_first_tok(texts, **kwargs)
                first.tokenize = _first_tokenize_no_task
                if is_main:
                    print("[compat] model[0].tokenize ne supporte pas 'task' → wrapper actif.")
    except Exception:
        pass

    try:
        model.max_seq_length = int(args.max_seq_len)
        print(f"[model] max_seq_length={model.max_seq_length}")
    except Exception:
        pass

    # =========================
    # NEW: mode inférence seule
    # =========================
    if args.infer:
        ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else out_dir

        def _load_st_dir(path_like: Path):
            try:
                return SentenceTransformer(str(path_like))
            except Exception:
                cands = sorted([d for d in Path(path_like).glob("checkpoint-*") if d.is_dir()])
                if cands:
                    return SentenceTransformer(str(cands[-1]))
                raise

        print(f"[infer] chargement checkpoint depuis: {ckpt_dir}")
        model = _load_st_dir(ckpt_dir)

        # re-applique le petit wrapper 'task' si besoin
        try:
            sig = inspect.signature(model.tokenize)
            if "task" not in sig.parameters:
                _orig_tokenize = model.tokenize
                def _tokenize_no_task(self, texts, **kwargs):
                    kwargs.pop("task", None)
                    return _orig_tokenize(texts, **kwargs)
                model.tokenize = MethodType(_tokenize_no_task, model)
        except Exception:
            pass
        try:
            first = model[0]
            if hasattr(first, "tokenize"):
                sig0 = inspect.signature(first.tokenize)
                if "task" not in sig0.parameters:
                    _orig_first_tok = first.tokenize
                    def _first_tokenize_no_task(texts, **kwargs):
                        kwargs.pop("task", None)
                        return _orig_first_tok(texts, **kwargs)
                    first.tokenize = _first_tokenize_no_task
        except Exception:
            pass

        # inférence + sauvegarde
        run_inference_and_print(
            model=model,
            df_dev=df_dev,
            df_test=df_test,
            label_list=label_list,
            label_template=args.label_template,
            batch_size=args.batch_size,
            infer_dir=infer_dir
        )
        print(f"[OK] Inférence terminée. Fichiers dans: {infer_dir}")
        return

    # --- Construction dataset: anchor, positive, negative_1..negative_K ---
    rival_probs = build_rival_probs_from_df(df, all_labels=label_list)

    anchors = []
    positives = []
    K = max(1, int(args.num_negatives_per_anchor))
    neg_cols = [f"negative_{i+1}" for i in range(K)]
    neg_data = {col: [] for col in neg_cols}

    for _, row in df.iterrows():
        anchor_text = str(row["text"])
        y_true = str(row["true_label"])
        pos_text = args.label_template.replace("{label}", y_true)

        others = [l for l in label_list if l != y_true]
        if not others:
            continue

        chosen_negs = []
        tried = 0
        while len(chosen_negs) < K and tried < K*5:
            neg_label = sample_negative_label(y_true, rival_probs, rng)
            if neg_label is None or neg_label == y_true or neg_label not in others:
                neg_label = rng.choice(others)
            neg_text = args.label_template.replace("{label}", str(neg_label))
            if neg_text not in chosen_negs:
                chosen_negs.append(neg_text)
            tried += 1
        # Si on n'a pas assez de distincts, on complète en tirant au hasard
        while len(chosen_negs) < K:
            neg_label = rng.choice(others)
            chosen_negs.append(args.label_template.replace("{label}", str(neg_label)))

        anchors.append(anchor_text)
        positives.append(pos_text)
        for col, neg_txt in zip(neg_cols, chosen_negs):
            neg_data[col].append(neg_txt)

    if not anchors:
        raise RuntimeError("Aucune instance générée (vérifie les labels/négatifs).")

    data_dict = {"anchor": anchors, "positive": positives}
    data_dict.update(neg_data)
    train_ds = Dataset.from_dict(data_dict)
    print(f"[train] {len(train_ds)} instances (colonnes: {', '.join(['anchor','positive']+neg_cols)}).")

    # --- Loss: MultipleNegativesRankingLoss (avec scale/température) ---
    loss = losses.MultipleNegativesRankingLoss(model=model, scale=float(args.scale))

    # --- TrainingArguments ---
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=(not args.no_amp),
        seed=args.seed,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,

        eval_strategy="no",     # éval custom via callback
        report_to=["tensorboard"],
        logging_dir=str(tb_dir),

        dataloader_drop_last=False,
    )

    # --- Trainer ---
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=loss,
    )

    # Callback: macro-F1 / acc DEV & TEST à chaque epoch
    trainer.add_callback(
        RetrievalEvalCallback(
            df_dev=df_dev,
            df_test=df_test,
            label_list=label_list,
            label_template=args.label_template,
            batch_size=args.batch_size,
            is_main=is_main,
        )
    )

    # Entraînement
    trainer.train()

    # NEW: inférence + sauvegarde des représentations après FT
    run_inference_and_print(
        model=trainer.model,
        df_dev=df_dev,
        df_test=df_test,
        label_list=label_list,
        label_template=args.label_template,
        batch_size=args.batch_size,
        infer_dir=infer_dir
    )

    # Évals finales (PRINT) — on garde ton bloc d'origine
    if df_dev is not None:
        m_dev = eval_retrieval(trainer.model, df_dev, label_list, args.label_template, args.batch_size)
        print(f"[FINAL] DEV:  macro_f1={m_dev['macro_f1']:.4f}  acc={m_dev['acc']:.4f}  n={m_dev['n']}")
    if df_test is not None:
        m_test = eval_retrieval(trainer.model, df_test, label_list, args.label_template, args.batch_size)
        print(f"[FINAL] TEST: macro_f1={m_test['macro_f1']:.4f}  acc={m_test['acc']:.4f}  n={m_test['n']}")

    print(f"[OK] entraînement (MNRL avec négatifs explicites à plat) terminé. Checkpoints dans: {out_dir}")
    if is_main:
        print(f"[TB] tensorboard --logdir \"{tb_dir}\"")
        print(f"[INFER] fichiers d'embeddings dans: {infer_dir}")

if __name__ == "__main__":
    main()
