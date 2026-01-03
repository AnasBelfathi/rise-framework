#!/usr/bin/env python
"""
Minimal training script for smoke-testing BERT-based classifiers on prepared Parquet data
with handy params for *petits tests* (subset, fast_dev_run, limit batches) and tqdm progress bars.

Policy: **keep only the BEST checkpoint** under
  runs/<task_name>/<exp_name>/seed-<seed>/checkpoints/best/
No root-level weights are saved.

Ajouts:
- Structure de sortie fixée: runs/<task_name>/<exp_name>/seed-<seed>/
- Répertoire inference/ créé à côté de checkpoints/
- TensorBoard (rang 0 seulement sur Jean Zay)
"""
try:
    import idr_torch
    on_jean_zay = True
except ImportError:
    on_jean_zay = False


import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
# from transformers import (
#     BertForSequenceClassification,
#     BertTokenizerFast,
#     get_linear_schedule_with_warmup,
# )
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup



from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

from tqdm import tqdm
# Configuration pour Jean Zay (si détecté) — NE PAS MODIFIER
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
    device = get_device(0)

import datasets, os
root_path = os.environ['DSDIR'] + '/HuggingFace_Models'


class SimpleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, label2id):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row.text),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.label2id[str(row.label)]), dtype=torch.long)
        return item


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, scheduler, device, limit_batches: int = 0, disable_tqdm: bool = False):
    model.train()
    total_loss = 0.0
    steps = 0
    progress = tqdm(dataloader, desc="Training", leave=False, disable=disable_tqdm)
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        total_loss += float(loss.item())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        steps += 1
        if limit_batches > 0 and steps >= limit_batches:
            break
    return total_loss / max(1, steps)


def evaluate(model, dataloader, device, limit_batches: int = 0, disable_tqdm: bool = False):
    model.eval()
    total, correct = 0, 0
    steps = 0
    progress = tqdm(dataloader, desc="Evaluating", leave=False, disable=disable_tqdm)
    with torch.no_grad():
        for batch in progress:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            total += int(labels.size(0))
            correct += int((preds == labels).sum().item())
            steps += 1
            if limit_batches > 0 and steps >= limit_batches:
                break
    acc = correct / total if total > 0 else 0.0
    return acc


def main():
    parser = argparse.ArgumentParser()
    # ======== NEW: nommage et structure de sortie ========
    parser.add_argument("--task_name", type=str, required=True, help="ex: biorc, scotus, ...")
    parser.add_argument("--exp_name", type=str, required=True, help="nom d'expérience défini par toi (indépendant du modèle HF)")
    # Data & model
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--label_map", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=Path, required=False, default=None,
                        help="Répertoire de sortie personnalisé. "
                             "Si non fourni : runs/<task>/<exp>/seed-<seed>")
    # Training basics
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    # Small-test helpers
    parser.add_argument("--subset_train", type=int, default=0)
    parser.add_argument("--subset_dev", type=int, default=0)
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_dev_batches", type=int, default=0)
    parser.add_argument("--fast_dev_run", action="store_true")
    # DataLoader perf
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    # UX
    parser.add_argument("--disable_tqdm", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    set_seed(args.seed)

    # ======== Détermination du dossier de sortie ========
    if args.output_dir is None:
        # Valeur par défaut : structure standard
        args.output_dir = Path("runs") / args.task_name / args.exp_name / f"seed-{args.seed}"
    else:
        # Si l’utilisateur fournit un dossier → on l’utilise tel quel
        args.output_dir = Path(args.output_dir) / args.task_name / args.exp_name / f"seed-{args.seed}"

    # Sous-dossiers standard
    ckpt_dir = args.output_dir / "checkpoints"
    best_dir = ckpt_dir / "best"
    infer_dir = args.output_dir / "inference"

    # Création si nécessaire
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)


    # ======== NEW: TensorBoard (uniquement rang 0) ========
    tb_writer = SummaryWriter(log_dir=args.output_dir / "tb_logs") #if is_main else None

    # Load dataset
    df = pd.read_parquet(args.data_path)
    with open(args.label_map, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    n_labels = len(label2id)
    logging.info("Loaded data: %d rows, %d labels", len(df), n_labels)

    train_df = df[df.split == "train"].copy()
    dev_df = df[df.split == "dev"].copy()

    # Apply subsets (deterministic order)
    if args.subset_train and args.subset_train > 0:
        train_df = train_df.head(args.subset_train)
    if args.subset_dev and args.subset_dev > 0:
        dev_df = dev_df.head(args.subset_dev)

    if args.fast_dev_run:
        args.limit_train_batches = 1
        args.limit_dev_batches = 1
        if len(train_df) > 64:
            train_df = train_df.head(64)
        if len(dev_df) > 64:
            dev_df = dev_df.head(64)
        logging.info("fast_dev_run enabled: using 1 batch train/dev and small subsets")

    # tokenizer = BertTokenizerFast.from_pretrained(root_path + '/' + args.model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(root_path + '/' + args.model_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("./models/" + args.model_name, use_fast=True)

    if tokenizer.pad_token is None:  # sécurité pour certains modèles
           tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
           # le modèle utilisera ce pad_id


    train_ds = SimpleDataset(train_df, tokenizer, args.max_len, label2id)
    dev_ds = SimpleDataset(dev_df, tokenizer, args.max_len, label2id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=bool(args.num_workers > 0),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=bool(args.num_workers > 0),
    )

    # (On laisse ta logique device telle quelle plus haut, ici on utilise un handle local)
    device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = BertForSequenceClassification.from_pretrained(
    #     root_path + '/' + args.model_name, num_labels=n_labels
    # ).to(device_local)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(root_path + '/' + args.model_name,
                                                                   num_labels=n_labels)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained("./models/" + args.model_name, num_labels=n_labels)


    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    # >>> manquant auparavant <<<
    model.to(device)

    # Compute steps per epoch accounting for limits
    steps_per_epoch = len(train_loader)
    if args.limit_train_batches and args.limit_train_batches > 0:
        steps_per_epoch = min(steps_per_epoch, args.limit_train_batches)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, steps_per_epoch * max(1, args.epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(0, total_steps // 10), num_training_steps=total_steps
    )

    best_acc = 0.0
    report = {"epochs": []}

    for epoch in range(1, max(1, args.epochs) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device_local,
            limit_batches=int(args.limit_train_batches or 0),
            disable_tqdm=bool(args.disable_tqdm),
        )
        dev_acc = evaluate(
            model,
            dev_loader,
            device_local,
            limit_batches=int(args.limit_dev_batches or 0),
            disable_tqdm=bool(args.disable_tqdm),
        )
        logging.info("Epoch %d: loss=%.4f dev_acc=%.4f", epoch, train_loss, dev_acc)
        report["epochs"].append({"epoch": epoch, "train_loss": train_loss, "dev_acc": dev_acc})

        # TensorBoard scalars
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Accuracy/dev", dev_acc, epoch)

        if dev_acc >= best_acc:
            best_acc = dev_acc
            # Save ONLY the best checkpoint in checkpoints/best
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logging.info("Saved new BEST checkpoint to %s", best_dir)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    # Save training report / meta (rang 0 uniquement)
    report["best_dev_acc"] = best_acc
    with open(args.output_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    meta = {
        "task_name": args.task_name,
        "exp_name": args.exp_name,
        "model_name": args.model_name,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "max_len": int(args.max_len),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "subset_train": int(args.subset_train or 0),
        "subset_dev": int(args.subset_dev or 0),
        "limit_train_batches": int(args.limit_train_batches or 0),
        "limit_dev_batches": int(args.limit_dev_batches or 0),
        "fast_dev_run": bool(args.fast_dev_run),
        "device": str(device_local),
        "n_labels": int(n_labels),
        "best_checkpoint_dir": str(best_dir),
        "tb_log_dir": str(args.output_dir / "tb_logs"),
        "inference_dir": str(infer_dir),
    }
    with open(args.output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
