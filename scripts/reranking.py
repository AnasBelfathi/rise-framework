#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v1 — HARD-only:
- Baseline (argmax logits)
- Rerank (argmax logits × cos) appliqué UNIQUEMENT aux HARD
- τ appris sur DEV: moyenne des confiances (softmax sur logits) des prédictions incorrectes
- Perf hybride: EASY=baseline, HARD=rerank
- Affiche exemples influencés / améliorés / dégradés (sur lesquels le rerank a été appliqué)


python overall_results.py \
  --task scotus-rhetorical_function \
  --exp roberta-base \
  --seed 1 --gather


"""

from pathlib import Path
import argparse
import json
import numpy as np
import random
import sys
import re
from typing import Optional




def discover_pairs_for_seed(seed: str):
    """
    Retourne toutes les paires (task, exp) qui ont un TEST pour seed donné,
    en scannant berts__... et llms__... (on ne change pas la logique d'évaluation).
    """
    pairs = set()
    for p in OUT_DIR.glob(f"*__*__*__seed-{seed}__test.jsonl"):
        parts = p.name.split("__")
        if len(parts) < 5:
            continue
        kind, task, exp = parts[0], parts[1], parts[2]
        pairs.add((task, exp))
    return sorted(pairs)

def emb_source_for_exp(exp: str, cli_emb_exp: Optional[str] = None):
    """
    Source d'embeddings à utiliser:
    - si l'utilisateur a fourni --emb-exp : on respecte.
    - sinon: pour modèles génératifs → 'bge-m3', sinon = exp.
    """
    GENERATIVE_MODELS = ["Mistral-7B-v0.3", "Meta-Llama-3-8B", "Qwen3-8B"]
    if cli_emb_exp is not None:
        return cli_emb_exp
    return "bge-m3" if exp in GENERATIVE_MODELS else exp


# ----- Mapping colonnes pour reproduire EXACTEMENT l'entête LaTeX -----
LEGAL_COLUMNS = [
    ("scotus-category",            r"\textsc{Scotus}\textsubscript{Category}"),
    ("scotus-rhetorical_function", r"\textsc{Scotus}\textsubscript{RF}"),
    ("scotus-steps",               r"\textsc{Scotus}\textsubscript{Steps}"),
    ("legal-eval",                 r"\textsc{LegalEval}"),
    ("DeepRhole",                  r"\textsc{DeepRhole}"),
]
MED_COLUMNS = [
    ("PubMed_20k_RCT",             r"\textsc{PubMed}"),
    ("biorc",                      r"\textsc{BioRC}"),
]
SCI_COLUMNS = [
    ("csabstracts",                r"\textsc{CS-Abstracts}"),
]
TASK_COLUMNS = LEGAL_COLUMNS + MED_COLUMNS + SCI_COLUMNS  # ordre figé

def _fmt_cell(x):
    import numpy as _np
    if x is None or (isinstance(x, float) and (_np.isnan(x) or _np.isinf(x))):
        return ""
    return f"{100.0 * x:.2f}".rstrip("0").rstrip(".")

def _render_latex_table(rows):
    from collections import defaultdict as _dd
    import numpy as _np

    table = _dd(lambda: _dd(dict))
    models = set()
    for r in rows:
        exp = r["exp"]; models.add(exp)
        mode = r["mode"]
        table[exp][mode][r["task"]] = (r["macro_all"], r["weighted_all"])

    models = sorted(models)

    header = []
    header.append(r"\begin{table*}[ht]")
    header.append(r"\centering")
    header.append(r"\small")
    header.append(r"\arrayrulecolor{gray!30}")
    header.append(r"\rowcolors{2}{white}{gray!08}")
    header.append(r"\resizebox{\linewidth}{!}{")

    nb_cols = 1 + 2 * (len(TASK_COLUMNS) + 1)  # +1 pour Average
    header.append(r"\begin{tabular}{>{\bfseries}l" + "c" * (nb_cols - 1) + "}")
    header.append(r"\toprule")

    header.append(
        r"& \multicolumn{10}{c}{\includegraphics[height=1.2em]{images/legal.png}~~\textbf{Legal}}"
        r" & \multicolumn{4}{c}{\includegraphics[height=1.2em]{images/medical.png}~~\textbf{Medical}}"
        r" & \multicolumn{2}{c}{\includegraphics[height=1.2em]{images/science.png}~~\textbf{Scientific}}"
        r" & \multicolumn{2}{c}{\multirow{2}{*}{\textbf{Average}}} \\"
    )
    header.append(r"\cmidrule(lr){2-11} \cmidrule(lr){12-15} \cmidrule(lr){16-17}")

    sub = [f"\\multicolumn{{2}}{{c}}{{{disp}}}" for (_, disp) in TASK_COLUMNS]
    sub.append(r"\multicolumn{2}{c}{}")  # placeholder Average
    header.append(" & " + " & ".join(sub) + r" \\")

    # cmidrules (inclure aussi Average = 18-19)
    cmis, start = [], 2
    for _ in TASK_COLUMNS:
        cmis.append((start, start + 1)); start += 2
    cmis.append((start, start + 1))  # Average
    header.append(" ".join([rf"\cmidrule(lr){{{i}-{j}}}" for (i, j) in cmis]))

    header.append(" & " + " & ".join(["mF1 & wF1"] * (len(TASK_COLUMNS) + 1)) + r" \\")
    header.append(r"\midrule")

    def _avg_for(exp, mode):
        ms = [table[exp].get(mode, {}).get(t, (None, None))[0] for t, _ in TASK_COLUMNS]
        ws = [table[exp].get(mode, {}).get(t, (None, None))[1] for t, _ in TASK_COLUMNS]
        ms = [v for v in ms if v is not None and _np.isfinite(v)]
        ws = [v for v in ws if v is not None and _np.isfinite(v)]
        avg_m = float(_np.mean(ms)) if ms else None
        avg_w = float(_np.mean(ws)) if ws else None
        return avg_m, avg_w

    lines = []
    for exp in models:
        # -------- baseline row: first cell = model name --------
        vals = []
        for task_key, _ in TASK_COLUMNS:
            m, w = table[exp].get("baseline", {}).get(task_key, (None, None))
            vals.extend([_fmt_cell(m), _fmt_cell(w)])

        avg_m, avg_w = _avg_for(exp, "baseline")
        vals.extend([_fmt_cell(avg_m), _fmt_cell(avg_w)])

        lines.append(exp + "  & " + " & ".join(vals) + r" \\")

        # -------- rerank row: first cell = "+ Rerank" --------
        vals = []
        for task_key, _ in TASK_COLUMNS:
            m, w = table[exp].get("rerank", {}).get(task_key, (None, None))
            vals.extend([_fmt_cell(m), _fmt_cell(w)])

        avg_m, avg_w = _avg_for(exp, "rerank")
        vals.extend([_fmt_cell(avg_m), _fmt_cell(avg_w)])

        lines.append(r"+ Rerank  & " + " & ".join(vals) + r" \\")
        lines.append("")

    footer = []
    footer.append(r"\bottomrule")
    footer.append(r"\end{tabular}")
    footer.append("}")
    footer.append(r"\caption{Global Macro-F1 and Weighted-F1 on \textbf{TEST} for baseline and HARD-only rerank (EASY=baseline, HARD=logit$\times$cos; $\tau$ from DEV).}")
    footer.append(r"\label{tab:hard-only-hybrid}")
    footer.append(r"\vspace{-1.3em}")
    footer.append(r"\end{table*}")

    latex = "\n".join(header + lines + footer)
    print("\n==================== BLOC LATEX GÉNÉRÉ ====================\n")
    print(latex)
    return latex




# --------- CONFIG DE BASE ---------
# Tu peux remplacer par un argument --root si tu veux l’adapter à Colab/Drive
ROOT = Path("..")
OUT_DIR = ROOT / "outputs"
EMB_ROOT = ROOT / "finetuning-multiple-negatives"
EPS = 1e-12
RANDOM_SEED = 13

CANDIDATE_SID_KEYS = ("sent_id", "sid", "id", "sentence_id", "sentid", "global_sent_id", "index", "i")
SID_RE = re.compile(r"^(?P<doc>.+?)#(?P<num>\d+)$")

def set_seed(s=RANDOM_SEED):
    random.seed(s)
    np.random.seed(s)

# --------- UTILS GÉNÉRIQUES ---------
def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def normalize_sid(sid: str) -> str:
    if sid is None:
        return ""
    s = str(sid).strip()
    m = SID_RE.match(s)
    if m:
        doc = m.group("doc"); num = int(m.group("num"))
        return f"{doc}#{num}"
    return s

def sid_variants(s: str):
    out = set()
    n = normalize_sid(s)
    out.add(n)
    m = SID_RE.match(n)
    if m:
        doc = m.group("doc"); num = int(m.group("num"))
        out.add(f"{doc}#{num:05d}")
    return out

def build_sent_id(row, idx: int) -> str:
    for k in CANDIDATE_SID_KEYS:
        if k in row and row[k] not in (None, ""):
            return str(row[k])
    doc = row.get("doc_id")
    sid_field = row.get("sentence_id")
    if doc is not None and sid_field is not None:
        try:
            return f"{doc}#{int(sid_field)}"
        except Exception:
            pass
    return f"__idx__#{idx:06d}"

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + EPS)
    return a @ b.T

def topk(vec: np.ndarray, k=3):
    k = min(k, len(vec))
    idx = np.argpartition(-vec, kth=k-1)[:k]
    idx = idx[np.argsort(-vec[idx])]
    return idx, vec[idx]

def softmax_logits(L: np.ndarray) -> np.ndarray:
    x = L - np.max(L)
    e = np.exp(x)
    return e / (np.sum(e) + EPS)



def topk_logit_variance(L: np.ndarray, k: int = 7) -> float:
    """
    v(x) = Var(Top-k(logits)).
    Si L est None: retourne NaN.
    """
    if L is None:
        return float("nan")
    L = np.asarray(L, dtype=np.float64)
    kk = min(k, L.shape[0])
    # Top-k sans trier complètement
    topk_vals = np.partition(L, -kk)[-kk:]
    return float(np.var(topk_vals, ddof=0))


# --------- MÉTRIQUES ---------
def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not labels:
        return float("nan")
    f1s = []
    for c in labels:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2*prec*rec/(prec+rec))
    return float(np.mean(f1s))

def weighted_f1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not labels:
        return float("nan")
    N = len(y_true)
    total = 0.0
    for c in labels:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        supp = int((y_true == c).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1c = 0.0 if (prec + rec) == 0 else 2*prec*rec/(prec+rec)
        total += f1c * (supp / N if N > 0 else 0.0)
    return float(total)

# --------- CHARGEMENT OUTPUTS ---------
def load_outputs_split(task: str, exp: str, seed: str, split: str):
    # NOTE: nommage 'seed-{seed}' pour rester compatible avec ton repo
    GENERATIVE_MODELS = ["Mistral-7B-v0.3", "Meta-Llama-3-8B", "Qwen3-8B"]
    if exp not in GENERATIVE_MODELS:
        p = OUT_DIR / f"berts__{task}__{exp}__seed-{seed}__{split}.jsonl"
    else:
        p = OUT_DIR / f"llms__{task}__{exp}__seed-{seed}__{split}.jsonl"

    if not p.exists():
        return None, None
    rows = list(load_jsonl(p))
    if not rows:
        return None, None
    labels = rows[0].get("labels", None)
    data = {}
    for r in rows:
        sid = r.get("sent_id")
        if sid is None:
            continue
        true_id = r.get("true_id")
        if r.get("logits") is not None:
            L = np.asarray(r["logits"], dtype=np.float64)
            probs = None
        elif r.get("probs") is not None:
            P = np.asarray(r["probs"], dtype=np.float64)
            L = None
            probs = P
        else:
            # si rien, skip
            continue
        data[str(sid)] = dict(true=int(true_id) if true_id is not None else -1,
                              logits=L, probs=probs)
    return labels, data

# --------- CHARGEMENT EMBEDDINGS ---------
def load_embeddings(task: str, emb_exp: str, emb_seed: str):
    base = EMB_ROOT / task / emb_exp / emb_seed / "inference"
    if not base.exists():
        root = EMB_ROOT / task / emb_exp
        if not root.exists():
            sys.exit(f"[STOP] Embeddings introuvables: {base}")
        seed_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "inference").exists()]
        if not seed_dirs:
            sys.exit(f"[STOP] Aucun dossier seed dans: {root}")
        base = seed_dirs[0] / "inference"
        print(f"[INFO] emb_seed '{emb_seed}' absent → fallback: {base.parent.name}")

    labels_path = base / "labels.json"
    lab_emb = json.load(labels_path.open("r", encoding="utf-8")).get("labels", None)
    if not lab_emb:
        sys.exit(f"[STOP] 'labels.json' mal formé: {labels_path}")

    E_L = np.load(base / "label_embeddings.npy")          # (C, D)
    E_S = np.load(base / "test_sentence_embeddings.npy")  # (N, D)
    rows = list(load_jsonl(base / "test_texts.jsonl"))    # N lignes

    if len(rows) != len(E_S):
        sys.exit("[STOP] Mismatch len(test_texts) vs len(test_sentence_embeddings).")

    sid2idx = {}
    for i, r in enumerate(rows):
        sid = build_sent_id(r, i)
        for v in sid_variants(sid):
            sid2idx[v] = i

    return dict(labels=lab_emb, E_L=E_L, E_S=E_S, sid2idx=sid2idx)

def align_label_order(labels_logits, labels_emb, E_L):
    if labels_logits is None:
        return labels_emb, E_L
    if list(labels_logits) == list(labels_emb):
        return labels_emb, E_L
    pos = {lab: i for i, lab in enumerate(labels_emb)}
    idx = []
    for lab in labels_logits:
        if lab not in pos:
            raise RuntimeError(f"Label '{lab}' absent dans labels.json (embeddings).")
        idx.append(pos[lab])
    idx = np.asarray(idx, dtype=int)
    return list(labels_logits), E_L[idx]

def compute_tau_from_dev(dev_data: dict, k: int = 7) -> float:
    """
    τ = moyenne de v(x)=Var(Top-k(logits)) sur les cas incorrects en DEV.
    Règle LRSL: hard si v(x) < τ.
    Fallback: moyenne globale des v(x) en DEV si aucun incorrect.
    NOTE: nécessite logits; si logits absents, on ignore l'exemple pour le calcul de τ.
    """
    if not dev_data:
        return None

    vals_incorrect = []
    vals_all = []

    for sid, obj in dev_data.items():
        L = obj.get("logits", None)
        if L is None:
            # pas de logits => on ne peut pas calculer Var(Top-k(logits)) proprement
            continue

        L = np.asarray(L, dtype=np.float64)
        pb = int(np.argmax(L))  # baseline pred
        true_id = int(obj.get("true", -1))

        v = topk_logit_variance(L, k=k)
        if not np.isfinite(v):
            continue

        vals_all.append(v)
        if true_id >= 0 and pb != true_id:
            vals_incorrect.append(v)

    if vals_incorrect:
        return float(np.mean(vals_incorrect))
    if vals_all:
        return float(np.mean(vals_all))
    return None





def evaluate_one_pair(task, exp, seed, emb_exp, emb_seed,
                      show_k_improved=0, show_k_influenced=0, show_k_degraded=0, debug_k=0):
    """
    Exécute EXACTEMENT ta logique actuelle pour (task, exp, seed):
      - τ depuis DEV (fallback TEST)
      - HARD = conf <= τ
      - Hybrid = logits sur EASY, logit×cos sur HARD
      - Retourne les métriques baseline/hybrid.
    Les affichages d'exemples peuvent être coupés (k=0) en mode collecte.
    """
    # 0) DEV pour τ
    _, dev_data = load_outputs_split(task, exp, seed, "dev")
    tau = compute_tau_from_dev(dev_data) if dev_data else None

    # TEST
    labels_logits, test_data = load_outputs_split(task, exp, seed, "test")
    if not test_data:
        raise RuntimeError(f"Aucune donnée TEST pour {task}/{exp}/seed-{seed}")
    if tau is None:
        tau = compute_tau_from_dev(test_data)
        if tau is None:
            tau = 0.5
            print(f"[WARN] τ par défaut = 0.5 pour {task}/{exp}/seed-{seed}")

    # Embeddings
    emb = load_embeddings(task, emb_exp, emb_seed)
    labels_emb, E_L, E_S, sid2idx = emb["labels"], emb["E_L"], emb["E_S"], emb["sid2idx"]

    # Alignement
    try:
        _, E_L_aligned = align_label_order(labels_logits, labels_emb, E_L)
    except Exception as e:
        raise RuntimeError(f"Incompatibilité labels {task}/{exp}: {e}")

    # Prédictions + HARD
    y_true, y_pred_base, y_pred_hybrid = [], [], []
    used_sids, rerank_applied = [], []
    K_TOPK_VAR = 7  # tu peux le mettre en config globale si tu veux

    # ...
    for sid, obj in test_data.items():
        true_id = int(obj.get("true", -1))
        if true_id < 0:
            continue

        L, P = obj.get("logits"), obj.get("probs")

        # baseline pred (inchangé)
        if L is not None:
            pb = int(np.argmax(L))
            v = topk_logit_variance(L, k=K_TOPK_VAR)
        else:
            probs = np.asarray(P, dtype=np.float64)
            pb = int(np.argmax(probs))
            v = float("nan")  # pas de logits => pas de variance fiable

        ph = pb
        applied = False

        # HARD si v < tau (LRSL), et rerank seulement si logits dispo + embedding dispo
        if (np.isfinite(v) and (v < tau) and (normalize_sid(sid) in sid2idx) and (L is not None)):
            idx = sid2idx[normalize_sid(sid)]
            es = E_S[idx:idx+1]
            w = cosine_sim_matrix(es, E_L_aligned)[0]
            r = L * w
            ph = int(np.argmax(r))
            applied = True

        y_true.append(true_id); y_pred_base.append(pb); y_pred_hybrid.append(ph)
        rerank_applied.append(applied); used_sids.append(sid)


    # Métriques
    y_true = np.asarray(y_true, dtype=int)
    y_pred_base = np.asarray(y_pred_base, dtype=int)
    y_pred_hybrid = np.asarray(y_pred_hybrid, dtype=int)

    mf1_base = macro_f1(y_true, y_pred_base)
    wf1_base = weighted_f1(y_true, y_pred_base)
    mf1_hyb  = macro_f1(y_true, y_pred_hybrid)
    wf1_hyb  = weighted_f1(y_true, y_pred_hybrid)

    # (Optionnel) affichage d'exemples — on respecte ta logique mais on le coupe par défaut en collecte
    if any(k > 0 for k in (show_k_improved, show_k_influenced, show_k_degraded, debug_k)):
        labs = labels_logits if labels_logits else [str(i) for i in range(len(test_data[used_sids[0]]["logits"]))]
        rerank_applied = np.asarray(rerank_applied, dtype=bool)
        changed_idx  = [i for i in range(len(used_sids)) if rerank_applied[i] and (y_pred_base[i] != y_pred_hybrid[i])]
        improved_idx = [i for i in changed_idx if (y_pred_base[i] != y_true[i] and y_pred_hybrid[i] == y_true[i])]
        degraded_idx = [i for i in changed_idx if (y_pred_base[i] == y_true[i] and y_pred_hybrid[i] != y_true[i])]

        # … (tu peux réutiliser ton code d’affichage existant ici si tu veux)

    return mf1_base, wf1_base, mf1_hyb, wf1_hyb






# --------- MAIN ---------
def main():
    set_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="ex: scotus-rhetorical_function")
    ap.add_argument("--exp", required=True, help="ex: bert-base-uncased")
    ap.add_argument("--seed", required=True, help="ex: 0 (sera utilisé comme 'seed-0')")
    ap.add_argument("--emb-exp", default=None, help="par défaut = --exp")
    ap.add_argument("--emb-seed", default=None, help="par défaut = --seed")
    ap.add_argument("--show-k-improved", type=int, default=8)
    ap.add_argument("--show-k-influenced", type=int, default=8)
    ap.add_argument("--show-k-degraded", type=int, default=4)
    ap.add_argument("--debug-k", type=int, default=0)
    ap.add_argument("--latex-out", default="model_hard_hybrid_table.tex",
                    help="nom de fichier .tex à écrire sous ROOT (défaut: model_hard_hybrid_table.tex)")
    ap.add_argument("--gather", action="store_true",
                    help="Si activé: évalue toutes les paires (task,exp) disponibles pour seed donné, et génère un seul tableau LaTeX.")

    args = ap.parse_args()



    emb_exp  = args.emb_exp  if args.emb_exp  else args.exp
    emb_seed = args.emb_seed if args.emb_seed else args.seed


    if args.gather:
        # 1) Découvrir toutes les paires (task, exp) pour ce seed
        pairs = discover_pairs_for_seed(args.seed)
        if not pairs:
            sys.exit(f"[STOP] Aucune paire (task,exp) trouvée pour seed-{args.seed} dans {OUT_DIR}")

        rows = []
        for task, exp in pairs:
            # Source d'embeddings (on respecte --emb-exp si fourni; sinon règle bge-m3 pour LLMs)
            local_emb_exp  = emb_source_for_exp(exp, emb_exp)
            local_emb_seed = args.emb_seed if args.emb_seed else args.seed
            try:
                mf1_base, wf1_base, mf1_hyb, wf1_hyb = evaluate_one_pair(
                    task, exp, args.seed,
                    local_emb_exp, local_emb_seed,
                    show_k_improved=0, show_k_influenced=0, show_k_degraded=0, debug_k=0
                )
                # 2 lignes par modèle dans le tableau (baseline + rerank)
                rows.append({"task": task, "exp": exp, "mode": "baseline",
                             "macro_all": mf1_base, "weighted_all": wf1_base})
                rows.append({"task": task, "exp": exp, "mode": "rerank",
                             "macro_all": mf1_hyb,  "weighted_all": wf1_hyb})
                print(f"[OK] {task} / {exp} → mF1 base={mf1_base:.4f} | mF1 hyb={mf1_hyb:.4f}")
            except Exception as e:
                print(f"[WARN] skip {task}/{exp} : {e}")

        if not rows:
            sys.exit("[STOP] Aucun résultat exploitable en mode collecte.")

        latex = _render_latex_table(rows)
        out_tex = ROOT / args.latex_out
        with out_tex.open("w", encoding="utf-8") as f:
            f.write(latex)
        print(f"\n[OK] Table LaTeX (collecte) sauvegardée: {out_tex}")
        return


if __name__ == "__main__":
    main()
