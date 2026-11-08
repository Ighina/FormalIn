"""Stage ``modernbert``: a fine-tuned text classifier on the step alone.

``answerdotai/ModernBERT-base`` is fine-tuned per (model, task) on the same
problem-level split as the probes, so its test AUROC bounds what the step's
*text* predicts about the verdict without any access to the autoformalizer.
The step text is the only input (no subset, no metadata).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from leansteer import config  # noqa: F401
from leansteer.data.faithform import build_task, load_af, problem_split
from leansteer.experiment import Run, write_json

log = logging.getLogger(__name__)
ENCODER = "answerdotai/ModernBERT-base"


def _batches(texts, labels, tok, bs, max_len, shuffle, rng):
    idx = np.arange(len(texts))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[k] for k in j], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        yield enc, torch.tensor([labels[k] for k in j], dtype=torch.float32)


def train_eval(train_text, train_y, val_text, val_y, test_text, test_y, cfg_mb: dict, device: str, seed: int = 0) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModelForSequenceClassification.from_pretrained(ENCODER, num_labels=1).to(device)
    epochs, bs, lr, max_len = int(cfg_mb["epochs"]), int(cfg_mb["batch_size"]), float(cfg_mb["lr"]), int(cfg_mb["max_length"])
    steps = epochs * ((len(train_text) + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * steps))) * max(0.0, 1 - s / steps))
    pos_w = torch.tensor([(len(train_y) - sum(train_y)) / max(1, sum(train_y))], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def scores(texts):
        model.eval(); out = []
        with torch.no_grad():
            for enc, _ in _batches(texts, [0] * len(texts), tok, 128, max_len, False, rng):
                out.append(model(**{k: v.to(device) for k, v in enc.items()}).logits[:, 0].float().cpu().numpy())
        model.train()
        return np.concatenate(out)

    best, best_state, hist = -1.0, None, []
    model.train()
    for ep in range(epochs):
        for enc, y in _batches(train_text, train_y, tok, bs, max_len, True, rng):
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[:, 0]
            loss = loss_fn(logits.float(), y.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        va = float(roc_auc_score(val_y, scores(val_text))) if len(set(val_y)) == 2 else 0.0
        hist.append(va)
        if va > best:
            best, best_state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    s_te = scores(test_text)
    return {"val_auc_by_epoch": hist, "best_val_auc": best, "test_auc": float(roc_auc_score(test_y, s_te)), "test_scores": s_te}


def run_modernbert(cfg: dict, model: str, tasks: Optional[list] = None, inputs: str = "step", seeds: int = 1) -> dict:
    """``inputs``: ``step`` (the step text) or ``step_statement`` (step ||| the Lean
    theorem the model wrote, for the teacher-forced tasks).  ``seeds`` > 1 repeats the
    fine-tuning with different seeds (same split) and reports mean/sd."""
    run = Run(cfg)
    a, mb = cfg["analysis"], cfg["modernbert"]
    seed = int(a["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    df = load_af(models=[model], with_outputs=(inputs != "step"))
    if inputs == "step_statement":
        df = df[~df.is_perturbed].copy()
        df["text"] = df.step_content + " ||| " + df.statement.fillna("(no theorem)")
    else:
        df["text"] = df.step_content
    out_dir = run.model_dir(model) / ("modernbert" if inputs == "step" else "modernbert_tf"); out_dir.mkdir(exist_ok=True)
    summary = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {}
    for task in list(tasks or cfg["tasks"]):
        table = build_task(df, task, seed, balance=task in cfg["balance"])
        assign = problem_split(table.group, a["test_frac"], seed)
        te = table.group.map(assign).eq("test").to_numpy()
        tr_t, te_t = table[~te], table[te]
        val_assign = problem_split(tr_t.group, a["val_frac"], seed + 1)
        va = tr_t.group.map(val_assign).eq("test").to_numpy()
        fit_t, val_t = tr_t[~va], tr_t[va]
        if cfg.get("limit"):
            fit_t = fit_t.head(int(cfg["limit"])); val_t = val_t.head(int(cfg["limit"])); te_t = te_t.head(int(cfg["limit"]))
        log.info("[%s/%s] ModernBERT(%s) fit %d val %d test %d", model, task, inputs, len(fit_t), len(val_t), len(te_t))
        runs = []
        for k in range(int(seeds)):
            r = train_eval(fit_t.text.tolist(), fit_t.label.tolist(), val_t.text.tolist(), val_t.label.tolist(),
                           te_t.text.tolist(), te_t.label.tolist(), mb, device, seed + k)
            if k == 0:
                np.save(out_dir / f"{task}_test_scores.npy", r.pop("test_scores"))
                (out_dir / f"{task}_test_uids.txt").write_text("\n".join(te_t.uid))
            else:
                r.pop("test_scores")
            runs.append(r)
        summary[task] = {"n_fit": len(fit_t), "n_val": len(val_t), "n_test": len(te_t), "inputs": inputs, **runs[0],
                         "seed_test_aucs": [x["test_auc"] for x in runs],
                         "test_auc_mean": float(np.mean([x["test_auc"] for x in runs])),
                         "test_auc_sd": float(np.std([x["test_auc"] for x in runs], ddof=1)) if len(runs) > 1 else None}
        log.info("[%s/%s] ModernBERT(%s) test AUROC %.3f (best val %.3f) over %d seed(s): %.3f +- %s", model, task, inputs, runs[0]["test_auc"],
                 runs[0]["best_val_auc"], len(runs), summary[task]["test_auc_mean"], summary[task]["test_auc_sd"])
        write_json(out_dir / "summary.json", summary)
    return summary
