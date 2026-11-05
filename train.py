"""
Double-skin (دوپوستگی) classifier for cropped billet images (Basler camera).

    dataset/0/*.jpg  -> OK
    dataset/1/*.jpg  -> double-skin (defect)

The defect class is tiny (~40 images vs ~2650), so:
  * evaluation uses grouped, stratified K-fold cross-validation. Consecutive frames of
    the same billet (timestamps a few seconds apart) are kept in the same fold, so the
    model is never tested on a near-copy of a training image;
  * each epoch samples a class-balanced subset (all defects oversampled + random OKs);
  * after CV, a final model is trained on all data and the threshold is set from the
    out-of-fold CV probabilities.

Usage:
    .venv/bin/python train.py --data data/dataset --out runs/v1            # CV + final
    .venv/bin/python train.py --data data/dataset --out runs/v1 --folds 0  # final only
"""
import argparse
import csv
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import v2 as T

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN, STD = 0.449, 0.226  # grayscale version of ImageNet stats


# ---------------------------------------------------------------- data
def parse_time(name: str):
    """Best-effort timestamp from the filename, used only to group consecutive frames."""
    m = re.match(r"(\d\d)_(\d\d)_(\d\d)_(\d{4})-(\d\d)-(\d\d)", name)
    if m:
        h, mi, s, y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, h, mi, s)
    m = re.match(r"(\d{4})(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)?", name)
    if m:
        y, mo, d, h, mi = map(int, m.groups()[:5])
        return datetime(y, mo, d, h, mi, int(m.group(6) or 0))
    return None


def collect(root: Path, gap_s: int = 20):
    items = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT and p.parent.name in ("0", "1"):
            items.append({"path": str(p), "label": int(p.parent.name), "time": parse_time(p.name)})
    # group frames of the same class taken within gap_s seconds of each other
    gid = 0
    for lab in (0, 1):
        sub = sorted([it for it in items if it["label"] == lab and it["time"]], key=lambda d: d["time"])
        prev = None
        for it in sub:
            if prev is None or (it["time"] - prev).total_seconds() > gap_s:
                gid += 1
            it["group"] = gid; prev = it["time"]
    for it in items:
        if "group" not in it:
            gid += 1; it["group"] = gid
    return items


def grouped_stratified_folds(items, k, seed):
    """Assign whole groups to folds, balancing defect count and image count per fold."""
    rng = random.Random(seed)
    fold_of = {}
    for lab in (1, 0):
        groups = {}
        for it in items:
            if it["label"] == lab:
                groups.setdefault(it["group"], []).append(it)
        keys = list(groups); rng.shuffle(keys)
        keys.sort(key=lambda g: -len(groups[g]))
        load = [0] * k
        for g in keys:
            f = int(np.argmin(load)); fold_of[g] = f; load[f] += len(groups[g])
    return [fold_of[it["group"]] for it in items]


def load_cache(items, H, W):
    """Decode + resize every image once, keep as uint8 grayscale tensor in RAM."""
    arr = np.zeros((len(items), 1, H, W), np.uint8)
    for i, it in enumerate(items):
        arr[i, 0] = np.asarray(Image.open(it["path"]).convert("L").resize((W, H), Image.BILINEAR))
    return torch.from_numpy(arr)


def make_aug():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomAffine(degrees=4, translate=(0.04, 0.04), scale=(0.92, 1.08))], p=0.7),
        T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.35)], p=0.8),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
    ])


def to_input(x_uint8):
    x = x_uint8.float() / 255.0
    x = (x - MEAN) / STD
    return x.expand(-1, 3, -1, -1)


# ---------------------------------------------------------------- model
def build_model(arch: str, pretrained=True):
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, 1))
    elif arch == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 1))
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    else:
        raise ValueError(arch)
    return m


class Deployable(nn.Module):
    """uint8 grayscale [B,1,H,W] (already resized) -> probability of double-skin. Averages hflip TTA."""

    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        x = (x.float() / 255.0 - MEAN) / STD
        x = x.expand(-1, 3, -1, -1)
        logit = (self.net(x) + self.net(torch.flip(x, dims=[3]))) / 2
        return torch.sigmoid(logit).squeeze(1)


# ---------------------------------------------------------------- metrics (numpy, no sklearn)
def roc_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allp = np.concatenate([pos, neg])  # average ranks for ties
    for v in np.unique(allp):
        idx = allp == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def avg_precision(y, p):
    y = np.asarray(y); order = np.argsort(-np.asarray(p)); ys = y[order]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(ys.sum(), 1))


def metrics(y, p, thr):
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    return {"threshold": round(float(thr), 4), "auc": roc_auc(y, p), "ap": avg_precision(y, p),
            "recall_defect": rec, "precision_defect": prec, "specificity": tn / max(tn + fp, 1),
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "acc": (tp + tn) / len(y),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def choose_thresholds(y, p):
    ts = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), p]))
    f1s = [metrics(y, p, t)["f1"] for t in ts]
    t_f1 = float(ts[int(np.argmax(f1s))])
    # highest threshold that still catches >= 95% of defects (safety-oriented operating point)
    cands = [t for t in ts if metrics(y, p, t)["recall_defect"] >= 0.95]
    t_rec = float(max(cands)) if cands else float(ts.min())
    return t_f1, t_rec


# ---------------------------------------------------------------- training
@torch.no_grad()
def predict(net, X, bs=64):
    net.eval(); d = Deployable(net); out = []
    for i in range(0, len(X), bs):
        out.append(d(X[i:i + bs]).numpy())
    return np.concatenate(out)


def train_one(X, y, args, log_prefix="", X_val=None, y_val=None):
    torch.manual_seed(args.seed)
    net = build_model(args.arch)
    aug = make_aug()
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n_pos_per_epoch = args.pos_per_epoch
    n_neg_per_epoch = args.neg_per_epoch
    steps = args.epochs * ((n_pos_per_epoch + n_neg_per_epoch) // args.bs)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        net.train(); t0 = time.time(); tot = n = 0
        idx = np.concatenate([rng.choice(pos, n_pos_per_epoch, replace=True),
                              rng.choice(neg, n_neg_per_epoch, replace=False)])
        rng.shuffle(idx)
        for b in range(len(idx) // args.bs):
            bi = idx[b * args.bs:(b + 1) * args.bs]
            xb = torch.stack([aug(X[i]) for i in bi])
            yb = torch.from_numpy(y[bi]).float()
            logit = net(to_input(xb)).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, yb * 0.95 + 0.025)  # light label smoothing
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(bi); n += len(bi)
        msg = f"{log_prefix}ep {ep:02d} loss {tot / n:.4f} ({time.time() - t0:.0f}s)"
        if X_val is not None and (ep % args.eval_every == 0 or ep == args.epochs):
            pv = predict(net, X_val); m = metrics(y_val, pv, 0.5)
            msg += f" | val auc {m['auc']:.4f} ap {m['ap']:.4f} rec@.5 {m['recall_defect']:.3f} fp@.5 {m['fp']}"
        print(msg, flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--pos_per_epoch", type=int, default=160)
    ap.add_argument("--neg_per_epoch", type=int, default=480)
    ap.add_argument("--folds", type=int, default=5, help="0 = skip CV, train final model only")
    ap.add_argument("--eval_every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    items = collect(Path(args.data))
    y = np.array([it["label"] for it in items])
    print(f"images: {len(items)}  OK={int((y == 0).sum())}  defect={int((y == 1).sum())}  "
          f"defect groups={len({it['group'] for it in items if it['label'] == 1})}")
    t0 = time.time(); X = load_cache(items, args.height, args.width)
    print(f"cached {tuple(X.shape)} in {time.time() - t0:.0f}s", flush=True)

    results = {"args": vars(args)}
    thr_f1 = thr_rec = 0.5
    if args.folds > 1:
        folds = np.array(grouped_stratified_folds(items, args.folds, args.seed))
        oof = np.zeros(len(items))
        for k in range(args.folds):
            tr, va = folds != k, folds == k
            print(f"\n=== fold {k + 1}/{args.folds}: train {tr.sum()} (defect {y[tr].sum()}), "
                  f"val {va.sum()} (defect {y[va].sum()})", flush=True)
            net = train_one(X[tr], y[tr], args, f"[f{k + 1}] ", X[va], y[va])
            oof[va] = predict(net, X[va])
        thr_f1, thr_rec = choose_thresholds(y, oof)
        results["cv"] = {"oof@0.5": metrics(y, oof, 0.5), "oof@best_f1": metrics(y, oof, thr_f1),
                         "oof@recall95": metrics(y, oof, thr_rec),
                         "per_fold_auc": [roc_auc(y[folds == k], oof[folds == k]) for k in range(args.folds)]}
        with open(out / "cv_predictions.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["path", "label", "fold", "prob_double_skin"])
            for it, fo, p in sorted(zip(items, folds, oof), key=lambda r: -r[2]):
                w.writerow([it["path"], it["label"], fo, f"{p:.4f}"])
        print("\n=== CROSS-VALIDATION (out-of-fold, every image predicted by a model that never saw it)")
        for k_, v in results["cv"].items():
            print(k_, json.dumps(v))

    print("\n=== training final model on all data", flush=True)
    net = train_one(X, y, args, "[final] ")
    net.eval()
    meta = {"arch": args.arch, "height": args.height, "width": args.width, "threshold": thr_f1,
            "threshold_high_recall": thr_rec, "input": "uint8 grayscale, resized to (height,width)",
            "classes": {"0": "OK", "1": "double_skin"}}
    torch.save({"state_dict": net.state_dict(), **meta}, out / "model.pt")
    json.dump(meta, open(out / "model_meta.json", "w"), indent=2)
    dep = Deployable(net).eval()
    ex = torch.zeros(1, 1, args.height, args.width, dtype=torch.uint8)
    torch.jit.trace(dep, ex).save(str(out / "model_ts.pt"))
    try:
        torch.onnx.export(dep, ex, str(out / "model.onnx"), input_names=["image_u8"],
                          output_names=["prob_double_skin"], opset_version=17,
                          dynamic_axes={"image_u8": {0: "batch"}, "prob_double_skin": {0: "batch"}}, dynamo=False)
        print("onnx exported")
    except Exception as e:  # onnx package may be unavailable
        print("onnx export skipped:", e)
    json.dump(results, open(out / "results.json", "w"), indent=2)
    print("saved to", out)


if __name__ == "__main__":
    main()
