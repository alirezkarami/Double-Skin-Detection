"""
Inference: probability that a cropped billet image is double-skin (دوپوسته).

    .venv/bin/python predict.py runs/v1 path/to/img.jpg
    .venv/bin/python predict.py runs/v1 path/to/folder --csv out.csv

From code (e.g. inside the Basler / pypylon grabbing loop):
    from predict import DoubleSkinDetector
    det = DoubleSkinDetector("runs/v1")
    prob = det.predict(cropped_frame)        # numpy array (mono8 or RGB) or PIL image
    if prob >= det.threshold: ...            # double-skin
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DoubleSkinDetector:
    def __init__(self, run_dir, threshold=None, threads=None):
        run_dir = Path(run_dir)
        meta = json.load(open(run_dir / "model_meta.json"))
        self.h, self.w = meta["height"], meta["width"]
        self.threshold = threshold if threshold is not None else meta["threshold"]
        self.threshold_high_recall = meta.get("threshold_high_recall")
        if threads:
            torch.set_num_threads(threads)
        self.model = torch.jit.load(str(run_dir / "model_ts.pt"), map_location="cpu").eval()

    def _prep(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("L").resize((self.w, self.h), Image.BILINEAR)
        return np.asarray(img, np.uint8)[None]

    @torch.no_grad()
    def predict_batch(self, imgs):
        x = torch.from_numpy(np.stack([self._prep(i) for i in imgs]))
        return self.model(x).numpy().reshape(-1)

    def predict(self, img):
        return float(self.predict_batch([img])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    det = DoubleSkinDetector(a.run_dir, a.threshold)
    inp = Path(a.input)
    files = sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXT) if inp.is_dir() else [inp]
    rows = []
    for i in range(0, len(files), 32):
        chunk = files[i:i + 32]
        for f, p in zip(chunk, det.predict_batch([Image.open(f) for f in chunk])):
            label = "DOUBLE_SKIN" if p >= det.threshold else "OK"
            rows.append((str(f), f"{p:.4f}", label))
            print(f"{p:.4f}  {label:11s}  {f}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["path", "prob_double_skin", "decision"]); w.writerows(rows)


if __name__ == "__main__":
    main()
