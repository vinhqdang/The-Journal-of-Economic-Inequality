"""Collect first-stage fit statistics (out-of-fold R2 of log income, Duan smearing
factor, C1 calibration slope) from the E2 checkpoints into output/e2_fit.csv."""
import pickle
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for p in sorted((ROOT / "data" / "e2_checkpoints").glob("main_rep*.pkl")):
    d = pickle.load(open(p, "rb"))["meta"]
    for k, v in d["fit"].items():
        regime, model = k.split("_", 1)
        rows.append(dict(rep=d["rep"], regime=regime, model=model, **v))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "output" / "e2_fit.csv", index=False)
print(df.groupby(["regime", "model"])[["oof_r2_log", "smear", "C1_slope"]].mean().round(3))
