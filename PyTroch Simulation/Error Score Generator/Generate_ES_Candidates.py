from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

BIT_RE = re.compile(r"bit\s*(\d+)", re.I)

def bit_no(name: str) -> int:
    m = BIT_RE.fullmatch(name.strip())
    if not m:
        raise ValueError(f"Invalid bit column: {name}")
    return int(m.group(1))

def baseline_value(df, bit_cols, method, rows, explicit):
    if explicit is not None:
        return float(explicit)
    values = df[bit_cols].to_numpy(float)
    if method == "max":
        return float(np.nanmax(values))
    if method == "first-row-median":
        return float(np.nanmedian(values[0]))
    return float(np.nanmedian(values[:min(max(rows,1),len(df))]))

def raw_score(ber, acc, baseline, method):
    loss = np.maximum(baseline - acc, 0.0)
    if method == "mean-loss":
        return float(loss.mean())
    if method == "max-loss":
        return float(loss.max())
    if method == "endpoint-loss":
        return float(loss[-1])
    x = np.log10(ber)
    x = x - x.min()
    return float(np.trapezoid(loss, x)) if len(x) > 1 else float(loss[0])

def minmax(v, lo, hi):
    v = np.asarray(v, float)
    if np.isclose(v.max(), v.min()):
        return np.full(len(v), lo, int)
    y = lo + (hi-lo)*(v-v.min())/(v.max()-v.min())
    return np.clip(np.rint(y), lo, hi).astype(int)

def rank_group(raw, levels, tol):
    raw = np.asarray(raw, float)
    order = np.argsort(-raw)
    out = np.empty(len(raw), int)
    groups, current, ref = [], [], None
    for idx in order:
        score = float(raw[idx])
        if not current:
            current, ref = [int(idx)], score
            continue
        rel = abs(ref-score)/max(abs(ref), 1e-12)
        if rel <= tol:
            current.append(int(idx))
        else:
            groups.append(current)
            current, ref = [int(idx)], score
    if current:
        groups.append(current)
    for gi, group in enumerate(groups):
        level = levels[min(gi, len(levels)-1)]
        for idx in group:
            out[idx] = level
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("--output-prefix", type=Path, default=Path("ES_candidates"))
    p.add_argument("--raw-method", choices=["log-auc","mean-loss","max-loss","endpoint-loss"], default="log-auc")
    p.add_argument("--baseline-method", choices=["max","first-row-median","low-ber-median"], default="low-ber-median")
    p.add_argument("--baseline-rows", type=int, default=2)
    p.add_argument("--baseline", type=float)
    p.add_argument("--es-min", type=int, default=1)
    p.add_argument("--es-max", type=int, default=31)
    p.add_argument("--group-levels", default="31,15,7,3,1")
    p.add_argument("--group-tolerance", type=float, default=0.10)
    a = p.parse_args()

    df = pd.read_csv(a.input_csv)
    if "BER" not in df.columns:
        raise ValueError("CSV must contain BER column")
    bit_cols = [c for c in df.columns if BIT_RE.fullmatch(c.strip())]
    if not bit_cols:
        raise ValueError("No columns like 'Bit 14' found")
    df = df[["BER", *bit_cols]].replace([np.inf,-np.inf], np.nan).dropna()
    df = df[df["BER"] > 0].sort_values("BER").reset_index(drop=True)

    base = baseline_value(df, bit_cols, a.baseline_method, a.baseline_rows, a.baseline)
    ber = df["BER"].to_numpy(float)

    rows = []
    for c in bit_cols:
        acc = df[c].to_numpy(float)
        loss = np.maximum(base-acc, 0.0)
        rows.append({
            "bit": bit_no(c),
            "column": c,
            "baseline_accuracy": base,
            "mean_accuracy_loss": float(loss.mean()),
            "maximum_accuracy_loss": float(loss.max()),
            "raw_importance": raw_score(ber, acc, base, a.raw_method),
        })
    out = pd.DataFrame(rows)
    raw = out["raw_importance"].to_numpy(float)
    shifted = np.maximum(raw - raw.min(), 0.0)

    out["ES_linear"] = minmax(raw, a.es_min, a.es_max)
    out["ES_log"] = minmax(np.log1p(shifted), a.es_min, a.es_max)
    out["ES_sqrt"] = minmax(np.sqrt(shifted), a.es_min, a.es_max)
    levels = [int(x.strip()) for x in a.group_levels.split(",")]
    out["ES_rank_group"] = rank_group(raw, levels, a.group_tolerance)

    out = out.sort_values(["raw_importance","bit"], ascending=[False,False]).reset_index(drop=True)
    prefix = a.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    table_path = prefix.with_name(prefix.name + "_table.csv")
    long_path = prefix.with_name(prefix.name + "_long.csv")
    config_path = prefix.with_name(prefix.name + "_config.json")
    out.to_csv(table_path, index=False, encoding="utf-8-sig")

    long = out.melt(
        id_vars=["bit","raw_importance"],
        value_vars=["ES_linear","ES_log","ES_sqrt","ES_rank_group"],
        var_name="candidate",
        value_name="ES",
    )
    long["candidate"] = long["candidate"].str.replace("ES_","", regex=False)
    long.to_csv(long_path, index=False, encoding="utf-8-sig")

    cfg = {
        "baseline_accuracy": base,
        "raw_method": a.raw_method,
        "es_range": [a.es_min, a.es_max],
        "group_levels": levels,
        "group_tolerance": a.group_tolerance,
        "candidate_ES": {
            name.replace("ES_",""): {str(int(r.bit)): int(getattr(r,name)) for r in out.itertuples()}
            for name in ["ES_linear","ES_log","ES_sqrt","ES_rank_group"]
        },
    }
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Baseline accuracy: {base:.6f}")
    print(f"Raw-score method: {a.raw_method}\n")
    print(out[["bit","raw_importance","ES_linear","ES_log","ES_sqrt","ES_rank_group"]].to_string(index=False))
    print(f"\nSaved:\n- {table_path}\n- {long_path}\n- {config_path}")

if __name__ == "__main__":
    main()
