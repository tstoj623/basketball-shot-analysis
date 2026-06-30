"""
Ranks a shot's biomechanical faults by how unlike known-good shooters each
metric's per-frame distribution is, via KL divergence against a good-shot
reference. Threshold-free and label-free: only a corpus of good shots is needed.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Half-window sizes (in seconds) around each detected phase frame, used to pool
# the per-frame samples this module takes a KL divergence of.
SP_HALF_SEC = 0.233   # ~7 frames at 30fps, around the set point
RE_HALF_SEC = 0.167   # ~5 frames at 30fps, around release

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "kl_severity")
REFERENCE_PKL = os.path.join(MODEL_DIR, "kl_reference.pkl")

FEATURES_CSV = "complete_features_v2.csv"
GOOD_SHOTS_TXT = "good_shots.txt"
FPS = 30.0

# Number of points the two KDEs are evaluated on when integrating the KL sum.
GRID_N = 256
# Floor on the reference density so log(q/p) stays finite in regions the good
# shots never visited (this is what makes a never-seen pose score *high*).
EPS = 1e-6


# ============================================================
# Metric registry
# ============================================================
# Each metric is a per-frame series extracted over a phase window. `column` is
# read straight from the features frame; `derive` builds a series from other
# columns (e.g. lateral drift = right_wrist_x - nose_x). `window` selects which
# phase span to pool frames from.
#
# Only metrics that are genuinely per-frame distributions belong here. Per-shot
# scalars (kinetic-sequence delta, body-twist delta, fluidity reversals) have no
# within-shot distribution to take a KL of and stay in shot_scorer.py.

@dataclass(frozen=True)
class MetricSpec:
    name: str
    column: Optional[str]       # direct column, or None if derived
    window: str                 # "setpoint" | "setpoint_to_release"
    derive: Optional[str] = None  # key into _DERIVERS when column is None
    label: str = ""             # human-facing fault name


METRICS = [
    MetricSpec("elbow_flare",      "shooting_plane_deviation",    "setpoint",            label="Elbow Alignment"),
    MetricSpec("set_point_height", "wrist_nose_diff",             "setpoint",            label="Set Point Height"),
    MetricSpec("knee_valgus",      "valgus_ratio",                "setpoint",            label="Knee Valgus"),
    MetricSpec("foot_width",       "ankle_hip_ratio",             "setpoint",            label="Stance Width"),
    MetricSpec("push_shot",        "forearm_angle_from_vertical", "setpoint_to_release", label="Push Shot"),
    MetricSpec("lateral_drift",    None, "setpoint", derive="wrist_nose_x", label="Shot Line Alignment"),
]

_DERIVERS = {
    # right wrist horizontal offset from nose — the lateral-drift signal
    "wrist_nose_x": lambda d: (d["right_wrist_x"] - d["nose_x"])
                              if {"right_wrist_x", "nose_x"} <= set(d.columns) else None,
}


# ============================================================
# Per-frame series extraction
# ============================================================

def _frame_col(df: pd.DataFrame) -> str:
    return "frame" if "frame" in df.columns else "Frame"


def _window_bounds(phases: dict, window: str, fps: float):
    """Return (start_frame, end_frame) for a metric's phase window, or None."""
    sp = phases.get("setpoint_frame")
    re = phases.get("release_frame")
    if window == "setpoint":
        if sp is None:
            return None
        half = max(1, round(SP_HALF_SEC * fps))
        return sp - half, sp + half
    if window == "setpoint_to_release":
        if sp is None or re is None or re <= sp:
            return None
        # mirror score_push_shot: drop the last couple frames (follow-through)
        return sp - max(1, round(0.033 * fps)), re - max(1, round(0.067 * fps))
    return None


def _series(df: pd.DataFrame, spec: MetricSpec, phases: dict, fps: float) -> Optional[np.ndarray]:
    """Per-frame values of `spec` over its phase window for one shot."""
    bounds = _window_bounds(phases, spec.window, fps)
    if bounds is None:
        return None
    lo, hi = bounds
    fc = _frame_col(df)
    win = df[(df[fc] >= lo) & (df[fc] <= hi)]
    if len(win) == 0:
        return None

    if spec.column is not None:
        if spec.column not in win.columns:
            return None
        s = win[spec.column]
    else:
        s = _DERIVERS[spec.derive](win)
        if s is None:
            return None

    vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    return vals if len(vals) else None


# ============================================================
# KL machinery
# ============================================================

def _kde(samples: np.ndarray) -> Optional[gaussian_kde]:
    """Gaussian KDE, robust to (near-)constant samples that make it singular."""
    if len(samples) < 2:
        return None
    if np.std(samples) < 1e-9:
        # all-but-identical values — jitter so the covariance is non-singular
        samples = samples + np.random.default_rng(0).normal(0, 1e-6, size=len(samples))
    try:
        return gaussian_kde(samples)
    except np.linalg.LinAlgError:
        return None


def _kl(q_samples: np.ndarray, p_kde: gaussian_kde, grid: np.ndarray) -> Optional[float]:
    """KL( q || p ) with q estimated from q_samples, both evaluated on `grid`."""
    q_kde = _kde(q_samples)
    if q_kde is None:
        return None
    dx = grid[1] - grid[0]
    q = q_kde(grid)
    p = p_kde(grid)
    # normalize to proper discrete distributions over the grid
    q = q / (q.sum() * dx + EPS)
    p = p / (p.sum() * dx + EPS)
    q = np.clip(q, EPS, None)
    p = np.clip(p, EPS, None)
    kl = np.sum(q * np.log(q / p) * dx)
    return float(max(kl, 0.0))   # KL >= 0; clip tiny negative numerical noise


# ============================================================
# Reference model
# ============================================================

def build_reference(features_csv: str = FEATURES_CSV,
                    good_shots_txt: str = GOOD_SHOTS_TXT,
                    fps: float = FPS) -> dict:
    """Pool per-frame values across the good shots and build one KDE per metric.

    Stores, per metric: the standardizing mean/std (from pooled good frames), the
    good-form KDE in standardized space, and the evaluation grid. Pickled to
    REFERENCE_PKL.
    """
    df = pd.read_csv(features_csv, low_memory=False)
    good = [l.strip() for l in open(good_shots_txt) if l.strip()]
    have = set(df["shot"].unique())

    ref = {}
    print(f"Building KL reference from {len(good)} good shots\n")
    for spec in METRICS:
        pooled = []
        used = 0
        for shot in good:
            if shot not in have:
                continue
            sdf = df[df["shot"] == shot]
            phases = _phases_from_row(sdf)
            vals = _series(sdf, spec, phases, fps)
            if vals is not None:
                pooled.append(vals)
                used += 1
        if not pooled:
            print(f"  ⚠️  {spec.name:18s} no good-shot frames, skipping")
            continue
        pooled = np.concatenate(pooled)
        mean, std = float(pooled.mean()), float(pooled.std())
        std = std if std > 1e-9 else 1.0
        z = (pooled - mean) / std
        kde = _kde(z)
        if kde is None:
            print(f"  ⚠️  {spec.name:18s} KDE failed, skipping")
            continue
        # Standardized grid: good form is centered at 0; pad generously so a
        # shot landing far in a tail is still represented on the grid.
        grid = np.linspace(z.min() - 4, z.max() + 4, GRID_N)
        ref[spec.name] = {"mean": mean, "std": std, "kde": kde, "grid": grid}
        print(f"  {spec.name:18s} n={used:2d} shots, {len(pooled):4d} frames  "
              f"mean={mean:8.3f} std={std:7.3f}")

    # Second pass: each metric has a non-zero baseline KL even among good shots
    # (a single shot is a narrow distribution inside the broad pooled reference).
    # Measure that baseline so rank_faults can report KL *in excess of* normal
    # good-shooter variation, making the ranking fair across metrics. Still
    # label-free — only good shots are used.
    baseline = {name: [] for name in ref}
    for shot in good:
        if shot not in have:
            continue
        sdf = df[df["shot"] == shot]
        phases = _phases_from_row(sdf)
        for spec in METRICS:
            r = ref.get(spec.name)
            if r is None:
                continue
            vals = _series(sdf, spec, phases, fps)
            if vals is None or len(vals) < 2:
                continue
            kl = _kl((vals - r["mean"]) / r["std"], r["kde"], r["grid"])
            if kl is not None:
                baseline[spec.name].append(kl)
    for name, kls in baseline.items():
        arr = np.asarray(kls)
        ref[name]["base_mean"] = float(arr.mean()) if len(arr) else 0.0
        ref[name]["base_std"] = float(arr.std()) if len(arr) > 1 and arr.std() > 1e-9 else 1.0

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(REFERENCE_PKL, "wb") as fh:
        pickle.dump(ref, fh)
    print(f"\n✅ wrote {REFERENCE_PKL}  ({len(ref)} metrics)")
    return ref


def load_reference(path: str = REFERENCE_PKL) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run `python kl_scorer.py --build` first")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _phases_from_row(shot_df: pd.DataFrame) -> dict:
    """Phase frames are constant across a shot's rows; pull them from the first."""
    row = shot_df.iloc[0]
    out = {}
    for k in ("gather_frame", "setpoint_frame", "release_frame"):
        v = row.get(k)
        out[k] = int(v) if pd.notna(v) else None
    return out


# ============================================================
# Scoring
# ============================================================

@dataclass
class FaultRank:
    metric: str
    label: str
    kl: float            # raw KL divergence from good form (higher = more unlike)
    kl_z: float          # KL in good-shot-baseline std units (the fair score)
    n_frames: int        # frames the shot contributed to its distribution


def rank_faults(shot_df: pd.DataFrame, phases: dict, ref: dict,
                fps: float = FPS, by: str = "kl_z") -> list:
    """Rank a shot's metrics by divergence from good form, worst first.

    Threshold-free and label-free: the ordering itself is the output. The top
    entry is the joint "most unlike good shooters" for this shot.

    Two scores per metric:
      kl    — raw KL( q_shot || p_good ). Comparable *within* a metric over time.
      kl_z  — (kl - good-shot baseline mean) / baseline std for that metric.
              Comparable *across* metrics, so the top of the list is a genuine
              fault and not just the metric with the broadest reference.
    `by` selects the sort key ("kl_z" default, or "kl" for the raw ordering).
    """
    ranking = []
    for spec in METRICS:
        r = ref.get(spec.name)
        if r is None:
            continue
        vals = _series(shot_df, spec, phases, fps)
        if vals is None or len(vals) < 2:
            continue
        kl = _kl((vals - r["mean"]) / r["std"], r["kde"], r["grid"])
        if kl is None:
            continue
        kl_z = (kl - r.get("base_mean", 0.0)) / r.get("base_std", 1.0)
        ranking.append(FaultRank(spec.name, spec.label, kl, kl_z, len(vals)))

    ranking.sort(key=lambda f: getattr(f, by), reverse=True)
    return ranking


# ============================================================
# CLI
# ============================================================

def _demo(shots: list, features_csv: str = FEATURES_CSV, fps: float = FPS):
    ref = load_reference()
    df = pd.read_csv(features_csv, low_memory=False)
    have = set(df["shot"].unique())
    for shot in shots:
        if shot not in have:
            print(f"\n{shot}: not in {features_csv}")
            continue
        sdf = df[df["shot"] == shot]
        ranking = rank_faults(sdf, _phases_from_row(sdf), ref, fps=fps)
        print(f"\n{shot}  — faults ranked by KL divergence from good form")
        print(f"  {'metric':18s} {'label':20s} {'KL':>8s} {'KL_z':>8s}  frames")
        print("  " + "-" * 60)
        for f in ranking:
            print(f"  {f.metric:18s} {f.label:20s} {f.kl:8.3f} {f.kl_z:8.2f}  {f.n_frames:3d}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build the good-form reference")
    ap.add_argument("--demo", nargs="*", metavar="SHOT", help="rank faults for these shots")
    args = ap.parse_args()

    if args.build:
        build_reference()
    if args.demo is not None:
        _demo(args.demo or ["form_42.mp4"])
    if not args.build and args.demo is None:
        ap.print_help()


if __name__ == "__main__":
    main()
