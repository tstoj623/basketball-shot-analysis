"""
Gather Frame Classifier
=======================
Trains an XGBoost classifier to predict the gather frame.
Same architecture as set point classifier (GroupKFold, per-frame features).

Usage:
    python train_gather_classifier.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_recall_curve
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
REF_CSV    = ROOT / "complete_features_v2.csv"
OUTPUT_DIR = ROOT / "models" / "gather"
SEARCH_WINDOW = 60
RANDOM_STATE  = 42
N_SPLITS      = 5

FEATURES = [
    # Wrist position / motion
    "wrist_y_norm",          # normalized wrist height in window
    "height_progress",       # how far through the shot (0=gather, 1=release)
    "wrist_vy",              # wrist velocity
    "wrist_ay",              # wrist acceleration
    # Elbow
    "elbow_s",               # elbow angle (raw)
    "elbow_tightness",       # elbow - elbow_min in window (0 = most bent)
    "elbow_v",               # elbow angle rate of change
    # Zone
    "zone_score",            # 1.0=hip-nose, 0.15=above_nose, 0.05=below_hip
    # Shoulder/hip
    "height_raw",            # shoulder_y - wrist_y
    "h_score",               # how low wrist is vs window range (0-1)
    # Temporal
    "frames_to_release",     # frames remaining to release
    "temporal_progress",     # position in window (0=start, 1=release)
]


def build_dataset(df_ref):
    """
    Build per-frame training dataset from reference CSV.
    Each row = one frame in the gather search window.
    Label = 1 if frame is GT gather, 0 otherwise.
    """
    per_shot = (
        df_ref.groupby("shot")[["gather_frame", "release_frame"]]
        .first().reset_index()
    )
    per_shot = per_shot[
        per_shot["gather_frame"].notna() & per_shot["release_frame"].notna()
    ]

    rows = []
    for _, row in per_shot.iterrows():
        shot      = row["shot"]
        gt_gather = int(row["gather_frame"])
        gt_release = int(row["release_frame"])
        sdf = df_ref[df_ref["shot"] == shot].copy().reset_index(drop=True)
        frame_col = "frame" if "frame" in sdf.columns else "Frame"

        # Find release index
        rel_matches = sdf[sdf[frame_col] == gt_release]
        if rel_matches.empty:
            continue
        rel_idx = int(rel_matches.index[0] - sdf.index[0])

        start_idx = max(0, rel_idx - SEARCH_WINDOW)
        end_idx   = rel_idx
        n = end_idx - start_idx + 1
        if n < 5:
            continue

        seg = sdf.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)

        # Raw signals
        wrist_y   = seg["wrist_y_s"].to_numpy(float)
        shoulder_y = seg["shoulder_y_s"].to_numpy(float)
        elbow     = seg["elbow_s"].to_numpy(float)
        wrist_px  = seg["r_wrist_y_px"].to_numpy(float)
        hip_px    = seg["hip_mid_y"].to_numpy(float)
        nose_px   = seg["nose_y"].to_numpy(float)
        height    = shoulder_y - wrist_y

        # Derived
        wy_min, wy_max = wrist_y.min(), wrist_y.max()
        wrist_y_norm = (wrist_y - wy_min) / (wy_max - wy_min + 1e-8)

        # height_progress: 0 at gather, 1 at release
        h_range = max(wrist_y[0] - wrist_y[-1], 1e-6)
        height_progress = (wrist_y[0] - wrist_y) / h_range
        height_progress = np.clip(height_progress, 0, 1)

        elbow_min = elbow.min()
        elbow_tightness = elbow - elbow_min  # 0 = most bent

        elbow_v = np.gradient(elbow)
        wrist_vy = np.gradient(wrist_y)
        wrist_ay = np.gradient(wrist_vy)

        h_range2 = max(height.max() - height.min(), 1e-8)
        h_score = (height.max() - height) / h_range2  # high = wrist low

        zone_score = np.ones(n)
        for i in range(n):
            if wrist_px[i] < nose_px[i]:
                zone_score[i] = 0.15
            elif wrist_px[i] <= hip_px[i]:
                zone_score[i] = 1.0
            else:
                zone_score[i] = 0.05

        frames_to_release = np.arange(n - 1, -1, -1, dtype=float)
        temporal_progress = np.arange(n, dtype=float) / max(n - 1, 1)

        for i in range(n):
            f = int(seg[frame_col].iloc[i])
            label = 1 if f == gt_gather else 0
            rows.append({
                "shot":             shot,
                "frame":            f,
                "label":            label,
                "wrist_y_norm":     wrist_y_norm[i],
                "height_progress":  height_progress[i],
                "wrist_vy":         wrist_vy[i],
                "wrist_ay":         wrist_ay[i],
                "elbow_s":          elbow[i],
                "elbow_tightness":  elbow_tightness[i],
                "elbow_v":          elbow_v[i],
                "zone_score":       zone_score[i],
                "height_raw":       height[i],
                "h_score":          h_score[i],
                "frames_to_release": frames_to_release[i],
                "temporal_progress": temporal_progress[i],
            })

    return pd.DataFrame(rows)


def evaluate_model(df_data, model, scaler, threshold=0.5):
    """
    For each shot, predict gather = frame with highest probability.
    Report MAE vs GT.
    """
    per_shot = df_data.groupby("shot")
    results = []
    for shot, grp in per_shot:
        grp = grp.reset_index(drop=True)
        X = scaler.transform(grp[FEATURES].values)
        probs = model.predict_proba(X)[:, 1]
        best_i = int(np.argmax(probs))
        pred_frame = int(grp["frame"].iloc[best_i])
        gt_frame   = int(grp[grp["label"] == 1]["frame"].iloc[0]) if (grp["label"] == 1).any() else -1
        if gt_frame == -1:
            continue
        err = abs(pred_frame - gt_frame)
        results.append((shot, gt_frame, pred_frame, err, float(probs[best_i])))
    return results


def main():
    print("Loading data...")
    df_ref = pd.read_csv(REF_CSV, low_memory=False)
    df = build_dataset(df_ref)
    print(f"  {len(df)} frames from {df['shot'].nunique()} shots")
    print(f"  Positive frames: {df['label'].sum()}  ({df['label'].mean()*100:.1f}%)")

    X = df[FEATURES].values
    y = df["label"].values
    groups = df["shot"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )

    # Cross-validated predictions for honest evaluation
    print(f"\nRunning {N_SPLITS}-fold GroupKFold CV...")
    cv = GroupKFold(n_splits=N_SPLITS)
    probs_cv = cross_val_predict(model, X_scaled, y, groups=groups,
                                  cv=cv, method="predict_proba")[:, 1]

    # Find best threshold
    precision, recall, thresholds = precision_recall_curve(y, probs_cv)
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = float(thresholds[np.argmax(f1s)])
    print(f"  Best threshold: {best_thresh:.3f}  F1={f1s.max():.3f}")

    # Evaluate per-shot gather prediction using CV probs
    df["prob_cv"] = probs_cv
    results = []
    for shot, grp in df.groupby("shot"):
        grp = grp.reset_index(drop=True)
        best_i = int(np.argmax(grp["prob_cv"].values))
        pred_frame = int(grp["frame"].iloc[best_i])
        gt_frame   = int(grp[grp["label"] == 1]["frame"].iloc[0]) if (grp["label"] == 1).any() else -1
        if gt_frame == -1:
            continue
        err = abs(pred_frame - gt_frame)
        results.append((shot, gt_frame, pred_frame, err))

    errors = [r[3] for r in results]
    print(f"\n{'':=<55}")
    print(f"CV GATHER DETECTION RESULTS  ({len(results)} shots)")
    print(f"{'':=<55}")
    print(f"  MAE:        {np.mean(errors):.1f} frames")
    print(f"  Median:     {np.median(errors):.1f} frames")
    print(f"  Within  3f: {sum(e<=3 for e in errors)/len(errors)*100:.0f}%")
    print(f"  Within  5f: {sum(e<=5 for e in errors)/len(errors)*100:.0f}%")
    print(f"  Within 10f: {sum(e<=10 for e in errors)/len(errors)*100:.0f}%")

    print(f"\nTop 10 worst:")
    print(f"{'Shot':<22} {'GT':>5} {'Pred':>6} {'Err':>5}")
    print("-"*40)
    for shot, gt, pred, err in sorted(results, key=lambda x: x[3], reverse=True)[:10]:
        print(f"{shot:<22} {gt:>5} {pred:>6} {err:>5}")

    # Compare to current heuristic baseline
    from test_elbow_gather import detect_gather_current
    baseline_errors = []
    for shot, gt, pred, err in results:
        sdf = df_ref[df_ref['shot'] == shot].copy().reset_index(drop=True)
        gt_r = int(df_ref[df_ref['shot']==shot]['release_frame'].iloc[0])
        rel_idx = int((sdf['frame'] - gt_r).abs().argmin())
        b_idx = detect_gather_current(sdf, rel_idx)
        if b_idx is not None:
            b_f = int(sdf.iloc[b_idx]['frame'])
            baseline_errors.append(abs(b_f - gt))

    print(f"\nComparison:")
    print(f"  Heuristic MAE: {np.mean(baseline_errors):.1f}  ≤3f: {sum(e<=3 for e in baseline_errors)/len(baseline_errors)*100:.0f}%  ≤5f: {sum(e<=5 for e in baseline_errors)/len(baseline_errors)*100:.0f}%")
    print(f"  ML model  MAE: {np.mean(errors):.1f}  ≤3f: {sum(e<=3 for e in errors)/len(errors)*100:.0f}%  ≤5f: {sum(e<=5 for e in errors)/len(errors)*100:.0f}%")

    # Train final model on all data and save
    print(f"\nTraining final model on all data...")
    model.fit(X_scaled, y)

    joblib.dump(model,   OUTPUT_DIR / "gather_model.joblib")
    joblib.dump(scaler,  OUTPUT_DIR / "gather_scaler.joblib")
    joblib.dump(FEATURES, OUTPUT_DIR / "gather_features.joblib")
    joblib.dump(best_thresh, OUTPUT_DIR / "gather_threshold.joblib")
    print(f"  Saved to {OUTPUT_DIR}/gather_*.joblib")


if __name__ == "__main__":
    main()
