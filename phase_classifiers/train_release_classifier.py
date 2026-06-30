"""
Release Frame Classifier
========================
XGBoost per-frame classifier over the full video.
Honest GroupKFold CV evaluation vs heuristic baseline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
import joblib, warnings
warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent.parent
REF_CSV      = ROOT / "complete_features_v2.csv"
OUTPUT_DIR   = ROOT / "models" / "release"
RANDOM_STATE = 42
N_SPLITS     = 5

FEATURES = [
    "wrist_y_norm",          # normalized wrist height (0=lowest, 1=highest in video)
    "wrist_vy_norm",         # normalized wrist velocity
    "elbow_norm",            # normalized elbow angle
    "elbow_v",               # elbow rate of change
    "above_shoulder",        # 1 if wrist above shoulder
    "height_raw_norm",       # shoulder-wrist height normalized
    "temporal_progress",     # position in video (0=start, 1=end)
    "wrist_ay_norm",         # acceleration
    "elbow_ext",             # elbow extension score (clipped 120-175°)
    "elbow_stable",          # 1 if elbow not rapidly flexing
]


# ── Heuristic baseline (reproduce production logic) ──────────────────────────

def detect_release_heuristic(sdf):
    if 'wrist_y_s' not in sdf.columns:
        return None
    wrist_y   = sdf['wrist_y_s'].to_numpy(float)
    shoulder_y = sdf['shoulder_y_s'].to_numpy(float)
    elbow     = sdf['elbow_s'].to_numpy(float)
    height    = -wrist_y
    n         = len(sdf)

    prom = max(1e-6, np.std(height) * 0.25)
    peaks, _ = find_peaks(height, prominence=prom, distance=8)
    above = peaks[wrist_y[peaks] < shoulder_y[peaks]]
    if len(above) > 0:
        peaks = above
    if len(peaks) == 0:
        return int(np.argmax(height))

    K = 5
    peaks_sorted = peaks[np.argsort(height[peaks])]
    top_peaks    = peaks_sorted[-min(K, len(peaks_sorted)):]

    elbow_v = np.gradient(elbow)
    h_norm  = (height - height.min()) / (height.max() - height.min() + 1e-8)
    ext     = np.clip((elbow - 120.0) / (175.0 - 120.0), 0.0, 1.0)

    scores = np.zeros(n)
    for p in top_peaks:
        lo, hi = max(0, p - 4), min(n - 1, p + 4)
        idxs   = np.arange(lo, hi + 1)
        stable = (elbow_v[idxs] > -2.0).astype(float)
        s = (0.70 * h_norm[idxs] + 0.30 * ext[idxs]) * stable
        scores[idxs] = np.maximum(scores[idxs], s)

    peak_scores = [(p, scores[p]) for p in top_peaks]
    return int(max(peak_scores, key=lambda x: x[1])[0])


# ── Feature builder ───────────────────────────────────────────────────────────

def build_features(sdf):
    n          = len(sdf)
    wrist_y    = sdf['wrist_y_s'].to_numpy(float)
    shoulder_y = sdf['shoulder_y_s'].to_numpy(float)
    elbow      = sdf['elbow_s'].to_numpy(float)

    # Normalize within video
    wy_min, wy_max = wrist_y.min(), wrist_y.max()
    wrist_y_norm = (wrist_y - wy_min) / (wy_max - wy_min + 1e-8)

    wrist_vy      = np.gradient(wrist_y)
    wrist_ay      = np.gradient(wrist_vy)
    vy_min, vy_max = wrist_vy.min(), wrist_vy.max()
    wrist_vy_norm = (wrist_vy - vy_min) / (vy_max - vy_min + 1e-8)
    ay_min, ay_max = wrist_ay.min(), wrist_ay.max()
    wrist_ay_norm = (wrist_ay - ay_min) / (ay_max - ay_min + 1e-8)

    elbow_v    = np.gradient(elbow)
    e_min, e_max = elbow.min(), elbow.max()
    elbow_norm = (elbow - e_min) / (e_max - e_min + 1e-8)
    elbow_ext  = np.clip((elbow - 120.0) / (175.0 - 120.0), 0.0, 1.0)
    elbow_stable = (elbow_v > -2.0).astype(float)

    above_shoulder = (wrist_y < shoulder_y).astype(float)

    height     = shoulder_y - wrist_y
    h_min, h_max = height.min(), height.max()
    height_raw_norm = (height - h_min) / (h_max - h_min + 1e-8)

    temporal_progress = np.arange(n, dtype=float) / max(n - 1, 1)

    return np.column_stack([
        wrist_y_norm, wrist_vy_norm, elbow_norm, elbow_v,
        above_shoulder, height_raw_norm, temporal_progress,
        wrist_ay_norm, elbow_ext, elbow_stable,
    ])


# ── CV evaluation ─────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df_ref = pd.read_csv(REF_CSV, low_memory=False)
    per_shot = (
        df_ref.groupby('shot')['release_frame']
        .first().reset_index()
    )
    per_shot = per_shot[per_shot['release_frame'].notna()].reset_index(drop=True)
    print(f"  {len(per_shot)} shots\n")

    # Pre-cache
    shot_data = {}
    for _, row in per_shot.iterrows():
        shot  = row['shot']
        gt_r  = int(row['release_frame'])
        sdf   = df_ref[df_ref['shot'] == shot].copy().reset_index(drop=True)
        fc    = 'frame' if 'frame' in sdf.columns else 'Frame'
        X     = build_features(sdf)
        frames = sdf[fc].tolist()
        shot_data[shot] = dict(sdf=sdf, fc=fc, X=X, frames=frames, gt_r=gt_r)

    groups = np.array(list(shot_data.keys()))
    gkf    = GroupKFold(n_splits=N_SPLITS)

    heuristic_res = {}
    ml_res        = {}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(groups, groups=groups), 1):
        train_shots = groups[train_idx]
        test_shots  = groups[test_idx]

        X_tr, y_tr = [], []
        for s in train_shots:
            d = shot_data[s]
            X_tr.append(d['X'])
            y = np.zeros(len(d['frames']), dtype=int)
            for i, f in enumerate(d['frames']):
                if f == d['gt_r']: y[i] = 1
            y_tr.append(y)
        X_train = np.vstack(X_tr)
        y_train = np.concatenate(y_tr)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_train)
        pos    = max((y_train == 1).sum(), 1)
        model  = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / pos,
            random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0,
        )
        model.fit(X_sc, y_train)

        for s in test_shots:
            d     = shot_data[s]
            gt_r  = d['gt_r']
            fc    = d['fc']

            # Heuristic
            h_idx = detect_release_heuristic(d['sdf'])
            if h_idx is not None:
                h_frame = int(d['sdf'].iloc[h_idx][fc])
                heuristic_res[s] = (gt_r, h_frame, abs(h_frame - gt_r))

            # ML
            probs  = model.predict_proba(scaler.transform(d['X']))[:, 1]
            best_i = int(np.argmax(probs))
            ml_frame = int(d['frames'][best_i])
            ml_res[s] = (gt_r, ml_frame, abs(ml_frame - gt_r))

        print(f"  Fold {fold}: {len(test_shots)} test shots")

    def fmt(d):
        errs = [v[2] for v in d.values()]
        return (f"MAE={np.mean(errs):.2f}  med={np.median(errs):.0f}  "
                f"≤3f={sum(e<=3 for e in errs)/len(errs)*100:.0f}%  "
                f"≤5f={sum(e<=5 for e in errs)/len(errs)*100:.0f}%  "
                f"≤10f={sum(e<=10 for e in errs)/len(errs)*100:.0f}%  "
                f"n={len(errs)}")

    print(f"\n{'='*62}")
    print(f"RELEASE DETECTION  (GroupKFold CV, no leakage)")
    print(f"{'='*62}")
    print(f"  Heuristic : {fmt(heuristic_res)}")
    print(f"  ML model  : {fmt(ml_res)}")

    # Shots that changed
    print(f"\nShots that changed:")
    print(f"{'Shot':<25} {'GT':>5}  {'heur':>6} {'he':>4}  {'ml':>6} {'me':>4}  Δ")
    print("-"*58)
    changed = [(s, heuristic_res[s][0], heuristic_res[s][1], heuristic_res[s][2],
                ml_res[s][1], ml_res[s][2])
               for s in heuristic_res if s in ml_res
               and heuristic_res[s][1] != ml_res[s][1]]
    for s, gt, hf, he, mf, me in sorted(changed, key=lambda x: x[3]-x[5], reverse=True):
        flag = "✓" if he > me else "✗"
        print(f"{s:<25} {gt:>5}  {hf:>6} {he:>4}  {mf:>6} {me:>4}  {he-me:>+4} {flag}")

    # Train final model on all data and save
    print(f"\nTraining final model on all data...")
    all_X = np.vstack([shot_data[s]['X'] for s in shot_data])
    all_y = np.concatenate([
        np.array([1 if f == shot_data[s]['gt_r'] else 0
                  for f in shot_data[s]['frames']], dtype=int)
        for s in shot_data
    ])
    final_scaler = StandardScaler()
    final_X_sc   = final_scaler.fit_transform(all_X)
    pos = max((all_y == 1).sum(), 1)
    final_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(all_y == 0).sum() / pos,
        random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0,
    )
    final_model.fit(final_X_sc, all_y)
    joblib.dump(final_model,   f"{OUTPUT_DIR}/release_model.joblib")
    joblib.dump(final_scaler,  f"{OUTPUT_DIR}/release_scaler.joblib")
    joblib.dump(FEATURES,      f"{OUTPUT_DIR}/release_features.joblib")
    print(f"  Saved release_*.joblib")


if __name__ == "__main__":
    main()
