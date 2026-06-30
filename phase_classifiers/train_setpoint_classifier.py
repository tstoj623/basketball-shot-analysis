"""
Set Point Classifier - Training Script
=======================================
Trains a classifier to predict set point frames from basketball shooting videos.

KEY: Only trains on frames WITHIN the gather→release window (using labels).
This matches inference, where we only predict within that window.

Z-FEATURES: Disabled by default. Z-gap is used as post-hoc filter, not model input.

Usage:
    python train_setpoint_classifier_v2.py

Outputs:
    - set_point_model.joblib
    - set_point_scaler.joblib
    - set_point_features.joblib
    - set_point_threshold.joblib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, precision_recall_curve
import joblib
import warnings
warnings.filterwarnings("ignore")

# Try XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using RandomForest")


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "shot_phase_dataset_v4.csv"
OUTPUT_DIR = Path(".")

# Model settings
USE_XGBOOST = True
RANDOM_STATE = 42
N_SPLITS = 5

# Z-features: DISABLED by default (used as post-hoc filter instead)
USE_Z_FEATURES_IN_MODEL = False


# ============================================================
# Window Extraction (using labels)
# ============================================================

def extract_windows_from_labels(df):
    """
    Extract gather→release windows using the phase labels.
    """
    all_windows = []
    
    for video_name, group in df.groupby("video"):
        group = group.sort_values("frame").reset_index(drop=True)
        
        pocket_frames = group[group["Is_Pocket"] == 1]
        if len(pocket_frames) == 0:
            print(f"   ⚠️ {video_name}: No pocket labels, skipping")
            continue
        gather_idx = pocket_frames.index[0]
        
        release_frames = group[group["Is_Release"] == 1]
        if len(release_frames) == 0:
            print(f"   ⚠️ {video_name}: No release labels, skipping")
            continue
        release_idx = release_frames.index[-1]
        
        if release_idx <= gather_idx:
            print(f"   ⚠️ {video_name}: Invalid window, skipping")
            continue
        
        window = group.iloc[gather_idx:release_idx + 1].copy()
        
        n_setpoint = window["Is_Set_Point"].sum()
        if n_setpoint == 0:
            print(f"   ⚠️ {video_name}: No set point in window, skipping")
            continue
        
        window_size = len(window)
        print(f"   ✓ {video_name}: window [{gather_idx}→{release_idx}] = {window_size} frames, {n_setpoint} set point frames")
        
        all_windows.append(window)
    
    if not all_windows:
        return pd.DataFrame()
    
    return pd.concat(all_windows, ignore_index=True)


# ============================================================
# Feature Engineering
# ============================================================

def engineer_features(df):
    """
    Engineer features for each video's window.
    Z-features are computed but stored separately for post-hoc use.
    """
    all_rows = []
    
    # Check if raw Z data exists in the source
    has_z_features = 'shoulder_z' in df.columns and 'wrist_z' in df.columns
    if has_z_features:
        print("   ✓ Z-depth data available (will be saved for post-hoc filtering)")
    
    # Check if shoulder_y / elbow_y exist
    has_shoulder_y = 'shoulder_y' in df.columns
    has_elbow_y = 'elbow_y' in df.columns
    if has_shoulder_y:
        print("   ✓ shoulder_y data available")
    else:
        print("   ⚠️ shoulder_y not in dataset, wrist_shoulder_y_diff will be skipped")
    if has_elbow_y:
        print("   ✓ elbow_y data available")
    else:
        print("   ⚠️ elbow_y not in dataset, wrist_elbow_y_diff will be skipped")

    # Check if nose_y exists
    has_nose_y = 'nose_y' in df.columns
    if has_nose_y:
        print("   ✓ nose_y data available")
    else:
        print("   ⚠️ nose_y not in dataset, wrist_nose_y_offset feature will be skipped")
    
    for video_name, group in df.groupby("video"):
        group = group.sort_values("frame").copy()
        n = len(group)
        if n < 5: continue
        
        # ... [Keep your existing normalizations/calculations] ...
        wrist_y = group["wrist_y"].values
        wrist_y_velocity = group["wrist_y_velocity"].values
        elbow_angle = group["elbow_angle"].values
        elbow_velocity = np.gradient(elbow_angle)
        wrist_y_acceleration = np.gradient(wrist_y_velocity)
        
        # Shoulder/elbow relative heights (if available)
        if has_shoulder_y:
            shoulder_y = group["shoulder_y"].values
            shoulder_wrist_y_diff = shoulder_y - wrist_y  # positive = wrist below shoulder
        else:
            shoulder_wrist_y_diff = np.zeros(n)

        if has_elbow_y:
            elbow_y = group["elbow_y"].values
            wrist_elbow_y_diff = elbow_y - wrist_y   # positive = wrist above elbow
            elbow_shoulder_y_diff = shoulder_y - elbow_y if has_shoulder_y else np.zeros(n)
        else:
            wrist_elbow_y_diff = np.zeros(n)
            elbow_shoulder_y_diff = np.zeros(n)
        
        # Wrist-nose vertical offset (if available)
        if has_nose_y:
            nose_y = group["nose_y"].values
            wrist_nose_y_offset = wrist_y - nose_y
        else:
            wrist_nose_y_offset = np.zeros(n)
        
        # Normalize wrist_y
        wrist_y_min, wrist_y_max = wrist_y.min(), wrist_y.max()
        if wrist_y_max - wrist_y_min > 1e-6:
            wrist_y_norm = (wrist_y - wrist_y_min) / (wrist_y_max - wrist_y_min)
        else:
            wrist_y_norm = np.full(n, 0.5)

        # Relative frame
        relative_frame = np.arange(n) / max(n - 1, 1)

        # Height progress
        gather_wrist_y = wrist_y[0]
        release_wrist_y = wrist_y[-1]
        height_range = gather_wrist_y - release_wrist_y
        if abs(height_range) > 1e-6:
            height_progress = (gather_wrist_y - wrist_y) / height_range
            height_progress = np.clip(height_progress, 0, 1.5)
        else:
            height_progress = np.full(n, 0.5)

        # Handle Z data safely
        if has_z_features:
            shoulder_z_vals = group["shoulder_z"].values
            wrist_z_vals = group["wrist_z"].values
            shoulder_wrist_z_gap = wrist_z_vals - shoulder_z_vals
            z_gap_velocity = np.gradient(shoulder_wrist_z_gap)
        else:
            # Fill with zeros if missing so the script doesn't crash
            shoulder_z_vals = np.zeros(n)
            wrist_z_vals = np.zeros(n)
            shoulder_wrist_z_gap = np.zeros(n)
            z_gap_velocity = np.zeros(n)
        
        features_df = pd.DataFrame({
            "video": video_name,
            "frame": group["frame"].values,
            
            # Core model features
            "wrist_y": wrist_y_norm,
            "wrist_y_velocity": wrist_y_velocity,
            "elbow_angle": elbow_angle,
            "elbow_velocity": elbow_velocity,
            "wrist_y_acceleration": wrist_y_acceleration,
            "relative_frame": relative_frame,
            "height_progress": height_progress,
            "shoulder_wrist_y_diff": shoulder_wrist_y_diff,
            "wrist_elbow_y_diff": wrist_elbow_y_diff,
            "elbow_shoulder_y_diff": elbow_shoulder_y_diff,
            "wrist_nose_y_offset": wrist_nose_y_offset,
            
            # --- FIX STARTS HERE ---
            # You must explicitly pass the RAW Z columns here
            "shoulder_z": shoulder_z_vals,
            "wrist_z": wrist_z_vals,
            # -----------------------

            # Derived Z features
            "shoulder_wrist_z_gap": shoulder_wrist_z_gap,
            "z_gap_velocity": z_gap_velocity,
            
            # Labels
            "Is_Set_Point": group["Is_Set_Point"].values,
        })
        
        all_rows.append(features_df)
    
    return pd.concat(all_rows, ignore_index=True)

# ============================================================
# Training
# ============================================================

def train_classifier(df_features):
    """
    Train the set point classifier.
    Z-features excluded from model by default.
    """
    # Core feature columns (NO z-features by default)
    feature_cols = [
        "wrist_y",
        "wrist_y_velocity",
        "elbow_angle",
        "elbow_velocity",
        "wrist_y_acceleration",
        #"relative_frame",
        #"height_progress",
        "shoulder_wrist_y_diff",
        "wrist_elbow_y_diff",
        "elbow_shoulder_y_diff",
       #"shoulder_z",
       #"wrist_z",
       #"shoulder_wrist_z_gap"
    ]
    

    
    # Optionally add z-features (disabled by default)
    if USE_Z_FEATURES_IN_MODEL:
        z_cols = ["shoulder_wrist_z_gap", "z_gap_velocity"]
        if all(c in df_features.columns for c in z_cols):
            feature_cols.extend(z_cols)
            print(f"\n   ⚠️ Z-features ENABLED in model (not recommended)")
    else:
        print(f"\n   ✓ Z-features EXCLUDED from model (will use for post-hoc filtering)")
    
    X = df_features[feature_cols].values
    y = df_features["Is_Set_Point"].values
    groups = df_features["video"].values
    
    print(f"\n📊 Training Data Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Positive (set point): {y.sum()} ({100*y.mean():.1f}%)")
    print(f"   Negative: {len(y) - y.sum()}")
    print(f"   Videos: {df_features['video'].nunique()}")
    print(f"   Features ({len(feature_cols)}): {feature_cols}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model
    if USE_XGBOOST:
        print(f"\n🚀 Using XGBoost...")
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
        )
    else:
        print(f"\n🌲 Using RandomForest...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    
    # Cross-validation
    n_unique = df_features["video"].nunique()
    n_splits = min(N_SPLITS, n_unique)
    print(f"\n🔄 Running {n_splits}-fold cross-validation...")
    
    cv = GroupKFold(n_splits=n_splits)
    
    y_proba = cross_val_predict(
        model, X_scaled, y,
        cv=cv, groups=groups,
        method="predict_proba"
    )[:, 1]
    
    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    
    print(f"\n📈 Cross-Validation Results:")
    print(f"   Best threshold: {best_threshold:.3f}")
    print(f"   Best F1 score: {f1_scores[best_idx]:.3f}")
    
    y_pred = (y_proba >= best_threshold).astype(int)
    print(f"\n   Classification Report:")
    print(classification_report(y, y_pred, target_names=["Not Set Point", "Set Point"]))

    # Event-level evaluation (argmax per video)
    print(f"\n🎯 Event-Level Evaluation (argmax per window):")
    exact = within_1 = within_2 = 0
    errors = []
    n_videos = 0
    df_features_copy = df_features.copy()
    df_features_copy["_proba"] = y_proba

    for video_name, group in df_features_copy.groupby("video"):
        sp_frames = group[group["Is_Set_Point"] == 1]["frame"].tolist()
        if not sp_frames:
            continue
        # True frame = middle of labeled run
        true_frame = int(np.median(sp_frames))
        # Predicted frame = argmax probability in window
        pred_frame = int(group.loc[group["_proba"].idxmax(), "frame"])
        err = abs(pred_frame - true_frame)
        errors.append(err)
        if err == 0: exact += 1
        if err <= 1: within_1 += 1
        if err <= 2: within_2 += 1
        n_videos += 1

    if n_videos > 0:
        print(f"   Videos evaluated: {n_videos}")
        print(f"   Exact match (±0):  {exact}/{n_videos} ({100*exact/n_videos:.1f}%)")
        print(f"   Within ±1 frame:   {within_1}/{n_videos} ({100*within_1/n_videos:.1f}%)")
        print(f"   Within ±2 frames:  {within_2}/{n_videos} ({100*within_2/n_videos:.1f}%)")
        print(f"   Mean abs error:    {np.mean(errors):.2f} frames")
        print(f"   Median abs error:  {np.median(errors):.1f} frames")

    # Train final model on all data
    print(f"\n🏋️ Training final model on all data...")
    model.fit(X_scaled, y)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        print(f"\n📊 Feature Importance:")
        for _, row in importance.iterrows():
            bar = "█" * int(row["importance"] * 40)
            print(f"   {row['feature']:25s} {row['importance']:.3f} {bar}")
    
    return model, scaler, feature_cols, best_threshold


def save_artifacts(model, scaler, features, threshold, output_dir):
    """Save model artifacts."""
    output_dir = Path(output_dir)
    prefix = "set_point"
    
    joblib.dump(model, output_dir / f"{prefix}_model.joblib")
    joblib.dump(scaler, output_dir / f"{prefix}_scaler.joblib")
    joblib.dump(features, output_dir / f"{prefix}_features.joblib")
    joblib.dump(threshold, output_dir / f"{prefix}_threshold.joblib")
    
    print(f"\n💾 Saved model artifacts:")
    print(f"   ✅ {prefix}_model.joblib")
    print(f"   ✅ {prefix}_scaler.joblib")
    print(f"   ✅ {prefix}_features.joblib")
    print(f"   ✅ {prefix}_threshold.joblib")


def main():
    print("=" * 60)
    print("Set Point Classifier - Training (v2)")
    print("Z-features: post-hoc filter only")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📂 Loading dataset: {DATASET_PATH}")
    
    if not Path(DATASET_PATH).exists():
        print(f"❌ Dataset not found!")
        return
    
    df = pd.read_csv(DATASET_PATH)
    print(f"   Loaded {len(df)} total rows from {df['video'].nunique()} videos")
    print(f"   Total set point frames: {df['Is_Set_Point'].sum()}")
    
    # Extract windows
    print(f"\n🎯 Extracting gather→release windows from labels...")
    df_windows = extract_windows_from_labels(df)
    
    if df_windows.empty:
        print("❌ No valid windows found!")
        return
    
    print(f"\n   Window extraction complete:")
    print(f"   Total frames in windows: {len(df_windows)}")
    print(f"   Videos with valid windows: {df_windows['video'].nunique()}")
    print(f"   Set point frames: {df_windows['Is_Set_Point'].sum()}")
    
    # Engineer features
    print(f"\n🔧 Engineering features...")
    df_features = engineer_features(df_windows)
    
    # Train
    model, scaler, features, threshold = train_classifier(df_features)
    
    # Save
    save_artifacts(model, scaler, features, threshold, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()