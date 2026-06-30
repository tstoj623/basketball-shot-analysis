# Basketball Shot Analysis

A computer-vision pipeline that analyses a basketball jump shot from a single video, flags potential form issues, and ranks them in order of significance, using a label-free **KL-divergence** approach.

## Pipeline

1. **Feature extraction** ([`feature_extractor.py`](feature_extractor.py)). MediaPipe 3D pose
   and YOLOv8 ball detection turn each frame into interpretable biomechanical metrics (joint
   angles, elbow flare, wrist alignment, knee valgus, kinetic-chain velocities).

2. **Phase detection** ([`phase_classifiers/`](phase_classifiers/)). Three per-frame
   **XGBoost** classifiers locate the three key moments of a shot: the **gather**, the
   **set point**, and the **release**. These define the windows the scorer looks at.

3. **KL fault ranking** ([`kl_scorer.py`](kl_scorer.py)). For each metric, take the whole
   per-frame distribution the joint traces out during the shot window and compare it to the
   pooled distribution of known-good shots using Kullback-Leibler divergence:

   ```
   p_good(x) = KDE over per-frame values from all known-good shots
   q_shot(x) = KDE over this shot's per-frame values
   D(metric) = KL( q_shot || p_good )
   ```
