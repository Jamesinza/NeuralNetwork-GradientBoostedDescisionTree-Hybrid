# Neural Network & Gradient Boosted Descision Tree Hybrid

**Compact hybrid ensemble (Neural Net → GBDT residuals → meta-logit)**

This repository contains a single script prototype `hybrid_nn_gbdt_ensemble.py` that demonstrates a practical hybrid approach:
A compact TensorFlow neural network is trained first, its probability residuals are computed, and a LightGBM regressor is then trained to predict those residuals. Optionally, a small logistic meta-learner stacks the NN probability and the GBDT-predicted residual to produce the final probability.

> This script is intended as a clean, extensible base for experimentation (feature engineering, alternative stacking strategies, calibration, K-fold stacking, saving models, etc.). The included training logs show concrete results from a run on the UCI Adult dataset.

---

## Contents

- `hybrid_nn_gbdt_ensemble.py`: main script containing preprocessing, model definitions, training and lightweight ensembling logic.
- `datasets/adult.csv`: expected path to the UCI Adult dataset (included).

---

## Quick summary of the approach

1. **Preprocessing & feature engineering**:
   ***numeric features are normalized using*** `tf.keras.layers.Normalization()`
   - categorical columns are encoded as integer category codes and small embeddings are used in the NN branch;
   - a set of engineered columns (log transforms, boolean thresholds, ratio features, combined categorical codes) are created in `preprocess_adult()`;

2. **Neural network model**: 
   ***a compact, robust hybrid-friendly NN*** `build_nn_model`
   - a gated numeric block (learned gate × learned value)
   - small embedding table per categorical feature
   - a tiny Deep & Cross Network (DCN) style cross layer (`CrossLayer`) for low-cost explicit cross features
   - multiple residual branches (3 by default) with different activations and a controlled residual combination (shrinkage + alpha parameters)
   - outputs a single logit (trained with `BinaryCrossentropy(from_logits=True)`)

3. **GBDT residual model:**
   - LightGBM (`LGBMRegressor`) is trained to predict the pseudo-residuals computed on the probability scale `r = y - sigmoid(nn_logits)`
   - the GBDT operates on a simple matrix combining numeric values and categorical codes

4. **Ensembling / stacking**
   - final ensemble probability is `clip(p_nn + pred_r_gbdt, 0, 1)`
   - the script also demonstrates training a logistic meta-learner on `[p_nn, pred_r]` if you prefer a learned combiner

---

## Requirements

Create an environment (tested with Python 3.13) and install the core packages:

```bash
pip install tensorflow keras pandas scikit-learn lightgbm
```

---

## Usage

Place the UCI Adult CSV in `datasets/adult.csv` and run the script:

```bash
python hybrid_nn_gbdt_ensemble.py
```

The script will:
- preprocess the dataset
- train the NN (with early stopping on `val_auc`)
- predict probabilities and compute pseudo-residuals
- train LightGBM on residuals (with early stopping)
- print AUCs for the NN, the residual corrected ensemble, and the stacked meta-logit

---

## Reproducibility

- A global `SEED = 42` is set and used for `numpy`, `random`, and `tf.random` seeds. LightGBM is also given `random_state=SEED`.

---

## Results (from included training logs)

Run summary:

- **NN val AUC**: `0.9182574417518953`
- **Ensemble val AUC (NN + GBDT residuals)**: `0.9269955782729484`
- **Stacked meta-logit val AUC**: `0.9201303682040808`

LightGBM best iteration: `125` (early stopping on L2 residual loss).

The logs show steady NN training with `val_auc` around `0.917–0.918` and the GBDT residual stage provided a measurable uplift (~+0.009 absolute AUC in the validation split used).

---

## File level notes & design decisions

- `preprocess_adult()` performs both cleaning and feature engineering. It returns:
  - `X_num` (numpy float32 matrix for numeric features)
  - `X_cat` (numpy int32 matrix of categorical codes)
  - `y` target
  - a fitted `tf.keras.layers.Normalization()` instance (used directly inside the model)
  - `vocab_sizes`, `numeric_features`, `categorical_features` lists for reproducible mapping between arrays and model inputs.

- The NN uses small embedding dimensions computed with `min(embed_dim, max(3, int(vocab_size ** 0.25)))` to keep parameter counts down.

- The residual stacking uses probabilities (sigmoid of logits) rather than logits by design; this is simple and stable, but other choices are valid.

- The script converts categorical columns into integer codes and feeds those same codes to LightGBM as categorical features (via `categorical_feature` parameter in `LGBMRegressor.fit()`).

---

## Suggested next steps & enhancements

If you want to iterate quickly, here are high-leverage ideas:

- **K-Fold stacking**: Train the NN with K-fold cross-validated out-of-fold predictions and use those out-of-fold `p_nn`/residuals to train GBDT, avoiding a single holdout split.
- **Residual target variants**: Train the GBDT on logit-space residuals (`logit(y_pred)`) or on sample-weighted residuals; experiment which space gives better correction.
- **Calibrate**: Use temperature scaling or isotonic regression on the final probabilities to improve probability calibration.
- **Feature importance & interpretability**: Use SHAP/TreeSHAP on the GBDT residual model to see which features the trees use to correct the NN. This often surfaces systematic NN blindspots.
- **Save artifacts**: Add `model.save()` and `joblib`/pickle saving for LightGBM + meta models and a small `predict.py` evaluation script.
- **Hyperparameter search**: Use Optuna, scikit-optimize or KerasTuner for NN architecture, learning rate schedule, and LightGBM hyperparameters.
- **Alternate GBDT libraries**: Try `catboost` (robust categorical handling) or `xgboost` and compare.
- **Productionize**: Export a minimal inference pipeline that loads NN weights, loads the GBDT model, and executes the ensemble in a single API call.

---

## Troubleshooting tips

- If you see `nan` or exploding losses: check numeric ranges, missing values, and ensure `Normalization.adapt()` receives representative numeric data.
- LightGBM warnings about feature names: when passing NumPy arrays to a scikit-learn wrapper that was trained with column names, you can either train with arrays (no feature names) or pass DataFrame inputs consistently.
- If GPU nondeterminism is a concern, prefer CPU-only runs for reproducibility (This project was created using CPU only).

---

## License & Author

This repository is released under the **MIT License**: copy, modify, and experiment freely.

Author: James Sheldon (ML Developer, ZA)

---

## CONTRIBUTING

PRs welcome. If you make improvements, consider:
- splitting preprocessing into a dedicated module, adding tests for preprocessing edge-cases
- adding a `train.py` + `predict.py` split and small CLI
- adding automated unit tests for shape checks and example predictions
- consider modularising this script for easier reading and maintenance

---
