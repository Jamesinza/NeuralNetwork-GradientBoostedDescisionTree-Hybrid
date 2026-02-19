import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# Preprocessing
def preprocess_adult(df):
    df = df.copy()

    # Replace missing
    for col in ['workclass', 'occupation', 'native-country']:
        if col in df.columns:
            df[col] = df[col].replace('?', 'Unknown')

    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)

    # engineered features
    df["log_capital_gain"] = np.log1p(df.get("capital-gain", 0))
    df["log_capital_loss"] = np.log1p(df.get("capital-loss", 0))
    df["age_over_40"] = (df["age"] > 40).astype(int)
    df["age_over_50"] = (df["age"] > 50).astype(int)
    df["hours_over_45"] = (df["hours-per-week"] > 45).astype(int)
    df["hours_over_55"] = (df["hours-per-week"] > 55).astype(int)
    df["has_capital_gain"] = (df["capital-gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital-loss"] > 0).astype(int)

    df["edu_occ"] = df["education"].astype(str) + "_" + df["occupation"].astype(str)
    df["edu_occ"] = df["edu_occ"].astype("category").cat.codes

    df["mar_rel"] = df["marital-status"].astype(str) + "_" + df["relationship"].astype(str)
    df["mar_rel"] = df["mar_rel"].astype("category").cat.codes

    df["gain_per_age"] = df.get("capital-gain", 0) / (df.get("age", 0) + 1)
    df["gain_per_edu"] = df.get("capital-gain", 0) / (df.get("educational-num", 0) + 1)

    numeric_features = [
        'age', 'educational-num', 'hours-per-week',
        'capital-gain', 'capital-loss', 'log_capital_gain',
        'log_capital_loss', 'gain_per_age', 'gain_per_edu',
        'age_over_40', 'age_over_50', 'hours_over_45',
        'hours_over_55', 'has_capital_gain', 'has_capital_loss'
    ]

    categorical_features = [
        'workclass', 'education', 'marital-status',
        'occupation', 'relationship', 'race',
        'gender', 'native-country','edu_occ', 'mar_rel'
    ]

    # keep only available columns
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # encode categorical features as integer codes if not already
    vocab_sizes = {}
    for col in categorical_features:
        df[col] = df[col].astype("category")
        vocab_sizes[col] = int(df[col].nunique()) + 1
        df[col] = df[col].cat.codes.astype(int)

    X_num = df[numeric_features].astype("float32").values
    X_cat = df[categorical_features].astype("int32").values
    y = df['income'].astype("float32").values

    normalizer = layers.Normalization()
    normalizer.adapt(X_num)
    return X_num, X_cat, y, normalizer, vocab_sizes, numeric_features, categorical_features


# Build a compact NN model (gated + DCN + residual stacking simplified)
@tf.keras.utils.register_keras_serializable(package="Custom")
class CrossLayer(layers.Layer):
    def build(self, input_shape):
        dim = input_shape[-1]
        self.w = self.add_weight(shape=(dim, 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(dim,), initializer="zeros", trainable=True)
    def call(self, x0, xl):
        xw = tf.matmul(xl, self.w)
        return x0 * xw + self.b + xl

    
def gated_numeric_block(x, units):
    gate = layers.Dense(units, activation="sigmoid")(x)
    value = layers.Dense(units, activation="relu")(x)
    return gate * value

    
def build_nn_model(normalizer, vocab_sizes, numeric_features, categorical_features,
                   embed_dim=8, cross_layers=1, n_branches=3, shrinkage=0.04, alpha=0.06):
    # inputs
    numeric_input = layers.Input(shape=(len(numeric_features),), name="numeric")
    cat_inputs = []
    embeds = []
    for f in categorical_features:
        inp = layers.Input(shape=(1,), dtype="int32", name=f)
        cat_inputs.append(inp)
        dim = min(embed_dim, max(3, int(vocab_sizes[f] ** 0.25)))
        emb = layers.Embedding(input_dim=vocab_sizes[f], output_dim=dim,
                               embeddings_regularizer=regularizers.l2(1e-4))(inp)
        emb = layers.Flatten()(emb)
        emb = layers.Dropout(0.15)(emb)
        embeds.append(emb)

    x_num = normalizer(numeric_input)
    x_num = gated_numeric_block(x_num, 32)
    x_num = layers.Dense(32, activation="relu")(x_num)

    x_base = layers.Concatenate()([x_num] + embeds)
    # small DCN
    x0 = x_base
    xl = x_base
    for _ in range(cross_layers):
        xl = CrossLayer()(x0, xl)
    x_cross = xl

    cumulative_logit = None
    branch_logits = []
    activations = ["relu", "gelu", "tanh", "selu"]
    for i in range(n_branches):
        branch_feat = layers.Dropout(0.18 + 0.02*i)(x_base)
        if i > 0:
            branch_feat = layers.Concatenate()([branch_feat, x_cross])
        if cumulative_logit is not None:
            residual_signal = layers.Lambda(lambda t: tf.stop_gradient(t))(cumulative_logit)
            branch_feat = layers.Concatenate()([branch_feat, residual_signal])
        h = layers.Dense(128, activation=activations[i % len(activations)],
                         kernel_regularizer=regularizers.l2(1e-4))(branch_feat)
        h = layers.Dropout(0.3)(h)
        logit_i = layers.Dense(1, kernel_initializer="zeros")(h)
        branch_logits.append(logit_i)
        if cumulative_logit is None:
            cumulative_logit = logit_i
        else:
            cumulative_logit = cumulative_logit + alpha * logit_i + shrinkage * (logit_i * cumulative_logit)

    model = models.Model(inputs=[numeric_input] + cat_inputs, outputs=cumulative_logit, name="nn_boost")
    return model


# Utility: build feature matrix for GBDT (numeric + categorical codes)
def build_gbdt_matrix(X_num, X_cat, numeric_features, categorical_features):
    # stack numeric then categorical codes (both numpy arrays)
    # returns X (n, D) and list of categorical column indices (relative to X)
    X = np.hstack([X_num, X_cat])
    n_num = X_num.shape[1]
    cat_indices = list(range(n_num, X.shape[1]))
    return X, cat_indices


# Main training + hybrid ensemble flow
def run_hybrid_pipeline(csv_path="datasets/adult.csv"):
    df = pd.read_csv(csv_path)
    X_num, X_cat, y, normalizer, vocab_sizes, num_feats, cat_feats = preprocess_adult(df)

    # split train / validation for ensemble training
    Xn_train, Xn_val, Xc_train, Xc_val, y_train, y_val = train_test_split(
        X_num, X_cat, y, test_size=0.10, random_state=SEED, stratify=y
    )

    # Build NN
    nn = build_nn_model(normalizer, vocab_sizes, num_feats, cat_feats,
                        embed_dim=8, cross_layers=1, n_branches=3, shrinkage=0.04, alpha=0.06)

    nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.AUC(name="auc", from_logits=True)]
    )
    nn.summary()

    # Prepare dict inputs
    def make_input_dict(X_num_arr, X_cat_arr):
        d = {"numeric": X_num_arr}
        for i, col in enumerate(cat_feats):
            d[col] = X_cat_arr[:, i]
        return d

    train_inputs = make_input_dict(Xn_train, Xc_train)
    val_inputs = make_input_dict(Xn_val, Xc_val)

    # Train NN
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=20, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.9, patience=5, verbose=1)
    ]

    nn.fit(train_inputs, y_train, validation_data=(val_inputs, y_val),
           epochs=1000, batch_size=128, callbacks=callbacks, verbose=1)

    # Predictions from NN (logits -> probabilities)
    logits_train = nn.predict(train_inputs, batch_size=1024)
    logits_val = nn.predict(val_inputs, batch_size=1024)
    p_train = tf.math.sigmoid(logits_train).numpy().ravel()
    p_val = tf.math.sigmoid(logits_val).numpy().ravel()

    print("NN val AUC:", roc_auc_score(y_val, p_val))

    # Pseudo-residuals on probability scale
    r_train = (y_train - p_train).astype("float32")
    r_val = (y_val - p_val).astype("float32")

    # Build GBDT feature matrix (numeric + categorical codes)
    X_train_gbdt, cat_cols = build_gbdt_matrix(Xn_train, Xc_train, num_feats, cat_feats)
    X_val_gbdt, _ = build_gbdt_matrix(Xn_val, Xc_val, num_feats, cat_feats)

    # LightGBM regressor to predict residuals
    gbdt = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=64,
        random_state=SEED,
        n_jobs=-1
    )

    # Fit with early stopping on residuals
    gbdt.fit(
        X_train_gbdt, r_train,
        eval_set=[(X_val_gbdt, r_val)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
        categorical_feature=cat_cols,
    )

    # Predictions from GBDT (predicted residuals)
    pred_r_val = gbdt.predict(X_val_gbdt, num_iteration=gbdt.best_iteration_)
    pred_r_train = gbdt.predict(X_train_gbdt, num_iteration=gbdt.best_iteration_)

    # Ensemble: add residual prediction to NN probability, clip to [0,1]
    p_val_ens = np.clip(p_val + pred_r_val, 0.0, 1.0)
    p_train_ens = np.clip(p_train + pred_r_train, 0.0, 1.0)

    print("NN  val AUC: {:.6f}".format(roc_auc_score(y_val, p_val)))
    print("GBDT residuals -> ensemble val AUC: {:.6f}".format(roc_auc_score(y_val, p_val_ens)))

    # Optional: train a small logistic meta-learner on [p_nn, pred_r]
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(np.vstack([p_train, pred_r_train]).T, y_train)
    p_meta_val = meta.predict_proba(np.vstack([p_val, pred_r_val]).T)[:,1]
    print("Stacked meta-logit val AUC: {:.6f}".format(roc_auc_score(y_val, p_meta_val)))

    return {
        "nn": nn,
        "gbdt": gbdt,
        "meta": meta,
        "results": {
            "nn_val_auc": roc_auc_score(y_val, p_val),
            "ens_val_auc": roc_auc_score(y_val, p_val_ens),
            "meta_val_auc": roc_auc_score(y_val, p_meta_val)
        }
    }

# Run
if __name__ == "__main__":
    out = run_hybrid_pipeline("datasets/adult.csv")
    print("Done. Results:", out["results"])
