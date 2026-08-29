"""Keras port of NAF's WorldModelLSTM (src/model/lstm.py).

Dual-head: a classification head predicts the next window's state class
(BENIGN/DDoS/DoS/PortScan -> infiltration probability + kill-chain stage),
and a regression head predicts the next window's raw 28-d feature vector
so `rollout()` can feed a predicted state back in and keep simulating —
that is what makes it a forward-simulating world model rather than a
one-step classifier. Attention weights over the input windows are exposed
for the "top contributing windows" explanation.
"""
from __future__ import annotations

import numpy as np

from .config import HIDDEN_DIM, INPUT_DIM, N_CLASSES, SEQUENCE_LENGTH


def _tf():
    import tensorflow as tf  # local import — TF is heavy
    return tf


ATTN_LAYER_NAME = "attn_weights"


def build_world_model(input_dim: int = INPUT_DIM, seq_len: int = SEQUENCE_LENGTH,
                      hidden_dim: int = HIDDEN_DIM, n_classes: int = N_CLASSES):
    tf = _tf()
    L = tf.keras.layers
    inp = L.Input(shape=(seq_len, input_dim), name="window_seq")
    outputs = L.LSTM(hidden_dim, return_sequences=True, name="lstm")(inp)
    scores = L.Dense(1, name="attn_score")(outputs)           # [B, T, 1]
    scores = L.Softmax(axis=1, name=ATTN_LAYER_NAME)(scores)  # over time
    context = L.Lambda(
        lambda z: tf.reduce_sum(z[0] * z[1], axis=1), name="context"
    )([outputs, scores])
    class_probs = L.Dense(n_classes, activation="softmax", name="class_probs")(context)
    next_state = L.Dense(input_dim, name="next_state")(context)
    return tf.keras.Model(inp, [class_probs, next_state], name="world_model_lstm")


def make_attention_model(model):
    tf = _tf()
    return tf.keras.Model(
        model.input, model.get_layer(ATTN_LAYER_NAME).output, name="attn_view"
    )


def rollout(model, initial_window_seq: np.ndarray, k_steps: int):
    """Autoregressive K-step forward simulation.

    initial_window_seq: scaled array [1, seq_len, input_dim] of real past
    window state vectors.
    Returns (stage_probs [k_steps, n_classes], next_states [k_steps, input_dim],
    attn_per_step [k_steps, seq_len]). Step k>1 is conditioned on the
    model's own earlier predicted state vectors, not ground truth.
    """
    attn_model = make_attention_model(model)
    seq = np.array(initial_window_seq, dtype=np.float32, copy=True)
    stage_probs, next_states, attn_hist = [], [], []
    for _ in range(int(k_steps)):
        probs, nxt = model.predict(seq, verbose=0)
        attn = attn_model.predict(seq, verbose=0)
        stage_probs.append(np.asarray(probs)[0])
        next_states.append(np.asarray(nxt)[0])
        attn_hist.append(np.asarray(attn)[0].reshape(-1))
        seq = np.concatenate([seq[:, 1:, :], np.asarray(nxt)[None, :, :]], axis=1)
    return (
        np.stack(stage_probs),
        np.stack(next_states),
        np.stack(attn_hist),
    )
