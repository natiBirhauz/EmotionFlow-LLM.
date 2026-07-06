"""
EmotionFlow-LLM Visualization
==============================
Shows how the network works and how emotions influence it.

Run:
    python visualize.py
"""

import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch

from emotionflow_llm import (
    EmotionFlowLLM, VOCAB_SIZE, EMOTIONS, LAYERS,
    EMOTION_DIM, MODEL_DIM, WORD_DIM, SEQ_LEN,
)

# ── Emotion colour palette ────────────────────────────────────────────────────
EMOTION_COLORS = {
    "joy":          "#FFD700",
    "sadness":      "#4169E1",
    "anger":        "#DC143C",
    "fear":         "#8B008B",
    "trust":        "#32CD32",
    "disgust":      "#556B2F",
    "surprise":     "#FF8C00",
    "anticipation": "#FF69B4",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_forward_with_hooks(model, tokens):
    """Run a forward pass and collect rich intermediate data."""
    data = {
        "word_emb": None,
        "emotion_emb": None,       # [B, S, 8]
        "emotion_bias": [],        # per-layer [B, S, dim]
        "attn_weights": [],        # per-layer [B, S, S]
        "layer_out": [],           # per-layer [B, S, dim]
    }

    hooks = []

    # Capture word + emotion embeddings
    def _emb_hook(module, inp, out):
        combined, emotions = out
        data["word_emb"]    = combined[..., :WORD_DIM].detach()
        data["emotion_emb"] = emotions.detach()

    hooks.append(model.embedding.register_forward_hook(_emb_hook))

    # Per-layer hooks
    for i, block in enumerate(model.layers):
        def _attn_hook(module, inp, out, idx=i):
            x, emotion = inp[0], inp[1]
            with torch.no_grad():
                bias  = module.emotion_proj(emotion)
                Q     = module.q(x) + bias
                K     = module.k(x) + bias
                sc    = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(module.dim)
                w     = torch.softmax(sc, dim=-1)
            data["attn_weights"].append(w.detach())
            data["emotion_bias"].append(bias.detach())

        hooks.append(block.attn.register_forward_hook(_attn_hook))

        def _block_hook(module, inp, out, idx=i):
            data["layer_out"].append(out.detach())

        hooks.append(block.register_forward_hook(_block_hook))

    with torch.no_grad():
        logits = model(tokens)

    for h in hooks:
        h.remove()

    data["logits"] = logits.detach()
    return data


def _emotion_vec(emotion_emb, batch=0):
    """Mean emotion vector across sequence → numpy [8]."""
    return emotion_emb[batch].mean(dim=0).numpy()


# ── Figure 1 – Architecture diagram ──────────────────────────────────────────

def plot_architecture(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("EmotionFlow-LLM Architecture", fontsize=14, fontweight="bold", pad=10)

    boxes = [
        (5, 13.0, "Input Tokens  [B, S]",          "#E8F4FD", 3.5, 0.55),
        (2, 11.2, "Word Embedding\n[B, S, 128]",    "#D5E8D4", 2.8, 0.8),
        (8, 11.2, "Emotion Encoder\n[B, S, 8]",     "#FFE6CC", 2.8, 0.8),
        (5,  9.4, "Concat → [B, S, 136]",           "#DAE8FC", 3.5, 0.55),
    ]

    layer_y = [7.8, 5.8, 3.8]
    layer_colors = ["#F8CECC", "#E1D5E7", "#D5E8D4"]

    for lx, ly, lc in zip([5, 5, 5], layer_y, layer_colors):
        rect = mpatches.FancyBboxPatch(
            (lx - 2.8, ly - 0.55), 5.6, 1.1,
            boxstyle="round,pad=0.08", linewidth=1.5,
            edgecolor="#555", facecolor=lc
        )
        ax.add_patch(rect)
        ax.text(lx, ly, f"TransformerBlock\n(EmotionAttention + FFN + LayerNorm)",
                ha="center", va="center", fontsize=7.5)

        # emotion arrow into each block
        ax.annotate("", xy=(lx + 2.8, ly), xytext=(lx + 4.2, ly),
                    arrowprops=dict(arrowstyle="->", color="#FF8C00", lw=1.5))
        ax.text(lx + 4.5, ly, "emotion\nbias", ha="left", va="center",
                fontsize=6.5, color="#FF8C00")

    for (bx, by, label, color, w, h) in boxes:
        rect = mpatches.FancyBboxPatch(
            (bx - w / 2, by - h / 2), w, h,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor="#555", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(bx, by, label, ha="center", va="center", fontsize=7.5)

    # Arrows: tokens → word emb + emotion enc
    ax.annotate("", xy=(2, 11.6), xytext=(4.2, 12.72),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    ax.annotate("", xy=(8, 11.6), xytext=(5.8, 12.72),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    # word emb + emotion enc → concat
    ax.annotate("", xy=(4.2, 9.67), xytext=(2, 10.8),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    ax.annotate("", xy=(5.8, 9.67), xytext=(8, 10.8),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    # concat → layer 1
    ax.annotate("", xy=(5, 8.35), xytext=(5, 9.12),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    # layer → layer
    for a, b in zip(layer_y[:-1], layer_y[1:]):
        ax.annotate("", xy=(5, b + 0.55), xytext=(5, a - 0.55),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    # LM head
    rect = mpatches.FancyBboxPatch((5 - 1.75, 2.45), 3.5, 0.55,
                                    boxstyle="round,pad=0.08", linewidth=1.2,
                                    edgecolor="#555", facecolor="#FFF2CC")
    ax.add_patch(rect)
    ax.text(5, 2.72, "LM Head  [B, S, VOCAB_SIZE]", ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(5, 3.0), xytext=(5, 3.25),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    # emotion encoder note
    ax.text(8, 10.3, "sigmoid → [0,1]\nvalidate(NaN/inf)", ha="center",
            va="top", fontsize=6, color="#888", style="italic")


# ── Figure 2 – Emotion radar ──────────────────────────────────────────────────

def plot_emotion_radar(ax, emotion_vec, title="Emotion Profile"):
    N = len(EMOTIONS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals = emotion_vec.tolist() + emotion_vec[:1].tolist()

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(EMOTIONS, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6)

    colors = [EMOTION_COLORS[e] for e in EMOTIONS]
    ax.plot(angles, vals, "o-", linewidth=2, color="#333")
    ax.fill(angles, vals, alpha=0.25, color="#4169E1")

    # colour each spoke
    for angle, val, color in zip(angles[:-1], emotion_vec, colors):
        ax.plot([angle, angle], [0, val], color=color, linewidth=3, alpha=0.8)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=15)


# ── Figure 3 – Emotion heatmap across sequence ───────────────────────────────

def plot_emotion_heatmap(ax, emotion_emb, seq_len=None, batch=0):
    """emotion_emb: [B, S, 8]"""
    data = emotion_emb[batch].numpy()          # [S, 8]
    if seq_len:
        data = data[:seq_len]

    im = ax.imshow(data.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(EMOTION_DIM))
    ax.set_yticklabels(EMOTIONS, fontsize=8)
    ax.set_xlabel("Token position", fontsize=9)
    ax.set_title("Emotion Vectors Across Sequence", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Intensity [0,1]")


# ── Figure 4 – Attention weight heatmaps ─────────────────────────────────────

def plot_attention_weights(axes, attn_weights_list, seq_len=8):
    """attn_weights_list: list of [B, S, S] per layer."""
    for i, (ax, w) in enumerate(zip(axes, attn_weights_list)):
        mat = w[0, :seq_len, :seq_len].numpy()
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max(),
                       interpolation="nearest")
        ax.set_title(f"Layer {i+1} Attention", fontsize=9, fontweight="bold")
        ax.set_xlabel("Key position", fontsize=8)
        ax.set_ylabel("Query position", fontsize=8)
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ── Figure 5 – Emotion bias magnitude per layer ───────────────────────────────

def plot_emotion_bias_magnitude(ax, emotion_bias_list, batch=0):
    """Show how strongly emotion shifts Q/K at each layer."""
    means = [b[batch].norm(dim=-1).mean().item() for b in emotion_bias_list]
    stds  = [b[batch].norm(dim=-1).std().item()  for b in emotion_bias_list]

    x = np.arange(1, LAYERS + 1)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=["#FF8C00", "#FF6347", "#DC143C"],
                  edgecolor="#333", linewidth=1.2, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i}" for i in x], fontsize=9)
    ax.set_ylabel("Mean ‖emotion bias‖", fontsize=9)
    ax.set_title("Emotion Bias Magnitude per Layer\n(how much emotion shifts Q & K)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)


# ── Figure 6 – Layer activation trajectory ───────────────────────────────────

def plot_activation_trajectory(ax, layer_outs, batch=0):
    """Mean activation magnitude across layers."""
    means = [lo[batch].norm(dim=-1).mean().item() for lo in layer_outs]
    stds  = [lo[batch].norm(dim=-1).std().item()  for lo in layer_outs]

    x = np.arange(1, LAYERS + 1)
    ax.errorbar(x, means, yerr=stds, fmt="o-", linewidth=2.5, markersize=8,
                capsize=5, color="#4169E1", ecolor="#AAA")
    ax.fill_between(x,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color="#4169E1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i}" for i in x], fontsize=9)
    ax.set_ylabel("Mean activation norm", fontsize=9)
    ax.set_title("Activation Trajectory Across Layers", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)


# ── Figure 7 – Emotion vs no-emotion comparison ──────────────────────────────

def plot_emotion_impact(ax, model, tokens):
    """Compare logit entropy with and without emotion influence."""
    # With emotion (normal forward)
    with torch.no_grad():
        logits_with = model(tokens)

    # Without emotion: zero out emotion embeddings
    original_weight = model.embedding.emotion_embedding.embedding.weight.data.clone()
    model.embedding.emotion_embedding.embedding.weight.data.zero_()
    with torch.no_grad():
        logits_without = model(tokens)
    model.embedding.emotion_embedding.embedding.weight.data = original_weight

    def entropy(logits):
        probs = torch.softmax(logits[0, :, :], dim=-1)
        ent   = -(probs * (probs + 1e-9).log()).sum(dim=-1)
        return ent.numpy()

    ent_with    = entropy(logits_with)
    ent_without = entropy(logits_without)
    positions   = np.arange(ent_with.shape[0])

    ax.plot(positions, ent_with,    "o-", label="With emotion",    color="#FF8C00", linewidth=2)
    ax.plot(positions, ent_without, "s--", label="Without emotion", color="#4169E1", linewidth=2)
    ax.fill_between(positions, ent_with, ent_without,
                    where=(ent_with > ent_without), alpha=0.15, color="#FF8C00",
                    label="Emotion increases entropy")
    ax.fill_between(positions, ent_with, ent_without,
                    where=(ent_with < ent_without), alpha=0.15, color="#4169E1",
                    label="Emotion decreases entropy")
    ax.set_xlabel("Token position", fontsize=9)
    ax.set_ylabel("Prediction entropy (nats)", fontsize=9)
    ax.set_title("Impact of Emotion on Output Distribution\n(entropy = uncertainty)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# ── Figure 8 – Dominant emotion per token ────────────────────────────────────

def plot_dominant_emotion_per_token(ax, emotion_emb, batch=0, max_tokens=16):
    data   = emotion_emb[batch, :max_tokens].numpy()   # [T, 8]
    dom    = data.argmax(axis=1)                        # [T]
    colors = [EMOTION_COLORS[EMOTIONS[d]] for d in dom]

    bars = ax.bar(range(len(dom)), data[np.arange(len(dom)), dom],
                  color=colors, edgecolor="#333", linewidth=0.8, alpha=0.9)

    ax.set_xticks(range(len(dom)))
    ax.set_xticklabels([f"t{i}" for i in range(len(dom))], fontsize=7)
    ax.set_ylabel("Dominant emotion intensity", fontsize=9)
    ax.set_title("Dominant Emotion per Token Position", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    legend_patches = [
        mpatches.Patch(color=EMOTION_COLORS[e], label=e) for e in EMOTIONS
    ]
    ax.legend(handles=legend_patches, fontsize=7, ncol=4,
              loc="upper right", framealpha=0.8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building model and running forward pass...")
    torch.manual_seed(42)
    model  = EmotionFlowLLM()
    model.eval()

    SEQ = 16
    tokens = torch.randint(0, VOCAB_SIZE, (1, SEQ))
    data   = _run_forward_with_hooks(model, tokens)

    emotion_vec = _emotion_vec(data["emotion_emb"])

    # ── Page 1: Architecture + emotion profile ────────────────────────────────
    fig1 = plt.figure(figsize=(16, 9))
    fig1.suptitle("EmotionFlow-LLM — Network Overview", fontsize=16, fontweight="bold")

    gs1 = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.35)
    ax_arch  = fig1.add_subplot(gs1[0])
    ax_radar = fig1.add_subplot(gs1[1], polar=True)

    plot_architecture(ax_arch)
    plot_emotion_radar(ax_radar, emotion_vec, title="Mean Emotion Profile\n(across sequence)")

    fig1.savefig("viz_architecture.png", dpi=150, bbox_inches="tight")
    print("  Saved viz_architecture.png")

    # ── Page 2: Emotion heatmap + dominant emotion per token ──────────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 9))
    fig2.suptitle("EmotionFlow-LLM — Emotion Encoding", fontsize=14, fontweight="bold")

    plot_emotion_heatmap(axes2[0], data["emotion_emb"], seq_len=SEQ)
    plot_dominant_emotion_per_token(axes2[1], data["emotion_emb"], max_tokens=SEQ)

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig("viz_emotion_encoding.png", dpi=150, bbox_inches="tight")
    print("  Saved viz_emotion_encoding.png")

    # ── Page 3: Attention weights per layer ───────────────────────────────────
    fig3, axes3 = plt.subplots(1, LAYERS, figsize=(5 * LAYERS, 5))
    fig3.suptitle("EmotionFlow-LLM — Attention Weights per Layer",
                  fontsize=14, fontweight="bold")

    plot_attention_weights(axes3, data["attn_weights"], seq_len=min(SEQ, 12))

    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    fig3.savefig("viz_attention.png", dpi=150, bbox_inches="tight")
    print("  Saved viz_attention.png")

    # ── Page 4: Emotion influence on the network ──────────────────────────────
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle("EmotionFlow-LLM — How Emotions Influence the Network",
                  fontsize=14, fontweight="bold")

    plot_emotion_bias_magnitude(axes4[0, 0], data["emotion_bias"])
    plot_activation_trajectory(axes4[0, 1], data["layer_out"])
    plot_emotion_impact(axes4[1, 0], model, tokens)

    # Emotion intensity distribution (violin-style via scatter)
    ax_dist = axes4[1, 1]
    emo_data = data["emotion_emb"][0].numpy()   # [S, 8]
    for i, (emo, color) in enumerate(zip(EMOTIONS, EMOTION_COLORS.values())):
        vals = emo_data[:, i]
        jitter = np.random.uniform(-0.2, 0.2, size=len(vals))
        ax_dist.scatter(np.full_like(vals, i) + jitter, vals,
                        color=color, alpha=0.6, s=20, edgecolors="none")
        ax_dist.plot([i - 0.3, i + 0.3], [vals.mean(), vals.mean()],
                     color="#333", linewidth=2)

    ax_dist.set_xticks(range(EMOTION_DIM))
    ax_dist.set_xticklabels(EMOTIONS, rotation=30, ha="right", fontsize=8)
    ax_dist.set_ylabel("Emotion intensity", fontsize=9)
    ax_dist.set_title("Emotion Intensity Distribution\n(each dot = one token position)",
                      fontsize=10, fontweight="bold")
    ax_dist.set_ylim(0, 1)
    ax_dist.grid(axis="y", alpha=0.3)

    fig4.tight_layout(rect=[0, 0, 1, 0.95])
    fig4.savefig("viz_emotion_influence.png", dpi=150, bbox_inches="tight")
    print("  Saved viz_emotion_influence.png")

    print("\nDone. Four PNG files saved:")
    print("  viz_architecture.png      — network diagram + emotion radar")
    print("  viz_emotion_encoding.png  — emotion heatmap + dominant emotion per token")
    print("  viz_attention.png         — attention weight matrices per layer")
    print("  viz_emotion_influence.png — emotion bias magnitude, activation trajectory,")
    print("                              entropy impact, intensity distribution")


if __name__ == "__main__":
    main()
