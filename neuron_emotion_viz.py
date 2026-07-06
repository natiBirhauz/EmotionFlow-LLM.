"""
Neuron-Level Emotion Visualization
===================================
Visualize how individual neurons in hidden layers carry emotional information.
Each neuron is colored by its dominant emotion.

Run:
    python neuron_emotion_viz.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from emotionflow_llm import EmotionFlowLLM, VOCAB_SIZE, EMOTIONS, LAYERS, MODEL_DIM

# Emotion color palette
EMOTION_COLORS = {
    "joy":          "#FFD700",  # Gold
    "sadness":      "#4169E1",  # Royal Blue
    "anger":        "#DC143C",  # Crimson
    "fear":         "#8B008B",  # Dark Magenta
    "trust":        "#32CD32",  # Lime Green
    "disgust":      "#556B2F",  # Dark Olive Green
    "surprise":     "#FF8C00",  # Dark Orange
    "anticipation": "#FF69B4",  # Hot Pink
}


def extract_neuron_emotions(model, tokens):
    """
    Extract emotion information for each neuron in each layer.
    
    Returns:
        dict with keys:
            - 'layer_activations': List of [B, S, MODEL_DIM] tensors
            - 'emotion_embeddings': [B, S, 8] emotion vectors
            - 'neuron_emotions': List of [MODEL_DIM, 8] emotion correlations per layer
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        logits = model(tokens)
        
        # Get emotion embeddings
        _, emotion_emb = model.embedding(tokens)
        
        # Get layer activations
        layer_acts = model.get_activations()
    
    # For each layer, compute correlation between each neuron and each emotion
    neuron_emotions = []
    
    for layer_idx, activations in enumerate(layer_acts):
        # activations: [B, S, MODEL_DIM]
        # emotion_emb: [B, S, 8]
        
        # Flatten batch and sequence dims
        act_flat = activations.reshape(-1, MODEL_DIM)  # [B*S, MODEL_DIM]
        emo_flat = emotion_emb.reshape(-1, 8)          # [B*S, 8]
        
        # Compute correlation between each neuron and each emotion
        # neuron_emo_corr: [MODEL_DIM, 8]
        neuron_emo_corr = torch.zeros(MODEL_DIM, 8)
        
        for neuron_idx in range(MODEL_DIM):
            for emo_idx in range(8):
                # Pearson correlation coefficient
                neuron_vals = act_flat[:, neuron_idx]
                emo_vals = emo_flat[:, emo_idx]
                
                # Normalize
                neuron_norm = (neuron_vals - neuron_vals.mean()) / (neuron_vals.std() + 1e-8)
                emo_norm = (emo_vals - emo_vals.mean()) / (emo_vals.std() + 1e-8)
                
                # Correlation
                corr = (neuron_norm * emo_norm).mean()
                neuron_emo_corr[neuron_idx, emo_idx] = corr.abs()  # Use absolute correlation
        
        neuron_emotions.append(neuron_emo_corr)
    
    return {
        'layer_activations': layer_acts,
        'emotion_embeddings': emotion_emb,
        'neuron_emotions': neuron_emotions
    }


def visualize_neuron_emotions_network(neuron_emotions, output_path="neuron_network.png"):
    """
    Visualize neurons as a network where each neuron is colored by dominant emotion.
    Shows all layers side by side.
    """
    num_layers = len(neuron_emotions)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 10))
    if num_layers == 1:
        axes = [axes]
    
    fig.suptitle("Neuron-Level Emotion Distribution Across Layers\n(Each circle = one neuron, colored by dominant emotion)",
                 fontsize=16, fontweight='bold', y=0.98)
    
    for layer_idx, (ax, emo_corr) in enumerate(zip(axes, neuron_emotions)):
        # emo_corr: [MODEL_DIM, 8]
        
        # Get dominant emotion for each neuron
        dominant_emotions = emo_corr.argmax(dim=1).numpy()  # [MODEL_DIM]
        emotion_strengths = emo_corr.max(dim=1).values.numpy()  # [MODEL_DIM]
        
        # Arrange neurons in a grid
        grid_size = int(np.ceil(np.sqrt(MODEL_DIM)))
        
        # Create scatter plot
        positions = []
        colors = []
        sizes = []
        
        for neuron_idx in range(MODEL_DIM):
            row = neuron_idx // grid_size
            col = neuron_idx % grid_size
            
            positions.append([col, row])
            
            # Color by dominant emotion
            emo_name = EMOTIONS[dominant_emotions[neuron_idx]]
            colors.append(EMOTION_COLORS[emo_name])
            
            # Size by strength
            sizes.append(100 + 400 * emotion_strengths[neuron_idx])
        
        positions = np.array(positions)
        
        # Plot neurons
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidths=0.5)
        
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f"Layer {layer_idx + 1}\n({MODEL_DIM} neurons)", 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add grid for reference
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.3)
            ax.axvline(i - 0.5, color='gray', linewidth=0.3, alpha=0.3)
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=EMOTION_COLORS[emo], 
                                      edgecolor='black', label=emo.capitalize())
                      for emo in EMOTIONS]
    fig.legend(handles=legend_elements, loc='lower center', ncol=8, 
              fontsize=11, frameon=True, fancybox=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")


def visualize_neuron_emotion_heatmap(neuron_emotions, output_path="neuron_heatmap.png"):
    """
    Heatmap showing emotion correlation for each neuron across all layers.
    """
    num_layers = len(neuron_emotions)
    
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers))
    if num_layers == 1:
        axes = [axes]
    
    fig.suptitle("Neuron-Emotion Correlation Heatmaps\n(Rows = neurons, Columns = emotions)",
                 fontsize=14, fontweight='bold')
    
    for layer_idx, (ax, emo_corr) in enumerate(zip(axes, neuron_emotions)):
        # emo_corr: [MODEL_DIM, 8]
        data = emo_corr.numpy()
        
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=data.max(),
                      interpolation='nearest')
        
        ax.set_xlabel("Emotion", fontsize=10)
        ax.set_ylabel("Neuron Index", fontsize=10)
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(8))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha='right', fontsize=9)
        
        # Show neuron indices every 10
        ytick_pos = range(0, MODEL_DIM, max(1, MODEL_DIM // 10))
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels([str(i) for i in ytick_pos], fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Correlation strength')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")


def visualize_emotion_flow_sankey(neuron_emotions, output_path="emotion_flow.png"):
    """
    Show how dominant emotions change from layer to layer.
    """
    # Count dominant emotions per layer
    emotion_counts_per_layer = []
    
    for emo_corr in neuron_emotions:
        dominant = emo_corr.argmax(dim=1).numpy()
        counts = np.bincount(dominant, minlength=8)
        emotion_counts_per_layer.append(counts)
    
    emotion_counts_per_layer = np.array(emotion_counts_per_layer)  # [num_layers, 8]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(neuron_emotions))
    width = 0.6
    
    bottom = np.zeros(len(neuron_emotions))
    
    for emo_idx, emo_name in enumerate(EMOTIONS):
        counts = emotion_counts_per_layer[:, emo_idx]
        ax.bar(x, counts, width, label=emo_name.capitalize(), 
              bottom=bottom, color=EMOTION_COLORS[emo_name],
              edgecolor='black', linewidth=0.5)
        bottom += counts
        
        # Add percentage labels
        for layer_idx, count in enumerate(counts):
            if count > 0:
                percentage = 100 * count / MODEL_DIM
                y_pos = bottom[layer_idx] - count / 2
                if percentage > 5:  # Only show if > 5%
                    ax.text(layer_idx, y_pos, f'{percentage:.1f}%',
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           color='white' if percentage > 15 else 'black')
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Neurons', fontsize=12, fontweight='bold')
    ax.set_title('Emotion Distribution Evolution Across Layers\n(How neurons shift emotional alignment)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {i+1}' for i in x], fontsize=11)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")


def visualize_neuron_connectivity(neuron_emotions, output_path="neuron_connectivity.png"):
    """
    Show top neurons for each emotion and their connections across layers.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    fig.suptitle("Top 20 Neurons per Emotion Across Layers\n(Tracking neuron specialization)",
                 fontsize=14, fontweight='bold')
    
    for emo_idx, (emo_name, ax) in enumerate(zip(EMOTIONS, axes)):
        # For each layer, get top neurons for this emotion
        for layer_idx, emo_corr in enumerate(neuron_emotions):
            # Get correlation with this emotion
            correlations = emo_corr[:, emo_idx].numpy()
            
            # Get top 20 neurons
            top_neurons = np.argsort(correlations)[-20:][::-1]
            top_strengths = correlations[top_neurons]
            
            # Plot as horizontal lines
            y_positions = np.arange(20)
            x_start = np.full(20, layer_idx)
            x_end = np.full(20, layer_idx + 0.8)
            
            for i, (neuron_idx, strength) in enumerate(zip(top_neurons, top_strengths)):
                color = EMOTION_COLORS[emo_name]
                alpha = 0.3 + 0.7 * (strength / top_strengths[0])  # Fade by strength
                ax.plot([x_start[i], x_end[i]], [y_positions[i], y_positions[i]],
                       color=color, linewidth=3, alpha=alpha, solid_capstyle='round')
        
        ax.set_xlim(-0.1, len(neuron_emotions))
        ax.set_ylim(-1, 20)
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('Top 20 Neurons', fontsize=9)
        ax.set_title(f'{emo_name.capitalize()}', fontsize=11, fontweight='bold',
                    color=EMOTION_COLORS[emo_name])
        ax.set_xticks(range(len(neuron_emotions)))
        ax.set_xticklabels([f'L{i+1}' for i in range(len(neuron_emotions))], fontsize=8)
        ax.set_yticks([0, 9, 19])
        ax.set_yticklabels(['#1', '#10', '#20'], fontsize=8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f8f8')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")


def main():
    """Generate all neuron-level emotion visualizations."""
    print("=" * 70)
    print("Neuron-Level Emotion Visualization")
    print("=" * 70)
    
    # Initialize model
    print("\nInitializing model...")
    torch.manual_seed(42)
    model = EmotionFlowLLM()
    model.eval()
    
    # Create sample input
    SEQ_LEN = 16
    tokens = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
    print(f"Sample input: {tokens.shape}")
    
    # Extract neuron-emotion relationships
    print("\nAnalyzing neuron-emotion correlations...")
    data = extract_neuron_emotions(model, tokens)
    neuron_emotions = data['neuron_emotions']
    
    print(f"Analyzed {len(neuron_emotions)} layers with {MODEL_DIM} neurons each")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("  1/4: Neuron network diagram...")
    visualize_neuron_emotions_network(neuron_emotions, "neuron_network.png")
    
    print("  2/4: Neuron-emotion heatmaps...")
    visualize_neuron_emotion_heatmap(neuron_emotions, "neuron_heatmap.png")
    
    print("  3/4: Emotion flow across layers...")
    visualize_emotion_flow_sankey(neuron_emotions, "emotion_flow.png")
    
    print("  4/4: Neuron connectivity by emotion...")
    visualize_neuron_connectivity(neuron_emotions, "neuron_connectivity.png")
    
    print("\n" + "=" * 70)
    print("Complete! Generated 4 visualizations:")
    print("  - neuron_network.png: Neurons as colored circles by dominant emotion")
    print("  - neuron_heatmap.png: Correlation heatmap (neurons × emotions)")
    print("  - emotion_flow.png: How emotion distribution changes per layer")
    print("  - neuron_connectivity.png: Top neurons for each emotion across layers")
    print("=" * 70)


if __name__ == "__main__":
    main()
