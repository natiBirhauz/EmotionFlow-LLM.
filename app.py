"""
EmotionFlow-LLM Gradio Web Interface

Interactive demo for emotion-conditioned text generation.
Allows users to control emotional tone through sliders.
"""

import gradio as gr
import torch
import numpy as np

from emotionflow_llm.emotion_flow_llm import EmotionFlowLLM
from emotionflow_llm.conditioning import (
    EmotionConditioner,
    EmotionConditioningConfig,
    create_emotion_config,
    EMOTIONS
)
from emotionflow_llm.tokenization import simple_tokenize, decode_tokens
from visualize import plot_emotion_distribution, plot_emotion_trajectory_for_ui


# Global model instance (cached)
_model = None
_conditioner = None


def get_model():
    """Load model once and cache it."""
    global _model, _conditioner
    if _model is None:
        print("Loading EmotionFlow-LLM model...")
        _model = EmotionFlowLLM()
        _model.eval()
        _conditioner = EmotionConditioner(_model)
        print("Model loaded successfully!")
    return _model, _conditioner


def generate_with_ui(
    prompt_text: str,
    joy: float,
    sadness: float,
    anger: float,
    fear: float,
    trust: float,
    disgust: float,
    surprise: float,
    anticipation: float,
    temperature: float,
    num_samples: int
):
    """
    Main generation function called by Gradio UI.
    
    Args:
        prompt_text: Input prompt text
        joy, sadness, etc.: Emotion slider values [0, 1]
        temperature: Sampling temperature
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (generated_text_markdown, emotion_distribution_plot, trajectory_plot)
    """
    try:
        # Validate inputs
        if not prompt_text or not prompt_text.strip():
            return "⚠️ Please enter a prompt", None, None
        
        # Create emotion dictionary
        emotions = {
            "joy": joy,
            "sadness": sadness,
            "anger": anger,
            "fear": fear,
            "trust": trust,
            "disgust": disgust,
            "surprise": surprise,
            "anticipation": anticipation
        }
        
        # Check if at least one emotion is > 0
        total = sum(emotions.values())
        if total == 0:
            return "⚠️ At least one emotion slider must be > 0", None, None
        
        # Load model
        model, conditioner = get_model()
        
        # Create conditioning config (auto-normalizes)
        config = create_emotion_config(
            emotion_dict=emotions,
            normalize=True,
            num_candidates=min(50, num_samples * 5),  # Generate 5x, keep best
            temperature=temperature,
            max_length=30
        )
        
        # Tokenize prompt
        prompt_tokens = simple_tokenize(prompt_text)
        
        # Generate conditioned text
        print(f"Generating {num_samples} samples with target emotions: {config.target_emotions}")
        outputs = conditioner.generate_conditioned(prompt_tokens, config)
        
        # Limit to requested number of samples
        outputs = outputs[:num_samples]
        
        if not outputs:
            return "⚠️ Generation failed - no outputs produced", None, None
        
        # Decode tokens to text
        generated_texts = []
        for i, output in enumerate(outputs):
            text = decode_tokens(output)
            generated_texts.append(f"**Sample {i+1}:**\n```\n{text}\n```")
        
        text_output = "\n\n".join(generated_texts)
        
        # Create visualizations
        try:
            # 1. Emotion distribution across samples
            emotion_dist_plot = plot_emotion_distribution(model, outputs)
            
            # 2. Emotion trajectory (use first sample)
            model(outputs[0])  # Run forward pass
            trajectory_plot = plot_emotion_trajectory_for_ui(model)
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
            emotion_dist_plot = None
            trajectory_plot = None
        
        return text_output, emotion_dist_plot, trajectory_plot
        
    except Exception as e:
        error_msg = f"⚠️ **Error:** {str(e)}\n\nPlease try again with different inputs."
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return error_msg, None, None


def build_interface():
    """Build and return the Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .emotion-sliders {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
    }
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    """
    
    with gr.Blocks(title="EmotionFlow-LLM Demo", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🎭 EmotionFlow-LLM: Emotion-Conditioned Text Generation
            
            Control the emotional tone of generated text by adjusting the emotion sliders below.
            The model will generate multiple candidates and select those closest to your target emotional profile.
            
            **How to use:**
            1. Enter a prompt (e.g., "The scientist discovered")
            2. Adjust emotion sliders to set desired emotional tone
            3. Click **Generate** and wait ~3-5 seconds
            4. View generated samples and emotion analysis
            """
        )
        
        with gr.Row():
            # Left column: Inputs
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="📝 Prompt",
                    placeholder="Enter your prompt here (e.g., 'The hero saved the day')...",
                    lines=3,
                    value="The hero saved the day"
                )
                
                gr.Markdown("### 🎨 Target Emotion Profile")
                gr.Markdown(
                    "*Adjust sliders to set desired emotions (will be auto-normalized to sum to 1.0)*"
                )
                
                # Emotion sliders with emojis
                with gr.Group():
                    joy = gr.Slider(0, 1, value=0.3, label="😊 Joy", step=0.05)
                    sadness = gr.Slider(0, 1, value=0.0, label="😢 Sadness", step=0.05)
                    anger = gr.Slider(0, 1, value=0.0, label="😠 Anger", step=0.05)
                    fear = gr.Slider(0, 1, value=0.0, label="😨 Fear", step=0.05)
                    trust = gr.Slider(0, 1, value=0.3, label="🤝 Trust", step=0.05)
                    disgust = gr.Slider(0, 1, value=0.0, label="🤢 Disgust", step=0.05)
                    surprise = gr.Slider(0, 1, value=0.2, label="😲 Surprise", step=0.05)
                    anticipation = gr.Slider(0, 1, value=0.2, label="🔮 Anticipation", step=0.05)
                
                gr.Markdown("### ⚙️ Generation Parameters")
                temperature = gr.Slider(
                    0.1, 2.0, value=0.9, step=0.1,
                    label="🌡️ Temperature",
                    info="Higher = more random, Lower = more focused"
                )
                num_samples = gr.Slider(
                    1, 10, value=5, step=1,
                    label="🔢 Number of Samples",
                    info="How many text samples to generate"
                )
                
                generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    ---
                    **Note:** Generation takes ~3-5 seconds on CPU. 
                    The model generates multiple candidates and filters for best emotional match.
                    """
                )
            
            # Right column: Outputs
            with gr.Column(scale=1):
                output_text = gr.Markdown(
                    label="📄 Generated Text",
                    value="*Generated samples will appear here...*"
                )
                
                gr.Markdown("### 📊 Emotion Analysis")
                
                with gr.Row():
                    emotion_dist_plot = gr.Plot(label="Emotion Distribution")
                
                with gr.Row():
                    trajectory_plot = gr.Plot(label="Emotion Trajectory Across Layers")
        
        # Examples section
        gr.Markdown("## 💡 Try These Examples")
        gr.Examples(
            examples=[
                # [prompt, joy, sad, anger, fear, trust, disgust, surprise, anticipation, temp, n_samples]
                [
                    "The hero saved the day",
                    0.6, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0,
                    0.9, 5
                ],
                [
                    "The dark forest whispered secrets",
                    0.0, 0.2, 0.0, 0.6, 0.0, 0.0, 0.2, 0.0,
                    0.9, 5
                ],
                [
                    "The scientist discovered something incredible",
                    0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.2,
                    0.9, 5
                ],
                [
                    "The villain's plan was revealed",
                    0.0, 0.1, 0.4, 0.3, 0.0, 0.2, 0.0, 0.0,
                    0.9, 5
                ],
                [
                    "The child laughed with pure delight",
                    0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0,
                    0.9, 5
                ],
            ],
            inputs=[
                prompt, joy, sadness, anger, fear, trust, disgust, 
                surprise, anticipation, temperature, num_samples
            ],
            label="Click an example to load it"
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            
            ### 🔬 About EmotionFlow-LLM
            
            EmotionFlow-LLM is an experimental transformer architecture that integrates 
            **8-dimensional emotion vectors** (based on Plutchik's Wheel of Emotions) into 
            the attention mechanism. This allows the model to reason about text with 
            emotional awareness.
            
            **Key Features:**
            - 🧠 Emotion-aware attention mechanism
            - 🎨 8 emotion dimensions (joy, sadness, anger, fear, trust, disgust, surprise, anticipation)
            - 🔍 Interpretable emotion trajectories across layers
            - 🎲 Multi-sample generation with emotion filtering
            
            **GitHub:** [natiBirhauz/EmotionFlow-LLM](https://github.com/natiBirhauz/EmotionFlow-LLM)
            
            ---
            *Model has 3.3M parameters and runs on CPU. Generation takes 3-5 seconds.*
            """
        )
        
        # Connect button to function
        generate_btn.click(
            fn=generate_with_ui,
            inputs=[
                prompt, joy, sadness, anger, fear, trust, disgust, 
                surprise, anticipation, temperature, num_samples
            ],
            outputs=[output_text, emotion_dist_plot, trajectory_plot]
        )
    
    return demo


def main():
    """Launch the Gradio interface."""
    print("=" * 70)
    print("EmotionFlow-LLM Gradio Interface")
    print("=" * 70)
    
    # Pre-load model
    get_model()
    
    # Build and launch interface
    demo = build_interface()
    
    print("\nLaunching Gradio interface...")
    print("Access the UI at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    demo.launch(
        share=False,  # Set to True to create public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
