"""
Emotion Conditioning Demo

Examples showing how to use the emotion conditioning API
to control the emotional tone of generated text.
"""

import torch
from emotionflow_llm.emotion_flow_llm import EmotionFlowLLM
from emotionflow_llm.conditioning import (
    EmotionConditioner,
    EmotionConditioningConfig,
    create_emotion_config,
    normalize_emotions
)
from emotionflow_llm.tokenization import simple_tokenize, decode_tokens
from emotionflow_llm.utils import emotion_profile


def example_1_single_emotion():
    """Example 1: Generate text with a single dominant emotion."""
    print("=" * 70)
    print("Example 1: Single Emotion (Pure Joy)")
    print("=" * 70)
    
    # Load model
    model = EmotionFlowLLM()
    conditioner = EmotionConditioner(model)
    
    # Create prompt
    prompt_text = "The celebration was"
    prompt = simple_tokenize(prompt_text)
    
    # Configure for pure joy
    config = create_emotion_config(
        emotion_dict={"joy": 1.0},
        num_candidates=30,
        temperature=0.9
    )
    
    print(f"\nPrompt: '{prompt_text}'")
    print(f"Target emotion: {config.target_emotions}")
    print(f"\nGenerating {config.num_candidates} candidates, filtering for best matches...\n")
    
    # Generate
    outputs = conditioner.generate_conditioned(prompt, config)
    
    # Display results
    for i, output in enumerate(outputs[:3], 1):
        text = decode_tokens(output)
        emo = emotion_profile(model, output)
        print(f"Sample {i} (emotion: {emo}):")
        print(f"  {text}")
        print()


def example_2_mixed_emotions():
    """Example 2: Generate text with mixed emotions."""
    print("=" * 70)
    print("Example 2: Mixed Emotions (Fear + Sadness)")
    print("=" * 70)
    
    # Load model
    model = EmotionFlowLLM()
    conditioner = EmotionConditioner(model)
    
    # Create prompt
    prompt_text = "The lost wanderer"
    prompt = simple_tokenize(prompt_text)
    
    # Configure for fear + sadness mix
    config = create_emotion_config(
        emotion_dict={
            "fear": 0.6,
            "sadness": 0.4
        },
        num_candidates=30,
        temperature=0.9
    )
    
    print(f"\nPrompt: '{prompt_text}'")
    print(f"Target emotion: {config.target_emotions}")
    print(f"\nGenerating {config.num_candidates} candidates...\n")
    
    # Generate
    outputs = conditioner.generate_conditioned(prompt, config)
    
    # Display results
    for i, output in enumerate(outputs[:3], 1):
        text = decode_tokens(output)
        emo = emotion_profile(model, output)
        print(f"Sample {i} (emotion: {emo}):")
        print(f"  {text}")
        print()


def example_3_analytical_tone():
    """Example 3: Generate analytical/neutral text (trust + anticipation)."""
    print("=" * 70)
    print("Example 3: Analytical Tone (Trust + Anticipation)")
    print("=" * 70)
    
    # Load model
    model = EmotionFlowLLM()
    conditioner = EmotionConditioner(model)
    
    # Create prompt
    prompt_text = "The research team analyzed"
    prompt = simple_tokenize(prompt_text)
    
    # Configure for analytical tone
    config = create_emotion_config(
        emotion_dict={
            "trust": 0.5,
            "anticipation": 0.3,
            "surprise": 0.2
        },
        num_candidates=30,
        temperature=0.9
    )
    
    print(f"\nPrompt: '{prompt_text}'")
    print(f"Target emotion: {config.target_emotions}")
    print(f"\nGenerating {config.num_candidates} candidates...\n")
    
    # Generate
    outputs = conditioner.generate_conditioned(prompt, config)
    
    # Display results
    for i, output in enumerate(outputs[:3], 1):
        text = decode_tokens(output)
        emo = emotion_profile(model, output)
        print(f"Sample {i} (emotion: {emo}):")
        print(f"  {text}")
        print()


def example_4_comparison():
    """Example 4: Compare same prompt with different emotion targets."""
    print("=" * 70)
    print("Example 4: Comparing Different Emotional Tones")
    print("=" * 70)
    
    # Load model
    model = EmotionFlowLLM()
    conditioner = EmotionConditioner(model)
    
    # Create prompt
    prompt_text = "The discovery changed everything"
    prompt = simple_tokenize(prompt_text)
    
    # Different emotion profiles
    emotion_profiles = [
        ("Joyful", {"joy": 0.8, "surprise": 0.2}),
        ("Fearful", {"fear": 0.7, "anticipation": 0.3}),
        ("Angry", {"anger": 0.6, "disgust": 0.4}),
    ]
    
    print(f"\nPrompt: '{prompt_text}'")
    print(f"Comparing outputs with different emotional tones...\n")
    
    for profile_name, emotions in emotion_profiles:
        print(f"--- {profile_name} Profile: {emotions} ---")
        
        config = create_emotion_config(
            emotion_dict=emotions,
            num_candidates=20,
            temperature=0.9
        )
        
        outputs = conditioner.generate_conditioned(prompt, config)
        
        # Show best match
        if outputs:
            text = decode_tokens(outputs[0])
            emo = emotion_profile(model, outputs[0])
            print(f"Best match (emotion: {emo}):")
            print(f"  {text}")
        print()


def example_5_normalization():
    """Example 5: Demonstrate automatic emotion normalization."""
    print("=" * 70)
    print("Example 5: Automatic Emotion Normalization")
    print("=" * 70)
    
    # Raw emotion values that don't sum to 1.0
    raw_emotions = {
        "joy": 5.0,
        "trust": 3.0,
        "surprise": 2.0
    }
    
    print(f"\nRaw emotion values (don't sum to 1.0):")
    print(f"  {raw_emotions}")
    print(f"  Sum: {sum(raw_emotions.values())}")
    
    # Normalize
    normalized = normalize_emotions(raw_emotions)
    
    print(f"\nNormalized emotions (sum to 1.0):")
    print(f"  {normalized}")
    print(f"  Sum: {sum(normalized.values()):.3f}")
    
    # Use in config
    config = create_emotion_config(
        emotion_dict=raw_emotions,  # Will be auto-normalized
        normalize=True,
        num_candidates=20
    )
    
    print(f"\nConfig target emotions: {config.target_emotions}")
    print(f"Sum: {sum(config.target_emotions.values()):.3f}")


def example_6_validation():
    """Example 6: Demonstrate emotion validation."""
    print("=" * 70)
    print("Example 6: Emotion Validation")
    print("=" * 70)
    
    # Valid config
    print("\n✓ Valid configuration:")
    try:
        config = EmotionConditioningConfig(
            target_emotions={"joy": 0.5, "trust": 0.5},
            strategy="filter"
        )
        config.validate()
        print("  Config is valid!")
        print(f"  Target emotions: {config.target_emotions}")
    except ValueError as e:
        print(f"  Error: {e}")
    
    # Invalid: emotions don't sum to 1.0
    print("\n✗ Invalid configuration (doesn't sum to 1.0):")
    try:
        config = EmotionConditioningConfig(
            target_emotions={"joy": 0.3, "trust": 0.3},  # Sum = 0.6
            strategy="filter"
        )
        config.validate()
        print("  Config is valid!")
    except ValueError as e:
        print(f"  Error: {e}")
    
    # Invalid: emotion out of range
    print("\n✗ Invalid configuration (value out of range):")
    try:
        config = EmotionConditioningConfig(
            target_emotions={"joy": 1.5, "trust": -0.5},
            strategy="filter"
        )
        config.validate()
        print("  Config is valid!")
    except ValueError as e:
        print(f"  Error: {e}")
    
    # Invalid: all zeros
    print("\n✗ Invalid configuration (all zeros):")
    try:
        config = EmotionConditioningConfig(
            target_emotions={"joy": 0.0, "sadness": 0.0},
            strategy="filter"
        )
        config.validate()
        print("  Config is valid!")
    except ValueError as e:
        print(f"  Error: {e}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "EMOTION CONDITIONING EXAMPLES" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    examples = [
        example_1_single_emotion,
        example_2_mixed_emotions,
        example_3_analytical_tone,
        example_4_comparison,
        example_5_normalization,
        example_6_validation,
    ]
    
    for i, example_func in enumerate(examples, 1):
        if i > 1:
            input("\nPress Enter to continue to next example...")
        
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠️  Example {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
