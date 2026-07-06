"""
Quick test script to verify all components work together.
"""

import torch
print("Testing EmotionFlow-LLM implementation...")

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from emotionflow_llm.emotion_flow_llm import EmotionFlowLLM
    from emotionflow_llm.conditioning import EmotionConditioner, create_emotion_config
    from emotionflow_llm.multi_head_emotion_attention import MultiHeadEmotionAttention
    from emotionflow_llm.tokenization import simple_tokenize, decode_tokens
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Load model
print("\n2. Testing model loading...")
try:
    model = EmotionFlowLLM()
    model.eval()
    print(f"   ✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    exit(1)

# Test 3: Basic forward pass
print("\n3. Testing forward pass...")
try:
    tokens = torch.randint(0, 10000, (1, 10))
    with torch.no_grad():
        logits = model(tokens)
    print(f"   ✓ Forward pass successful (logits shape: {logits.shape})")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    exit(1)

# Test 4: Tokenization
print("\n4. Testing tokenization...")
try:
    text = "The hero saved the day"
    tokens = simple_tokenize(text)
    decoded = decode_tokens(tokens)
    print(f"   ✓ Tokenization successful")
    print(f"      Input: '{text}'")
    print(f"      Tokens shape: {tokens.shape}")
    print(f"      Decoded: '{decoded[:50]}...'")
except Exception as e:
    print(f"   ✗ Tokenization failed: {e}")
    exit(1)

# Test 5: Emotion conditioning config
print("\n5. Testing emotion conditioning config...")
try:
    config = create_emotion_config(
        emotion_dict={"joy": 0.6, "trust": 0.4},
        num_candidates=10,
        temperature=0.9
    )
    config.validate()
    print(f"   ✓ Config creation successful")
    print(f"      Target emotions: {config.target_emotions}")
    print(f"      Sum: {sum(config.target_emotions.values()):.3f}")
except Exception as e:
    print(f"   ✗ Config creation failed: {e}")
    exit(1)

# Test 6: Emotion conditioner
print("\n6. Testing emotion conditioner (quick test with small samples)...")
try:
    conditioner = EmotionConditioner(model)
    prompt = simple_tokenize("The hero")
    
    # Use very small test
    test_config = create_emotion_config(
        emotion_dict={"joy": 0.8, "trust": 0.2},
        num_candidates=5,  # Small number for fast test
        temperature=0.9,
        max_length=10  # Short length
    )
    
    outputs = conditioner.generate_conditioned(prompt, test_config)
    print(f"   ✓ Emotion conditioning successful")
    print(f"      Generated {len(outputs)} filtered samples")
    if outputs:
        decoded = decode_tokens(outputs[0])
        print(f"      Sample output: '{decoded[:50]}...'")
except Exception as e:
    print(f"   ✗ Emotion conditioning failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Multi-head attention
print("\n7. Testing multi-head emotion attention...")
try:
    mh_attn = MultiHeadEmotionAttention(model_dim=136, num_heads=8)
    x = torch.randn(2, 10, 136)
    emotion = torch.rand(2, 10, 8)
    out = mh_attn(x, emotion)
    print(f"   ✓ Multi-head attention successful")
    print(f"      Input shape: {x.shape}")
    print(f"      Output shape: {out.shape}")
except Exception as e:
    print(f"   ✗ Multi-head attention failed: {e}")
    exit(1)

# Test 8: Visualization functions
print("\n8. Testing visualization functions...")
try:
    from visualize import plot_emotion_distribution, plot_emotion_trajectory_for_ui
    
    # Generate some test outputs
    test_outputs = [torch.randint(0, 10000, (1, 15)) for _ in range(3)]
    
    # Test plot functions (don't show, just create)
    fig1 = plot_emotion_distribution(model, test_outputs)
    model(test_outputs[0])  # Run forward pass
    fig2 = plot_emotion_trajectory_for_ui(model)
    
    print(f"   ✓ Visualization functions successful")
    print(f"      Created {2} matplotlib figures")
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close('all')
except Exception as e:
    print(f"   ✗ Visualization failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - visualization is optional

print("\n" + "=" * 70)
print("✓ All core tests passed!")
print("=" * 70)
print("\nYou can now:")
print("  1. Run the Gradio UI:     python app.py")
print("  2. Try examples:          python examples/emotion_conditioning_demo.py")
print("  3. Run full demo:         python emotionflow_llm.py")
print("=" * 70)
