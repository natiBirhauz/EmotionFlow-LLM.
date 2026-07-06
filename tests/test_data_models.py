"""Tests for core data models."""

import pytest
import torch
from hypothesis import given, settings
import hypothesis.strategies as st

from emotionflow_llm.data_models import (
    EmotionVector,
    GenerationConfig,
    ModelConfig,
    VisualizationConfig,
    PipelineResult,
)


class TestEmotionVector:
    """Tests for EmotionVector dataclass."""
    
    def test_to_tensor(self):
        """Test conversion to tensor."""
        ev = EmotionVector(
            joy=1.0, sadness=0.0, anger=0.0, fear=0.0,
            trust=0.5, disgust=0.0, surprise=0.0, anticipation=0.0
        )
        tensor = ev.to_tensor()
        
        assert tensor.shape == (8,)
        assert tensor.dtype == torch.float32
        assert tensor[0].item() == 1.0  # joy
        assert tensor[4].item() == 0.5  # trust
    
    def test_validate_valid(self):
        """Test validation with valid values."""
        ev = EmotionVector(
            joy=0.5, sadness=0.3, anger=0.1, fear=0.2,
            trust=0.8, disgust=0.0, surprise=1.0, anticipation=0.4
        )
        assert ev.validate() is True
    
    def test_validate_invalid(self):
        """Test validation with invalid values."""
        ev = EmotionVector(
            joy=1.5, sadness=0.3, anger=0.1, fear=0.2,
            trust=0.8, disgust=-0.1, surprise=1.0, anticipation=0.4
        )
        assert ev.validate() is False
    
    def test_dominant_emotion(self):
        """Test dominant emotion detection."""
        ev = EmotionVector(
            joy=0.9, sadness=0.1, anger=0.0, fear=0.0,
            trust=0.2, disgust=0.0, surprise=0.0, anticipation=0.0
        )
        assert ev.dominant_emotion() == 'joy'
        
        ev2 = EmotionVector(
            joy=0.1, sadness=0.1, anger=0.0, fear=0.95,
            trust=0.2, disgust=0.0, surprise=0.0, anticipation=0.0
        )
        assert ev2.dominant_emotion() == 'fear'


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.num_samples == 20
        assert config.max_length == 100
        assert config.strategy == "temperature"
        assert config.temperature == 0.8
        assert config.top_p == 0.9
        assert config.top_k == 50
    
    def test_validate_valid(self):
        """Test validation with valid parameters."""
        config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            top_k=40
        )
        config.validate()  # Should not raise
    
    def test_validate_invalid_temperature(self):
        """Test validation with invalid temperature."""
        config = GenerationConfig(temperature=3.0)
        with pytest.raises(ValueError, match="temperature must be in"):
            config.validate()
    
    def test_validate_invalid_top_p(self):
        """Test validation with invalid top_p."""
        config = GenerationConfig(top_p=1.5)
        with pytest.raises(ValueError, match="top_p must be in"):
            config.validate()
    
    def test_validate_invalid_top_k(self):
        """Test validation with invalid top_k."""
        config = GenerationConfig(top_k=150)
        with pytest.raises(ValueError, match="top_k must be in"):
            config.validate()
    
    def test_validate_invalid_strategy(self):
        """Test validation with invalid strategy."""
        config = GenerationConfig(strategy="invalid")
        with pytest.raises(ValueError, match="strategy must be"):
            config.validate()


class TestModelConfig:
    """Tests for ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.vocab_size == 10000
        assert config.word_dim == 128
        assert config.emotion_dim == 8
        assert config.num_layers == 3
        assert config.num_heads == 4
        assert config.hidden_dim == 136
        assert config.emotion_enabled is True
        assert config.lambda_emotion == 0.1


class TestVisualizationConfig:
    """Tests for VisualizationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = VisualizationConfig()
        assert config.method == "pca"
        assert config.output_format == "png"
        assert config.dpi == 300
        assert config.figsize == (10, 8)
        assert config.show_tokens is False
        assert config.color_scheme == "viridis"


@pytest.mark.property
class TestEmotionVectorProperties:
    """Property-based tests for EmotionVector."""
    
    @settings(max_examples=100)
    @given(
        joy=st.floats(0.0, 1.0),
        sadness=st.floats(0.0, 1.0),
        anger=st.floats(0.0, 1.0),
        fear=st.floats(0.0, 1.0),
        trust=st.floats(0.0, 1.0),
        disgust=st.floats(0.0, 1.0),
        surprise=st.floats(0.0, 1.0),
        anticipation=st.floats(0.0, 1.0),
    )
    def test_property_2_emotion_vector_normalization(
        self, joy, sadness, anger, fear, trust, disgust, surprise, anticipation
    ):
        """
        Feature: emotionflow-llm, Property 2: Emotion Vector Normalization
        **Validates: Requirements 1.2**
        
        For any generated emotion vector, all values SHALL be in the range [0, 1].
        """
        ev = EmotionVector(
            joy=joy, sadness=sadness, anger=anger, fear=fear,
            trust=trust, disgust=disgust, surprise=surprise, anticipation=anticipation
        )
        
        # All values should be in [0, 1]
        assert ev.validate() is True
        
        # Verify tensor conversion maintains range
        tensor = ev.to_tensor()
        assert torch.all(tensor >= 0.0)
        assert torch.all(tensor <= 1.0)
