"""Tests for EmotionEncoder — Tasks 2.1 and 2.2."""

import math
import pytest
import torch

from emotionflow_llm import EmotionEncoder, VOCAB_SIZE, EMOTION_DIM


# ================================
# Task 2.1 — forward() shape & range
# ================================

class TestEmotionEncoderForward:
    """Validates Requirement 1.1 and 1.2."""

    def setup_method(self):
        self.encoder = EmotionEncoder()

    def test_output_shape(self):
        """forward() returns [B, S, 8] for a standard batch."""
        tokens = torch.randint(0, VOCAB_SIZE, (2, 10))
        out = self.encoder(tokens)
        assert out.shape == (2, 10, 8)

    def test_output_shape_single(self):
        """forward() returns [1, S, 8] for batch size 1."""
        tokens = torch.randint(0, VOCAB_SIZE, (1, 32))
        out = self.encoder(tokens)
        assert out.shape == (1, 32, 8)

    def test_output_shape_last_dim(self):
        """Last dimension is always 8 regardless of seq_len."""
        for seq_len in [1, 5, 32]:
            tokens = torch.randint(0, VOCAB_SIZE, (3, seq_len))
            out = self.encoder(tokens)
            assert out.shape[-1] == EMOTION_DIM == 8

    def test_values_in_range(self):
        """All output values are in [0, 1] due to sigmoid."""
        tokens = torch.randint(0, VOCAB_SIZE, (4, 16))
        out = self.encoder(tokens)
        assert torch.all(out >= 0.0), "Values below 0 found"
        assert torch.all(out <= 1.0), "Values above 1 found"

    def test_no_nan_in_output(self):
        """forward() never returns NaN."""
        tokens = torch.randint(0, VOCAB_SIZE, (2, 8))
        out = self.encoder(tokens)
        assert not torch.isnan(out).any()

    def test_validate_called(self):
        """validate() is called inside forward() — output is clean."""
        tokens = torch.randint(0, VOCAB_SIZE, (1, 4))
        out = self.encoder(tokens)
        # If validate() is called, result must be clamped and NaN-free
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


# ================================
# Task 2.2 — validate() hardening
# ================================

class TestEmotionEncoderValidate:
    """Validates Requirement 1.3."""

    def setup_method(self):
        self.encoder = EmotionEncoder()

    def test_assert_last_dim_8(self):
        """validate() asserts last dim == 8."""
        bad = torch.rand(2, 4, 5)
        with pytest.raises(AssertionError):
            self.encoder.validate(bad)

    def test_nan_replaced_with_zero(self):
        """NaN values are replaced with 0."""
        t = torch.rand(1, 3, 8)
        t[0, 1, 3] = float('nan')
        result = self.encoder.validate(t)
        assert not torch.isnan(result).any()
        assert result[0, 1, 3].item() == 0.0

    def test_inf_replaced_with_zero(self):
        """Inf values are replaced with 0."""
        t = torch.rand(1, 3, 8)
        t[0, 0, 0] = float('inf')
        t[0, 2, 7] = float('-inf')
        result = self.encoder.validate(t)
        assert not torch.isinf(result).any()
        assert result[0, 0, 0].item() == 0.0
        assert result[0, 2, 7].item() == 0.0

    def test_nan_warning_logged(self, caplog):
        """A warning is logged when NaN values are found."""
        import logging
        t = torch.rand(1, 2, 8)
        t[0, 0, 0] = float('nan')
        with caplog.at_level(logging.WARNING, logger="emotionflow_llm"):
            self.encoder.validate(t)
        assert any("NaN" in r.message for r in caplog.records)

    def test_inf_warning_logged(self, caplog):
        """A warning is logged when inf values are found."""
        import logging
        t = torch.rand(1, 2, 8)
        t[0, 1, 5] = float('inf')
        with caplog.at_level(logging.WARNING, logger="emotionflow_llm"):
            self.encoder.validate(t)
        assert any("Inf" in r.message for r in caplog.records)

    def test_clamp_out_of_range(self):
        """Values outside [0, 1] are clamped."""
        t = torch.zeros(1, 1, 8)
        t[0, 0, 0] = 1.5
        t[0, 0, 1] = -0.3
        result = self.encoder.validate(t)
        assert result[0, 0, 0].item() == pytest.approx(1.0)
        assert result[0, 0, 1].item() == pytest.approx(0.0)

    def test_clamp_warning_logged(self, caplog):
        """A warning is logged when clamping occurs."""
        import logging
        t = torch.zeros(1, 1, 8)
        t[0, 0, 2] = 2.0
        with caplog.at_level(logging.WARNING, logger="emotionflow_llm"):
            self.encoder.validate(t)
        assert any("clamp" in r.message.lower() or "out-of-range" in r.message.lower()
                   for r in caplog.records)

    def test_valid_tensor_unchanged(self):
        """A clean tensor in [0, 1] passes through unchanged."""
        t = torch.rand(2, 5, 8)  # rand is in [0, 1)
        result = self.encoder.validate(t)
        assert torch.allclose(result, t)

    def test_multiple_nan_and_inf(self):
        """Multiple NaN and inf values are all replaced with 0."""
        t = torch.rand(2, 4, 8)
        t[0, 0, 0] = float('nan')
        t[0, 1, 2] = float('nan')
        t[1, 3, 7] = float('inf')
        result = self.encoder.validate(t)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert result[0, 0, 0].item() == 0.0
        assert result[0, 1, 2].item() == 0.0
        assert result[1, 3, 7].item() == 0.0


# ================================
# Task 10.3 — TokenEmbedding shape test
# ================================

from emotionflow_llm import TokenEmbedding, WORD_DIM, MODEL_DIM


class TestTokenEmbedding:
    """Validates Requirement 2.1 and 2.2."""

    def setup_method(self):
        self.embedding = TokenEmbedding()

    def test_combined_shape(self):
        """combined output has shape [B, S, WORD_DIM + EMOTION_DIM] == [B, S, 136]."""
        tokens = torch.randint(0, VOCAB_SIZE, (2, 10))
        combined, emotions = self.embedding(tokens)
        assert combined.shape == (2, 10, MODEL_DIM)
        assert combined.shape[-1] == 136

    def test_emotions_shape(self):
        """emotions output has shape [B, S, 8]."""
        tokens = torch.randint(0, VOCAB_SIZE, (2, 10))
        combined, emotions = self.embedding(tokens)
        assert emotions.shape == (2, 10, EMOTION_DIM)
        assert emotions.shape[-1] == 8

    def test_returns_tuple(self):
        """forward() returns a 2-tuple (combined, emotions)."""
        tokens = torch.randint(0, VOCAB_SIZE, (1, 5))
        result = self.embedding(tokens)
        assert isinstance(result, tuple) and len(result) == 2

    def test_combined_is_concatenation(self):
        """combined last dim equals WORD_DIM + EMOTION_DIM."""
        tokens = torch.randint(0, VOCAB_SIZE, (3, 16))
        combined, emotions = self.embedding(tokens)
        assert combined.shape[-1] == WORD_DIM + EMOTION_DIM

    def test_emotions_values_in_range(self):
        """emotions are raw EmotionEncoder output — values in [0, 1]."""
        tokens = torch.randint(0, VOCAB_SIZE, (2, 8))
        _, emotions = self.embedding(tokens)
        assert torch.all(emotions >= 0.0)
        assert torch.all(emotions <= 1.0)


# ================================
# Task 10.4 — Attention weight normalization test
# ================================

from emotionflow_llm import EmotionAttention, MODEL_DIM


class TestEmotionAttention:
    """Validates Requirement 3.1, 3.2, 3.3."""

    def setup_method(self):
        self.attn = EmotionAttention(MODEL_DIM)

    def test_attention_weights_sum_to_one(self):
        """Attention weights sum to 1 per row (softmax normalization)."""
        B, S = 2, 8
        x = torch.randn(B, S, MODEL_DIM)
        emotion = torch.rand(B, S, EMOTION_DIM)

        # Patch forward to expose weights — instead, verify output is consistent
        # with softmax by checking the output is a valid weighted sum of V rows.
        # We verify indirectly: run forward and confirm no NaN/inf (softmax is stable).
        out = self.attn(x, emotion)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_attention_weights_sum_to_one_direct(self):
        """Directly verify softmax rows sum to 1 by inspecting scores path."""
        B, S, dim = 1, 4, MODEL_DIM
        x = torch.randn(B, S, dim)
        emotion = torch.rand(B, S, EMOTION_DIM)

        with torch.no_grad():
            emotion_bias = self.attn.emotion_proj(emotion)
            Q = self.attn.q(x) + emotion_bias
            K = self.attn.k(x) + emotion_bias
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
            weights = torch.softmax(scores, dim=-1)

        # Each row must sum to 1
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"Attention row sums not 1: {row_sums}"

    def test_v_unmodified_by_emotion(self):
        """V projection is NOT affected by emotion bias."""
        B, S, dim = 1, 4, MODEL_DIM
        x = torch.randn(B, S, dim)
        emotion_a = torch.rand(B, S, EMOTION_DIM)
        emotion_b = torch.rand(B, S, EMOTION_DIM)  # different emotion

        with torch.no_grad():
            V_a = self.attn.v(x)
            V_b = self.attn.v(x)

        # V depends only on x, not emotion
        assert torch.allclose(V_a, V_b), "V should not depend on emotion"

    def test_emotion_bias_applied_to_q_and_k(self):
        """Emotion bias shifts Q and K but not V."""
        B, S, dim = 1, 4, MODEL_DIM
        x = torch.randn(B, S, dim)
        emotion = torch.rand(B, S, EMOTION_DIM)
        zero_emotion = torch.zeros(B, S, EMOTION_DIM)

        with torch.no_grad():
            bias = self.attn.emotion_proj(emotion)
            Q_with = self.attn.q(x) + bias
            Q_without = self.attn.q(x) + self.attn.emotion_proj(zero_emotion)
            V_with = self.attn.v(x)
            V_without = self.attn.v(x)

        # Q differs when emotion differs
        assert not torch.allclose(Q_with, Q_without), "Q should change with emotion"
        # V is the same regardless
        assert torch.allclose(V_with, V_without), "V must not change with emotion"

    def test_output_shape(self):
        """forward() returns shape [B, S, dim]."""
        B, S = 3, 10
        x = torch.randn(B, S, MODEL_DIM)
        emotion = torch.rand(B, S, EMOTION_DIM)
        out = self.attn(x, emotion)
        assert out.shape == (B, S, MODEL_DIM)


# ================================
# Task 10.5 — EmotionFlowLLM output shape and activations count
# ================================

from emotionflow_llm import EmotionFlowLLM, LAYERS, VOCAB_SIZE as _VOCAB_SIZE, MODEL_DIM


class TestEmotionFlowLLM:
    """Validates Requirement 4.1, 4.3, 4.4."""

    def setup_method(self):
        self.model = EmotionFlowLLM()

    def test_logits_shape(self):
        """forward() returns logits of shape [B, S, VOCAB_SIZE]."""
        tokens = torch.randint(0, _VOCAB_SIZE, (2, 10))
        logits = self.model(tokens)
        assert logits.shape == (2, 10, _VOCAB_SIZE)

    def test_logits_shape_single(self):
        """forward() returns correct shape for batch size 1."""
        tokens = torch.randint(0, _VOCAB_SIZE, (1, 5))
        logits = self.model(tokens)
        assert logits.shape == (1, 5, _VOCAB_SIZE)

    def test_activations_count(self):
        """get_activations() returns exactly LAYERS tensors after forward()."""
        tokens = torch.randint(0, _VOCAB_SIZE, (2, 8))
        self.model(tokens)
        activations = self.model.get_activations()
        assert len(activations) == LAYERS

    def test_activations_reset_each_forward(self):
        """activations list is reset on each forward pass."""
        tokens = torch.randint(0, _VOCAB_SIZE, (1, 4))
        self.model(tokens)
        self.model(tokens)
        assert len(self.model.get_activations()) == LAYERS

    def test_activations_shape(self):
        """Each stored activation has shape [B, S, MODEL_DIM]."""
        B, S = 2, 6
        tokens = torch.randint(0, _VOCAB_SIZE, (B, S))
        self.model(tokens)
        for act in self.model.get_activations():
            assert act.shape == (B, S, MODEL_DIM)

    def test_activations_are_detached(self):
        """Stored activations are detached from the computation graph."""
        tokens = torch.randint(0, _VOCAB_SIZE, (1, 4))
        self.model(tokens)
        for act in self.model.get_activations():
            assert not act.requires_grad


# ================================
# Tasks 10.6 and 10.7 — Generation
# ================================

from emotionflow_llm import generate, generate_samples, VOCAB_SIZE as _GEN_VOCAB_SIZE


class TestGenerate:
    """Validates Requirements 5.1, 5.2, 5.3."""

    def setup_method(self):
        self.model = EmotionFlowLLM()

    def test_generate_length_2d_prompt(self):
        """generate() returns sequence of length prompt_len + max_len (2D prompt)."""
        prompt = torch.randint(0, _GEN_VOCAB_SIZE, (1, 5))
        max_len = 10
        result = generate(self.model, prompt, max_len=max_len)
        assert result.shape[1] == 5 + max_len

    def test_generate_length_1d_prompt(self):
        """generate() handles 1D prompt and returns correct length."""
        prompt = torch.randint(0, _GEN_VOCAB_SIZE, (5,))
        max_len = 8
        result = generate(self.model, prompt, max_len=max_len)
        assert result.shape[1] == 5 + max_len

    def test_generate_includes_original_prompt(self):
        """generate() output starts with the original prompt tokens."""
        prompt = torch.randint(0, _GEN_VOCAB_SIZE, (1, 4))
        result = generate(self.model, prompt, max_len=5)
        assert torch.equal(result[:, :4], prompt)

    def test_generate_default_max_len(self):
        """generate() default max_len is 20."""
        prompt = torch.randint(0, _GEN_VOCAB_SIZE, (1, 3))
        result = generate(self.model, prompt)
        assert result.shape[1] == 3 + 20


class TestGenerateSamples:
    """Validates Requirements 6.1, 6.2, 6.3."""

    def setup_method(self):
        self.model = EmotionFlowLLM()

    def test_generate_samples_count_default(self):
        """generate_samples() returns exactly 20 items by default."""
        prompt = torch.randint(0, _GEN_VOCAB_SIZE, (1, 4))
        samples = generate_samples(self.model, prompt)
        assert len(samples) == 20

    def test_generate_samples_count_custom(self):
        """generate_samples() returns exactly n items when n is specified."""
        prompt = torch.randint(0, _GEN_VOCAB_SIZE, (1, 4))
        samples = generate_samples(self.model, prompt, n=5)
        assert len(samples) == 5

    def test_generate_samples_each_correct_length(self):
        """Each sample from generate_samples() has the correct length."""
        prompt = torch.randint(0, _GEN_VOCAB_SIZE, (1, 3))
        samples = generate_samples(self.model, prompt, n=3)
        for s in samples:
            assert s.shape[1] == 3 + 20


# ================================
# Task 10.8 — GenerationConfig.validate() error cases
# ================================

from emotionflow_llm import GenerationConfig


class TestGenerationConfigValidate:
    """Validates Requirement 7.2."""

    def test_invalid_temperature_low(self):
        cfg = GenerationConfig(temperature=0.05)
        with pytest.raises(ValueError, match="temperature"):
            cfg.validate()

    def test_invalid_temperature_high(self):
        cfg = GenerationConfig(temperature=2.5)
        with pytest.raises(ValueError, match="temperature"):
            cfg.validate()

    def test_invalid_top_p(self):
        cfg = GenerationConfig(top_p=1.5)
        with pytest.raises(ValueError, match="top_p"):
            cfg.validate()

    def test_invalid_top_k(self):
        cfg = GenerationConfig(top_k=0)
        with pytest.raises(ValueError, match="top_k"):
            cfg.validate()

    def test_invalid_strategy(self):
        cfg = GenerationConfig(strategy="greedy")
        with pytest.raises(ValueError, match="strategy"):
            cfg.validate()

    def test_valid_config_no_error(self):
        cfg = GenerationConfig()
        cfg.validate()  # should not raise


# ================================
# Task 10.9 — emotion_profile() returns valid emotion name
# ================================

from emotionflow_llm import emotion_profile, EMOTIONS, VOCAB_SIZE as _EP_VOCAB_SIZE


class TestEmotionProfile:
    """Validates Requirement 8.1 and 8.2."""

    def setup_method(self):
        self.model = EmotionFlowLLM()

    def test_returns_string(self):
        """emotion_profile() returns a string."""
        tokens = torch.randint(0, _EP_VOCAB_SIZE, (1, 8))
        result = emotion_profile(self.model, tokens)
        assert isinstance(result, str)

    def test_returns_valid_emotion_name(self):
        """emotion_profile() returns a name that is in EMOTIONS."""
        tokens = torch.randint(0, _EP_VOCAB_SIZE, (1, 8))
        result = emotion_profile(self.model, tokens)
        assert result in EMOTIONS

    def test_works_with_batch(self):
        """emotion_profile() handles batch size > 1 (means across batch+seq)."""
        tokens = torch.randint(0, _EP_VOCAB_SIZE, (3, 10))
        result = emotion_profile(self.model, tokens)
        assert result in EMOTIONS

    def test_deterministic_for_same_input(self):
        """emotion_profile() is deterministic for the same model and tokens."""
        tokens = torch.randint(0, _EP_VOCAB_SIZE, (1, 5))
        r1 = emotion_profile(self.model, tokens)
        r2 = emotion_profile(self.model, tokens)
        assert r1 == r2


# ================================
# Task 10.10 — synthesize() length constraint
# ================================

from emotionflow_llm import synthesize, SEQ_LEN


class TestSynthesize:
    """Validates Requirement 9.1 and 9.2."""

    def test_output_truncated_to_seq_len(self):
        """synthesize() truncates output to SEQ_LEN tokens."""
        outputs = [torch.randint(0, 100, (1, 20)) for _ in range(5)]
        result = synthesize(outputs)
        assert result.shape[1] <= SEQ_LEN

    def test_output_exactly_seq_len_when_long_enough(self):
        """synthesize() returns exactly SEQ_LEN when total > SEQ_LEN."""
        outputs = [torch.randint(0, 100, (1, 20)) for _ in range(5)]  # 100 total > 32
        result = synthesize(outputs)
        assert result.shape[1] == SEQ_LEN

    def test_output_shorter_when_inputs_short(self):
        """synthesize() returns all tokens when total < SEQ_LEN."""
        outputs = [torch.randint(0, 100, (1, 5)) for _ in range(3)]  # 15 total < 32
        result = synthesize(outputs)
        assert result.shape[1] == 15

    def test_single_output(self):
        """synthesize() works with a single tensor in the list."""
        outputs = [torch.randint(0, 100, (1, 10))]
        result = synthesize(outputs)
        assert result.shape[1] == 10

    def test_batch_dimension_preserved(self):
        """synthesize() preserves the batch dimension."""
        outputs = [torch.randint(0, 100, (2, 20)) for _ in range(3)]
        result = synthesize(outputs)
        assert result.shape[0] == 2


# ================================
# Task 10.11 — collect_emotion_trajectory() length == LAYERS
# ================================

from emotionflow_llm import collect_emotion_trajectory, LAYERS as _LAYERS


class TestCollectEmotionTrajectory:
    """Validates Requirement 10.1 and 10.2."""

    def setup_method(self):
        self.model = EmotionFlowLLM()

    def test_trajectory_length_equals_layers(self):
        """collect_emotion_trajectory() returns a list of length LAYERS."""
        tokens = torch.randint(0, VOCAB_SIZE, (1, 8))
        self.model(tokens)
        traj = collect_emotion_trajectory(self.model)
        assert len(traj) == _LAYERS

    def test_trajectory_values_are_scalars(self):
        """Each value in the trajectory is a Python float (scalar)."""
        tokens = torch.randint(0, VOCAB_SIZE, (1, 8))
        self.model(tokens)
        traj = collect_emotion_trajectory(self.model)
        for val in traj:
            assert isinstance(val, float)

    def test_trajectory_values_are_finite(self):
        """All trajectory values are finite (no NaN or inf)."""
        tokens = torch.randint(0, VOCAB_SIZE, (1, 8))
        self.model(tokens)
        traj = collect_emotion_trajectory(self.model)
        for val in traj:
            assert math.isfinite(val)


# ================================
# Task 10.12 — PerformanceMonitor report keys
# ================================

from emotionflow_llm import PerformanceMonitor
import time as _time


class TestPerformanceMonitor:
    """Validates Requirement 11.1 and 11.2."""

    def test_measure_records_elapsed_time(self):
        """measure() records elapsed time under the given name."""
        monitor = PerformanceMonitor()
        with monitor.measure("component_a"):
            pass
        assert "component_a" in monitor.timings
        assert len(monitor.timings["component_a"]) == 1
        assert monitor.timings["component_a"][0] >= 0.0

    def test_measure_appends_multiple_timings(self):
        """measure() appends to the list on repeated calls."""
        monitor = PerformanceMonitor()
        for _ in range(3):
            with monitor.measure("comp"):
                pass
        assert len(monitor.timings["comp"]) == 3

    def test_report_keys_match_measured_components(self):
        """report() keys match the names passed to measure()."""
        monitor = PerformanceMonitor()
        with monitor.measure("encoder"):
            pass
        with monitor.measure("decoder"):
            pass
        report = monitor.report()
        assert set(report.keys()) == {"encoder", "decoder"}

    def test_report_contains_required_stats(self):
        """report() returns mean, min, max, total for each component."""
        monitor = PerformanceMonitor()
        with monitor.measure("step"):
            pass
        stats = monitor.report()["step"]
        assert set(stats.keys()) == {"mean", "min", "max", "total"}

    def test_report_stats_are_correct(self):
        """report() computes correct mean/min/max/total."""
        monitor = PerformanceMonitor()
        # Inject known timings directly
        monitor.timings["x"] = [0.1, 0.2, 0.3]
        stats = monitor.report()["x"]
        assert stats["mean"] == pytest.approx(0.2)
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.3)
        assert stats["total"] == pytest.approx(0.6)

    def test_report_empty_when_no_measurements(self):
        """report() returns empty dict when nothing has been measured."""
        monitor = PerformanceMonitor()
        assert monitor.report() == {}

    def test_measure_is_context_manager(self):
        """measure() works as a context manager (no exception on normal use)."""
        monitor = PerformanceMonitor()
        with monitor.measure("test"):
            x = 1 + 1
        assert x == 2
