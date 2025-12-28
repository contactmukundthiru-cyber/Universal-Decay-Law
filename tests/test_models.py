"""Tests for decay models."""

import numpy as np
import pytest

from src.models.base import DecayModelRegistry
from src.models.stretched_exponential import StretchedExponentialModel
from src.models.power_law import PowerLawModel


class TestStretchedExponential:
    """Tests for stretched exponential model."""

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        model = StretchedExponentialModel()
        t = np.array([0, 1, 5, 10])
        E = model.evaluate(t, tau=10.0, gamma=0.5, E0=1.0)

        assert E[0] == pytest.approx(1.0)  # E(0) = E0
        assert E[-1] < E[0]  # Decay
        assert all(E >= 0)  # Non-negative

    def test_evaluate_gamma_effect(self):
        """Test that lower gamma gives slower decay."""
        model = StretchedExponentialModel()
        t = np.linspace(0, 50, 100)

        E_low_gamma = model.evaluate(t, tau=10.0, gamma=0.3)
        E_high_gamma = model.evaluate(t, tau=10.0, gamma=0.9)

        # At intermediate times, lower gamma decays slower
        mid_idx = len(t) // 2
        assert E_low_gamma[mid_idx] > E_high_gamma[mid_idx]

    def test_gradient_computation(self):
        """Test gradient computation."""
        model = StretchedExponentialModel()
        t = np.array([1.0, 5.0, 10.0])
        grads = model.gradient(t, tau=10.0, gamma=0.5, E0=1.0)

        assert "tau" in grads
        assert "gamma" in grads
        assert "E0" in grads
        assert all(np.isfinite(grads["tau"]))

    def test_fit_synthetic(self):
        """Test fitting to synthetic data."""
        model = StretchedExponentialModel()

        # Generate data with known parameters
        np.random.seed(42)
        true_tau, true_gamma = 15.0, 0.6
        t = np.linspace(0, 100, 50)
        E_true = model.evaluate(t, tau=true_tau, gamma=true_gamma)
        E_noisy = E_true + np.random.normal(0, 0.02, len(E_true))
        E_noisy = np.clip(E_noisy, 0.01, 1.0)

        result = model.fit(t, E_noisy)

        assert result.converged
        assert result.parameters["tau"] == pytest.approx(true_tau, rel=0.3)
        assert result.parameters["gamma"] == pytest.approx(true_gamma, rel=0.3)

    def test_universal_form(self):
        """Test universal form function."""
        x = np.array([0, 1, 2, 3])
        f = StretchedExponentialModel.universal_form(x, gamma=0.5)

        assert f[0] == pytest.approx(1.0)
        assert all(f[1:] < 1.0)


class TestPowerLaw:
    """Tests for power law model."""

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        model = PowerLawModel()
        t = np.array([0, 1, 10, 100])
        E = model.evaluate(t, tau=10.0, gamma=1.5, E0=1.0)

        assert E[0] == pytest.approx(1.0)
        assert E[-1] < E[0]

    def test_long_tail(self):
        """Test that power law has longer tail than exponential."""
        power_model = PowerLawModel()
        exp_model = StretchedExponentialModel()

        t = np.linspace(0, 200, 100)
        E_power = power_model.evaluate(t, tau=10.0, gamma=1.5)
        E_exp = exp_model.evaluate(t, tau=10.0, gamma=1.0)

        # At very long times, power law decays slower
        assert E_power[-1] > E_exp[-1]


class TestModelRegistry:
    """Tests for model registry."""

    def test_registry_contains_models(self):
        """Test that registry contains all models."""
        models = DecayModelRegistry.list_models()
        assert "stretched_exponential" in models
        assert "power_law" in models

    def test_get_model(self):
        """Test retrieving model from registry."""
        model = DecayModelRegistry.get("stretched_exponential")
        assert isinstance(model, StretchedExponentialModel)

    def test_get_invalid_model(self):
        """Test error on invalid model name."""
        with pytest.raises(ValueError):
            DecayModelRegistry.get("nonexistent_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
