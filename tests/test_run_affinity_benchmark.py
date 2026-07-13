from pathlib import Path

from scripts.run_affinity_benchmark import ModelProfile, _build_env
from scripts.run_affinity_benchmark import _resolve_profiles


def test_build_env_exports_model_specific_api_key_env_and_generic_key():
    """Test that _build_env sets both model-specific and generic API key variables."""
    profile = ModelProfile(
        name="gpt-5.4-mini",
        backend="cloud",
        model="gpt-5.4-mini",
        base_url=None,
        api_key="secret-value",
        api_key_env="OPENAI_CLOUD_API_KEY",
        requests_per_minute=500,
        enabled=True,
    )

    env = _build_env(profile, Path("/tmp/model-output"))

    assert env["OPENAI_CLOUD_API_KEY"] == "secret-value"
    assert env["OPENAI_API_KEY"] == "secret-value"
    assert env["OPENAI_MODEL"] == "gpt-5.4-mini"
    assert env["LLM_BACKEND"] == "cloud"
    # Verify the hint for subprocess about which API key env var to use
    assert env["OPENAI_API_KEY_ENV_VAR"] == "OPENAI_CLOUD_API_KEY"


def test_build_env_swaps_between_local_and_multiple_cloud_models():
    """Test API key swapping across local model → cloud OpenAI → cloud Google."""

    # ===== Model 1: Local Model (Ollama) =====
    local_profile = ModelProfile(
        name="gemma4:31b-cloud",
        backend="local",
        model="gemma4:31b-cloud",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        api_key_env=None,  # Local models don't use env vars
        requests_per_minute=9999,
        enabled=True,
    )

    env1 = _build_env(local_profile, Path("/tmp/model1-output"))

    # Local model sets generic OPENAI_API_KEY
    assert env1["OPENAI_API_KEY"] == "ollama"
    assert env1["OPENAI_MODEL"] == "gemma4:31b-cloud"
    assert env1["OPENAI_BASE_URL"] == "http://localhost:11434/v1"
    assert env1["LLM_BACKEND"] == "local"
    # No model-specific env var (api_key_env is None)
    assert "OPENAI_CLOUD_API_KEY" not in env1 or env1.get("OPENAI_CLOUD_API_KEY") != "ollama"
    assert "GOOGLE_API_KEY" not in env1 or env1.get("GOOGLE_API_KEY") != "ollama"
    # No API key env var hint for local models
    assert "OPENAI_API_KEY_ENV_VAR" not in env1 or env1.get("OPENAI_API_KEY_ENV_VAR") is None

    # ===== Model 2: Cloud OpenAI Model =====
    openai_profile = ModelProfile(
        name="gpt-5.4-mini",
        backend="cloud",
        model="gpt-5.4-mini",
        base_url=None,
        api_key="sk-proj-openai-key-123",
        api_key_env="OPENAI_CLOUD_API_KEY",
        requests_per_minute=500,
        enabled=True,
    )

    env2 = _build_env(openai_profile, Path("/tmp/model2-output"))

    # OpenAI model sets model-specific and generic API keys
    assert env2["OPENAI_CLOUD_API_KEY"] == "sk-proj-openai-key-123"
    assert env2["OPENAI_API_KEY"] == "sk-proj-openai-key-123"
    assert env2["OPENAI_MODEL"] == "gpt-5.4-mini"
    assert env2["LLM_BACKEND"] == "cloud"
    # Tell subprocess to use OPENAI_CLOUD_API_KEY when loading config
    assert env2["OPENAI_API_KEY_ENV_VAR"] == "OPENAI_CLOUD_API_KEY"
    # Base URL should not be set (None in profile)
    assert "OPENAI_BASE_URL" not in env2 or env2.get("OPENAI_BASE_URL") is None
    # Previous local model key should be overwritten
    assert env2.get("OPENAI_API_KEY") != "ollama"
    # Google API key should not be set by this model
    assert "GOOGLE_API_KEY" not in env2 or env2.get("GOOGLE_API_KEY") != "sk-proj-openai-key-123"

    # ===== Model 3: Cloud Google Model (Gemini) =====
    google_profile = ModelProfile(
        name="gemini-3-flash-preview",
        backend="cloud",
        model="gemini-3-flash-preview",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key="gsk-google-key-789",
        api_key_env="GOOGLE_API_KEY",
        requests_per_minute=500,
        enabled=True,
    )

    env3 = _build_env(google_profile, Path("/tmp/model3-output"))

    # Google model sets its specific API key
    assert env3["GOOGLE_API_KEY"] == "gsk-google-key-789"
    # Also sets generic API key for fallback compatibility
    assert env3["OPENAI_API_KEY"] == "gsk-google-key-789"
    assert env3["OPENAI_MODEL"] == "gemini-3-flash-preview"
    assert env3["OPENAI_BASE_URL"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert env3["LLM_BACKEND"] == "cloud"
    # Tell subprocess to use GOOGLE_API_KEY when loading config
    assert env3["OPENAI_API_KEY_ENV_VAR"] == "GOOGLE_API_KEY"
    # Previous OpenAI key should be overwritten
    assert env3.get("OPENAI_API_KEY") != "sk-proj-openai-key-123"
    # OPENAI_CLOUD_API_KEY should not be set (or not match this model's key)
    assert env3.get("OPENAI_CLOUD_API_KEY") != "gsk-google-key-789"

    # ===== Verify Clean Isolation =====
    # Each environment should be isolated with correct variables
    assert env1["LLM_BACKEND"] == "local"
    assert env2["LLM_BACKEND"] == "cloud"
    assert env3["LLM_BACKEND"] == "cloud"

    # Each should have different model names
    assert env1["OPENAI_MODEL"] == "gemma4:31b-cloud"
    assert env2["OPENAI_MODEL"] == "gpt-5.4-mini"
    assert env3["OPENAI_MODEL"] == "gemini-3-flash-preview"

    # API keys should be properly isolated
    assert env1["OPENAI_API_KEY"] == "ollama"
    assert env2["OPENAI_API_KEY"] == "sk-proj-openai-key-123"
    assert env3["OPENAI_API_KEY"] == "gsk-google-key-789"


def test_resolve_profiles_preserves_per_model_metadata_when_overridden(monkeypatch):
    """Test that launcher overrides keep the Google model's base URL and API key env."""

    settings = {
        "models": {
            "llm": {
                "default_backend": "local",
                "benchmark": {
                    "models": [
                        {
                            "name": "gpt-5.4-mini",
                            "backend": "cloud",
                            "model": "gpt-5.4-mini",
                            "api_key_env": "OPENAI_CLOUD_API_KEY",
                            "enabled": True,
                        },
                        {
                            "name": "gemini-3.1-flash-lite",
                            "backend": "cloud",
                            "model": "google/gemini-3.1-flash-lite",
                            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                            "api_key_env": "GOOGLE_API_KEY",
                            "enabled": True,
                        },
                    ]
                },
            }
        }
    }

    monkeypatch.setenv("AFFINITY_BENCHMARK_MODELS", "gemini-3.1-flash-lite")
    monkeypatch.setenv("AFFINITY_BENCHMARK_BACKEND", "cloud")

    profiles, forced_backend = _resolve_profiles(settings)

    assert forced_backend == "cloud"
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.name == "gemini-3.1-flash-lite"
    assert profile.model == "google/gemini-3.1-flash-lite"
    assert profile.backend == "cloud"
    assert profile.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert profile.api_key_env == "GOOGLE_API_KEY"