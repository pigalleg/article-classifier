from pathlib import Path

from scripts.run_affinity_benchmark import ModelProfile, _build_env


def test_build_env_exports_model_specific_api_key_env_and_generic_key():
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