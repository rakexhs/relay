from relay.config import Settings


def test_defaults_run_offline(monkeypatch):
    for key in ("RELAY_PROVIDER", "RELAY_DATABASE_URL", "RELAY_APPROVAL_TOOLS"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.provider == "stub"
    assert not settings.uses_postgres
    assert settings.approval_tools == ["draft_reply", "escalate_to_human"]


def test_approval_tools_accepts_csv(monkeypatch):
    monkeypatch.setenv("RELAY_APPROVAL_TOOLS", "escalate_to_human, draft_reply")
    settings = Settings(_env_file=None)
    assert settings.approval_tools == ["escalate_to_human", "draft_reply"]


def test_default_model_per_provider():
    assert Settings(_env_file=None, provider="anthropic").default_model == "claude-opus-5-5"
    assert Settings(_env_file=None, provider="openai", model="gpt-x").default_model == "gpt-x"
