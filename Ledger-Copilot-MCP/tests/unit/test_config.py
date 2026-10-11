from ledger_copilot.config import Settings


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.port == 8000


def test_settings_reads_env_var(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LEDGER_COPILOT_PORT", "9000")

    settings = Settings(_env_file=None)

    assert settings.port == 9000
