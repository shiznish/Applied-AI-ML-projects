from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # `.env.local` is read second so it overrides `.env` — the standard
    # convention for a personal, gitignored local override layered on top of
    # a committed default/example file.
    model_config = SettingsConfigDict(
        env_prefix="LEDGER_COPILOT_", env_file=(".env", ".env.local")
    )

    environment: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    vendor_risk_api_base_url: str = "https://api.opensanctions.org"
    vendor_risk_api_key: str | None = None
    vendor_risk_timeout_seconds: float = 5.0
    vendor_risk_max_retries: int = 3
    vendor_risk_circuit_failure_threshold: int = 5
    vendor_risk_circuit_cooldown_seconds: float = 30.0

    auth_jwks_uri: str | None = None
    auth_issuer: str | None = None
    auth_audience: str | None = None
    auth_algorithm: str = "RS256"

    audit_database_path: str = ":memory:"

    rate_limit_capacity: float = 20.0
    rate_limit_refill_per_second: float = 10.0
