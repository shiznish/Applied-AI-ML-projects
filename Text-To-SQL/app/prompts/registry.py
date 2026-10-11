"""Shared version-resolution for app/prompts/<family>/ packages — see each
family's __init__.py for how `active()` uses this."""
from ..config import settings


def resolve_version(family: str, default_version: str, available: dict) -> str:
    override = settings.prompt_version_overrides.get(family)
    if override is None:
        return default_version
    if override not in available:
        raise ValueError(
            f"TEXT2SQL_PROMPT_VERSION_OVERRIDES requests unknown version "
            f"'{override}' for prompt family '{family}' (available: "
            f"{sorted(available)})"
        )
    return override
