from . import v1, v2

_VERSIONS = {"v1": v1, "v2": v2}
DEFAULT_VERSION = "v2"          # bump this in a reviewed PR to promote a new version


def active():
    """Return the active version module for this prompt family, honoring
    TEXT2SQL_PROMPT_VERSION_OVERRIDES before falling back to DEFAULT_VERSION."""
    from ..registry import resolve_version
    version = resolve_version("intent", DEFAULT_VERSION, _VERSIONS)
    return _VERSIONS[version]
