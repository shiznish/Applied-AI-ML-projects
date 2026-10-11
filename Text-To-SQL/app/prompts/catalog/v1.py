"""Shared curated-catalog rendering (app/semantic/layer.yaml), used by both
the single-turn semantic router (app/prompts/router.py) and the multi-turn
merge-intent prompt (app/prompts/merge.py) — moved from
app/semantic/router.py so the two call sites can't drift out of sync by each
maintaining their own copy of how the catalog is listed out as plain text.
"""
from ...semantic.resolver import SemanticLayer

PROMPT_VERSION = "catalog-2026.10.1"


def catalog_prompt(layer: SemanticLayer) -> str:
    lines = []

    lines.append("Domains (context only — narrows what's shown; never pick tables "
                  "by domain membership):")
    for name, d in layer.domains.items():
        lines.append(
            f"- {name}: {d.description} "
            f"(metrics: {', '.join(d.metrics) or 'none'}; "
            f"dimensions: {', '.join(d.dimensions) or 'none'})"
        )

    lines.append("\nMetrics — canonical id: description (synonyms, if any):")
    for name, m in layer.metrics.items():
        syn = f" — synonyms: {', '.join(m.synonyms)}" if m.synonyms else ""
        lines.append(f"- {name}: {m.description}{syn}")

    lines.append("\nDimensions — canonical id: what it slices by (synonyms, if any):")
    for name, d in layer.dimensions.items():
        syn = f" — synonyms: {', '.join(d.synonyms)}" if d.synonyms else ""
        lines.append(f"- {name}{syn}")

    return "\n".join(lines)
