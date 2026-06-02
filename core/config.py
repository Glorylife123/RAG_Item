from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else project_root() / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _resolve_paths(data, config_path.parent)


def _resolve_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    app = config.setdefault("app", {})
    tools = config.setdefault("tools", {})

    for key in ("chroma_dir", "bm25_path"):
        if key in app:
            app[key] = str((base_dir / app[key]).resolve()) if str(app[key]).startswith(".") else app[key]

    drug = tools.setdefault("drug_interaction", {})
    if "local_csv" in drug:
        drug["local_csv"] = str((base_dir / drug["local_csv"]).resolve()) if str(drug["local_csv"]).startswith(".") else drug["local_csv"]

    return config
