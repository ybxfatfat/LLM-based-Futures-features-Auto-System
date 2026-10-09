"""Read and update the miner's JSON-backed research libraries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_LIBRARY_DIR = Path(__file__).resolve().parent
_FOUNDATION_FEATURES_FILE = _LIBRARY_DIR / "base_features.json"
_COMPOUND_OPERATORS_FILE = _LIBRARY_DIR / "compound_operators.json"
_EXTRA_BASE_FIELDS_FILE = _LIBRARY_DIR / "extra_base_fields.json"


def _load_mapping(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Library file must contain a JSON object: {path}")
    return data


def _with_record_names(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Make old records compatible with callers that display ``record['name']``."""
    normalized: dict[str, dict[str, Any]] = {}
    for name, record in data.items():
        if not isinstance(record, dict):
            raise ValueError(f"Library entry {name!r} must be a JSON object")
        normalized[name] = {**record, "name": name}
    return normalized


def _write_mapping(path: Path, data: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def load_foundation_features() -> dict[str, dict[str, Any]]:
    return _with_record_names(_load_mapping(_FOUNDATION_FEATURES_FILE))


def load_compound_operators() -> dict[str, dict[str, Any]]:
    return _with_record_names(_load_mapping(_COMPOUND_OPERATORS_FILE))


def load_extra_base_fields() -> dict[str, dict[str, Any]]:
    return _with_record_names(_load_mapping(_EXTRA_BASE_FIELDS_FILE))


def register_foundation_feature(
    name: str,
    formula: str,
    description: str,
    field_type: str,
) -> dict[str, Any]:
    if not name or not formula:
        raise ValueError("Foundation feature name and formula must not be empty")

    data = _load_mapping(_FOUNDATION_FEATURES_FILE)
    record = {
        "name": name,
        "formula": formula,
        "description": description,
        "field_type": field_type,
    }
    data[name] = record
    _write_mapping(_FOUNDATION_FEATURES_FILE, data)
    return record


def register_compound_operator(
    name: str,
    parameters: list[str],
    formula_template: str,
    description: str,
) -> dict[str, Any]:
    if not name or not formula_template:
        raise ValueError("Compound operator name and formula template must not be empty")
    if not isinstance(parameters, list) or not all(isinstance(item, str) for item in parameters):
        raise ValueError("Compound operator parameters must be a list of strings")

    data = _load_mapping(_COMPOUND_OPERATORS_FILE)
    record = {
        "name": name,
        "parameters": parameters,
        "formula_template": formula_template,
        "description": description,
    }
    data[name] = record
    _write_mapping(_COMPOUND_OPERATORS_FILE, data)
    return record


__all__ = [
    "load_compound_operators",
    "load_extra_base_fields",
    "load_foundation_features",
    "register_compound_operator",
    "register_foundation_feature",
]
