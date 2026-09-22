"""Config loading: default.yaml overlaid by a profile. See SPEC.md Section 9.

Not named in the SPEC.md Section 6 repo tree; placed under core/ since it is
transport-independent and needed by every stage. Recorded as a spec
interpretation in PROGRESS.md.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


class OptimizationFlags(BaseModel):
    """The naive/tuned flag table from SPEC.md Section 9."""

    turn_detection: Literal["silence", "semantic"]
    speculative_retrieval: bool
    prefix_cache_prompt_order: bool
    history_compaction: bool
    tts_chunking: Literal["full", "sentence", "clause"]
    semantic_cache: bool
    filler_on_tool_call: bool
    asr_router_parakeet: bool
    retrieval_bridge: Literal["mt", "multilingual", "hybrid"]


class Settings(BaseSettings):
    languages: list[str]
    flags: OptimizationFlags

    model_config = SettingsConfigDict(env_prefix="POLYGLOT_", env_nested_delimiter="__")


def load_settings(profile: Literal["naive", "tuned"], config_dir: Path = CONFIG_DIR) -> Settings:
    """Load config/default.yaml overlaid by config/profiles/<profile>.yaml.

    Environment variables prefixed POLYGLOT_ (with __ as the nested
    delimiter) override the merged YAML, per pydantic-settings precedence.
    """
    default_path = config_dir / "default.yaml"
    profile_path = config_dir / "profiles" / f"{profile}.yaml"

    with default_path.open("r", encoding="utf-8") as f:
        merged = yaml.safe_load(f) or {}
    with profile_path.open("r", encoding="utf-8") as f:
        overlay = yaml.safe_load(f) or {}

    merged = _deep_merge(merged, overlay)
    return Settings(**merged)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
