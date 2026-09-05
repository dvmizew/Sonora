from __future__ import annotations

import functools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sonora.core.constants import (
    ALBUM_MATCH_THRESHOLD,
    ARTIST_MATCH_THRESHOLD,
    DIRS,
    GENIUS_MATCH_THRESHOLD,
)
from sonora.core.logger import LOG

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass(frozen=True)
class SonoraConfig:
    """Immutable, typed Sonora application configuration with XDG and environment support."""

    artist_match_threshold: float = ARTIST_MATCH_THRESHOLD
    album_match_threshold: float = ALBUM_MATCH_THRESHOLD
    genius_match_threshold: float = GENIUS_MATCH_THRESHOLD

    disc_folder_patterns: tuple[str, ...] = (r"^(?:cd|disc|side)\s*\d+$",)

    generic_containers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "flac",
                "mp3",
                "music",
                "lossless",
                "audio",
                "downloads",
                "library",
                "singles",
                "tracks",
                "songs",
                "unknown",
                "unknown artist",
                "unknown album",
            }
        )
    )

    codec_rip_keywords: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "flac",
                "mp3",
                "320",
                "320kbps",
                "lossless",
                "rip",
                "cdrip",
                "webrip",
                "hq",
                "hd",
            }
        )
    )

    featuring_conjunctions: tuple[str, ...] = (
        "feat",
        "ft",
        "featuring",
        "with",
        "and",
        "vs",
        "cu",
        "și",
        "si",
        "con",
        "avec",
        "mit",
    )

    def is_disc_folder(self, folder_name: str) -> bool:
        clean = folder_name.strip()
        for pat in self.disc_folder_patterns:
            if re.match(pat, clean, re.IGNORECASE):
                return True
        return False

    def is_generic_container(self, folder_name: str) -> bool:
        return folder_name.strip().lower() in self.generic_containers


def _parse_config_data(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        suffix = path.suffix.lower()
        if suffix == ".toml":
            with path.open("rb") as config_file:
                data = tomllib.load(config_file)
            if isinstance(data, dict):
                sub = data.get("sonora")
                if isinstance(sub, dict):
                    return {str(k): v for k, v in sub.items()}
                return {str(k): v for k, v in data.items()}
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                sub = data.get("sonora")
                if isinstance(sub, dict):
                    return {str(k): v for k, v in sub.items()}
                return {str(k): v for k, v in data.items()}
    except (OSError, ValueError) as error:
        LOG.warning(f"Failed to parse configuration file {path}: {error}")
    return {}


def _discover_config_file() -> Path | None:
    candidate_paths = [
        DIRS.user_config_path / "config.toml",
        DIRS.user_config_path / "config.json",
        Path.home() / ".config" / "sonora" / "config.toml",
        Path.home() / ".config" / "sonora" / "config.json",
        Path("sonora.toml"),
        Path("sonora.json"),
    ]
    seen_paths: set[Path] = set()
    for path in candidate_paths:
        try:
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


@functools.cache
def get_config() -> SonoraConfig:
    config_file = _discover_config_file()
    if config_file is None:
        return SonoraConfig()

    raw_data = _parse_config_data(config_file)
    if not raw_data:
        return SonoraConfig()

    kwargs: dict[str, Any] = {}

    if "artist_match_threshold" in raw_data:
        try:
            kwargs["artist_match_threshold"] = float(raw_data["artist_match_threshold"])
        except (ValueError, TypeError):
            pass

    if "album_match_threshold" in raw_data:
        try:
            kwargs["album_match_threshold"] = float(raw_data["album_match_threshold"])
        except (ValueError, TypeError):
            pass

    if "genius_match_threshold" in raw_data:
        try:
            kwargs["genius_match_threshold"] = float(raw_data["genius_match_threshold"])
        except (ValueError, TypeError):
            pass

    if "disc_folder_patterns" in raw_data:
        patterns = raw_data["disc_folder_patterns"]
        if isinstance(patterns, list):
            kwargs["disc_folder_patterns"] = tuple(str(p) for p in patterns if p)

    if "generic_containers" in raw_data:
        containers = raw_data["generic_containers"]
        if isinstance(containers, list):
            kwargs["generic_containers"] = frozenset(
                str(c).lower() for c in containers if c
            )

    if "codec_rip_keywords" in raw_data:
        keywords = raw_data["codec_rip_keywords"]
        if isinstance(keywords, list):
            kwargs["codec_rip_keywords"] = frozenset(
                str(k).lower() for k in keywords if k
            )

    if "featuring_conjunctions" in raw_data:
        conjunctions = raw_data["featuring_conjunctions"]
        if isinstance(conjunctions, list):
            kwargs["featuring_conjunctions"] = tuple(
                str(c).lower() for c in conjunctions if c
            )

    return SonoraConfig(**kwargs)


def clear_config_cache() -> None:
    get_config.cache_clear()
    get_artist_split_pattern.cache_clear()
    get_disambiguation_pattern.cache_clear()
    get_duplicate_feat_pattern.cache_clear()
    get_bracket_feat_pattern.cache_clear()
    get_balanced_feat_pattern.cache_clear()
    get_feat_tokens_pattern.cache_clear()


@functools.cache
def get_artist_split_pattern() -> re.Pattern[str]:
    cfg = get_config()
    parts: list[str] = []
    for c in cfg.featuring_conjunctions:
        escaped = re.escape(c)
        if c in ("feat", "ft", "vs"):
            parts.append(rf"\s+{escaped}\.?\s+")
        else:
            parts.append(rf"\s+{escaped}\s+")
    parts.extend(
        [
            r"\s+[xX\u00d7]\s+",
            r"\s*&\s*",
            r"\s*,\s*",
            r"\s*;\s*",
            r"\s*/\s*",
        ]
    )
    return re.compile("|".join(parts), re.IGNORECASE)


@functools.cache
def get_disambiguation_pattern() -> re.Pattern[str]:
    cfg = get_config()
    words = "|".join(
        rf"{re.escape(c)}\.?" if c in ("feat", "ft", "vs") else re.escape(c)
        for c in cfg.featuring_conjunctions
    )
    return re.compile(
        rf"\s*\([^()]{{1,40}}\)(?=\s*(?:[,;/&+-\\]|\b(?:{words})\b|$))",
        re.IGNORECASE,
    )


@functools.cache
def get_duplicate_feat_pattern() -> re.Pattern[str]:
    cfg = get_config()
    words = "|".join(
        rf"{re.escape(c)}\.?" if c in ("feat", "ft") else re.escape(c)
        for c in cfg.featuring_conjunctions
        if c not in ("with", "and", "vs")
    )
    return re.compile(
        rf"\s+(?:{words})\s+([A-Za-z0-9\s\.\'\-]+?)(?=\s*[\(\[\{{]\s*(?:{words})\s+\1[\)\]\}}])",
        re.IGNORECASE,
    )


@functools.cache
def get_bracket_feat_pattern() -> re.Pattern[str]:
    cfg = get_config()
    words = "|".join(
        rf"{re.escape(c)}\.?" if c in ("feat", "ft") else re.escape(c)
        for c in cfg.featuring_conjunctions
        if c not in ("with", "and", "vs")
    )
    return re.compile(rf"\s*[\(\[\{{]\s*(?:{words})\s+.*?[\)\]\}}]", re.IGNORECASE)


@functools.cache
def get_balanced_feat_pattern() -> re.Pattern[str]:
    cfg = get_config()
    words = "|".join(
        rf"{re.escape(c)}\.?" if c in ("feat", "ft") else rf"{re.escape(c)}\b"
        for c in cfg.featuring_conjunctions
        if c not in ("with", "and", "vs")
    )
    return re.compile(rf"^(?:{words})\s+(.*)$", re.IGNORECASE)


@functools.cache
def get_feat_tokens_pattern() -> re.Pattern[str]:
    cfg = get_config()
    dotted = [c for c in cfg.featuring_conjunctions if c in ("feat", "ft", "vs")]
    regular = [c for c in cfg.featuring_conjunctions if c not in ("feat", "ft", "vs")]

    parts: list[str] = [r"[,;&+]"]
    if dotted:
        dot_expr = "|".join(re.escape(c) for c in dotted)
        parts.append(rf"\b(?:{dot_expr})\.?(?!\w)")
    if regular:
        reg_expr = "|".join(re.escape(c) for c in regular)
        parts.append(rf"\b(?:{reg_expr})\b")

    return re.compile("|".join(parts), re.IGNORECASE)
