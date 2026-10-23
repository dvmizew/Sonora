from __future__ import annotations

import functools
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dotenv import load_dotenv
from rich.markup import escape

from sonora.core.constants import DIRS
from sonora.core.logger import LOG

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@functools.cache
def _default_policy() -> Mapping[str, Any]:
    defaults = json.loads(
        (Path(__file__).parent.parent / "defaults.json").read_text(encoding="utf-8")
    )
    return MappingProxyType(
        {
            name: tuple(value)
            if isinstance(value, list)
            else MappingProxyType(value)
            if isinstance(value, dict)
            else value
            for name, value in defaults.items()
        }
    )


@dataclass(frozen=True)
class SonoraConfig:
    """Immutable, typed Sonora application configuration with XDG and environment support."""

    artist_match_threshold: float = field(
        default_factory=lambda: float(_default_policy()["artist_match_threshold"])
    )
    featured_artist_duplicate_threshold: float = field(
        default_factory=lambda: float(
            _default_policy()["featured_artist_duplicate_threshold"]
        )
    )
    group_member_match_threshold: float = field(
        default_factory=lambda: float(_default_policy()["group_member_match_threshold"])
    )
    album_match_threshold: float = field(
        default_factory=lambda: float(_default_policy()["album_match_threshold"])
    )
    genius_match_threshold: float = field(
        default_factory=lambda: float(_default_policy()["genius_match_threshold"])
    )
    title_match_threshold: float = field(
        default_factory=lambda: float(_default_policy()["title_match_threshold"])
    )
    short_title_match_threshold: float = field(
        default_factory=lambda: float(_default_policy()["short_title_match_threshold"])
    )
    short_title_length: int = field(
        default_factory=lambda: int(_default_policy()["short_title_length"])
    )
    artist_similarity_threshold: float = field(
        default_factory=lambda: float(_default_policy()["artist_similarity_threshold"])
    )
    remix_match_threshold: float = field(
        default_factory=lambda: float(_default_policy()["remix_match_threshold"])
    )
    version_equivalence_penalty: float = field(
        default_factory=lambda: float(_default_policy()["version_equivalence_penalty"])
    )
    version_mismatch_penalty: float = field(
        default_factory=lambda: float(_default_policy()["version_mismatch_penalty"])
    )
    remix_modifier_mismatch_penalty: float = field(
        default_factory=lambda: float(
            _default_policy()["remix_modifier_mismatch_penalty"]
        )
    )
    title_score_weight: float = field(
        default_factory=lambda: float(_default_policy()["title_score_weight"])
    )
    single_duration_tolerance_seconds: float = field(
        default_factory=lambda: float(
            _default_policy()["single_duration_tolerance_seconds"]
        )
    )
    single_identifier_title_threshold: float = field(
        default_factory=lambda: float(
            _default_policy()["single_identifier_title_threshold"]
        )
    )
    allow_unverified_lyrics: bool = field(
        default_factory=lambda: bool(_default_policy()["allow_unverified_lyrics"])
    )
    acoustid_match_margin: float = field(
        default_factory=lambda: float(_default_policy()["acoustid_match_margin"])
    )
    acoustid_confidence_threshold: float = field(
        default_factory=lambda: float(
            _default_policy()["acoustid_confidence_threshold"]
        )
    )
    key_confidence_threshold: float = field(
        default_factory=lambda: float(_default_policy()["key_confidence_threshold"])
    )
    bpm_search_min: float = field(
        default_factory=lambda: float(_default_policy()["bpm_search_min"])
    )
    bpm_search_max: float = field(
        default_factory=lambda: float(_default_policy()["bpm_search_max"])
    )
    bpm_octave_min: float | None = None
    bpm_octave_max: float | None = None

    artist_credit_match_threshold: float = field(
        default_factory=lambda: float(
            _default_policy()["artist_credit_match_threshold"]
        )
    )
    related_artist_match_threshold: float = field(
        default_factory=lambda: float(
            _default_policy()["related_artist_match_threshold"]
        )
    )
    related_album_match_threshold: float = field(
        default_factory=lambda: float(
            _default_policy()["related_album_match_threshold"]
        )
    )
    album_token_match_threshold: float = field(
        default_factory=lambda: float(_default_policy()["album_token_match_threshold"])
    )
    catalog_candidate_rank_threshold: float = field(
        default_factory=lambda: float(
            _default_policy()["catalog_candidate_rank_threshold"]
        )
    )
    album_candidate_rank_threshold: float = field(
        default_factory=lambda: float(
            _default_policy()["album_candidate_rank_threshold"]
        )
    )

    disc_folder_patterns: tuple[str, ...] = field(
        default_factory=lambda: tuple(_default_policy()["disc_folder_patterns"])
    )
    generic_containers: frozenset[str] = field(
        default_factory=lambda: frozenset(_default_policy()["generic_containers"])
    )
    codec_rip_keywords: frozenset[str] = field(
        default_factory=lambda: frozenset(_default_policy()["codec_rip_keywords"])
    )
    featuring_conjunctions: tuple[str, ...] = field(
        default_factory=lambda: tuple(_default_policy()["featuring_conjunctions"])
    )
    noise_genres: frozenset[str] = field(
        default_factory=lambda: frozenset(_default_policy()["noise_genres"])
    )
    outlier_genres: frozenset[str] = field(
        default_factory=lambda: frozenset(_default_policy()["outlier_genres"])
    )
    genre_aliases: tuple[tuple[str, str], ...] = field(
        default_factory=lambda: tuple(_default_policy()["genre_aliases"].items())
    )

    known_group_artists: frozenset[str] = field(
        default_factory=lambda: frozenset(_default_policy()["known_group_artists"])
    )

    discogs_token: str | None = None
    acoustid_api_key: str | None = None
    genius_api_token: str | None = None
    lastfm_api_key: str | None = None
    musixmatch_token: str | None = None
    fanart_api_key: str | None = None
    fanart_client_key: str | None = None
    enable_shazam: bool = True

    def __post_init__(self) -> None:
        for threshold in (
            self.artist_match_threshold,
            self.featured_artist_duplicate_threshold,
            self.group_member_match_threshold,
            self.album_match_threshold,
            self.genius_match_threshold,
            self.title_match_threshold,
            self.short_title_match_threshold,
            self.artist_similarity_threshold,
            self.remix_match_threshold,
            self.version_equivalence_penalty,
            self.version_mismatch_penalty,
            self.remix_modifier_mismatch_penalty,
            self.single_identifier_title_threshold,
            self.artist_credit_match_threshold,
            self.related_artist_match_threshold,
            self.related_album_match_threshold,
            self.album_token_match_threshold,
            self.catalog_candidate_rank_threshold,
            self.album_candidate_rank_threshold,
        ):
            if not 0 <= threshold <= 100:
                raise ValueError("Matching thresholds must be between 0 and 100")
        if (
            self.title_match_threshold < 85
            or self.short_title_match_threshold < 95
            or self.short_title_length < 8
        ):
            raise ValueError(
                "Title matching cannot relax the 85/95 safety thresholds or eight-character boundary"
            )
        for confidence in (
            self.acoustid_confidence_threshold,
            self.acoustid_match_margin,
            self.key_confidence_threshold,
            self.title_score_weight,
        ):
            if not 0 <= confidence <= 1:
                raise ValueError("Confidence thresholds must be between 0 and 1")
        if (
            not math.isfinite(self.single_duration_tolerance_seconds)
            or self.single_duration_tolerance_seconds < 0
        ):
            raise ValueError("Single duration tolerance must be finite and nonnegative")
        if not 0 < self.bpm_search_min < self.bpm_search_max or not math.isfinite(
            self.bpm_search_max
        ):
            raise ValueError(
                "BPM search bounds must be finite, positive, and increasing"
            )
        if (self.bpm_octave_min is None) != (self.bpm_octave_max is None):
            raise ValueError("Configure both BPM octave bounds or neither")
        if (
            self.bpm_octave_min is not None
            and self.bpm_octave_max is not None
            and (
                not 0 < self.bpm_octave_min
                or not math.isfinite(self.bpm_octave_max)
                or self.bpm_octave_max < 2 * self.bpm_octave_min
            )
        ):
            raise ValueError(
                "BPM octave bounds must be finite and span at least one octave"
            )
        for pattern in self.disc_folder_patterns:
            re.compile(pattern)

    def is_disc_folder(self, folder_name: str) -> bool:
        clean = folder_name.strip()
        for folder_pattern in self.disc_folder_patterns:
            if re.match(folder_pattern, clean, re.IGNORECASE):
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
                parsed_config_data = tomllib.load(config_file)
        elif suffix == ".json":
            parsed_config_data = json.loads(path.read_text(encoding="utf-8"))
        else:
            return {}
        if not isinstance(parsed_config_data, dict):
            raise TypeError("Configuration must contain an object")
        sonora_section = parsed_config_data.get("sonora", parsed_config_data)
        if not isinstance(sonora_section, dict):
            raise TypeError("The sonora configuration section must contain an object")
        return {str(key): value for key, value in sonora_section.items()}
    except (OSError, ValueError) as error:
        LOG.warning(
            f"Failed to parse configuration file {escape(str(path))}: {escape(str(error))}"
        )
    return {}


def _safe_load_dotenv(path: Path, seen_paths: set[Path]) -> None:
    try:
        resolved = path.resolve()
        if (
            resolved not in seen_paths
            and resolved.is_file()
            and resolved.stat().st_size > 0
        ):
            seen_paths.add(resolved)
            load_dotenv(dotenv_path=resolved)
    except OSError:
        pass


def load_app_environment(target_path: Path | None = None) -> None:
    """
    Load .env files from XDG config directories, the working directory,
    and target directory hierarchies.
    """
    seen_paths: set[Path] = set()

    load_dotenv()

    _safe_load_dotenv(DIRS.user_config_path / ".env", seen_paths)

    if target_path is not None:
        try:
            current_directory = target_path.resolve()
            if not current_directory.is_dir():
                current_directory = current_directory.parent
            while current_directory != current_directory.parent:
                _safe_load_dotenv(current_directory / ".env", seen_paths)
                current_directory = current_directory.parent
        except (OSError, RuntimeError):
            pass


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
    load_app_environment()
    config_file = _discover_config_file()
    raw_data = _parse_config_data(config_file) if config_file is not None else {}

    from sonora.core.text import safe_float

    kwargs: dict[str, Any] = {}

    if "allow_unverified_lyrics" in raw_data:
        if not isinstance(raw_data["allow_unverified_lyrics"], bool):
            raise TypeError("allow_unverified_lyrics must be a boolean")
        kwargs["allow_unverified_lyrics"] = raw_data["allow_unverified_lyrics"]

    for policy_name in (
        "artist_match_threshold",
        "featured_artist_duplicate_threshold",
        "group_member_match_threshold",
        "album_match_threshold",
        "genius_match_threshold",
        "title_match_threshold",
        "short_title_match_threshold",
        "artist_similarity_threshold",
        "remix_match_threshold",
        "version_equivalence_penalty",
        "version_mismatch_penalty",
        "remix_modifier_mismatch_penalty",
        "title_score_weight",
        "single_duration_tolerance_seconds",
        "single_identifier_title_threshold",
        "artist_credit_match_threshold",
        "related_artist_match_threshold",
        "related_album_match_threshold",
        "album_token_match_threshold",
        "catalog_candidate_rank_threshold",
        "album_candidate_rank_threshold",
        "acoustid_confidence_threshold",
        "acoustid_match_margin",
        "key_confidence_threshold",
        "bpm_search_min",
        "bpm_search_max",
        "bpm_octave_min",
        "bpm_octave_max",
    ):
        if policy_name not in raw_data:
            continue
        if (
            policy_name in ("bpm_octave_min", "bpm_octave_max")
            and raw_data[policy_name] is None
        ):
            kwargs[policy_name] = None
            continue
        if isinstance(raw_data[policy_name], bool):
            raise TypeError(f"{policy_name} must be a number, not a boolean")
        configured_value = safe_float(raw_data[policy_name])
        if configured_value is None:
            raise ValueError(f"{policy_name} must be a finite number")
        kwargs[policy_name] = configured_value
    if "short_title_length" in raw_data:
        short_title_length = raw_data["short_title_length"]
        if isinstance(short_title_length, bool) or not isinstance(
            short_title_length, int
        ):
            raise TypeError("short_title_length must be an integer")
        kwargs["short_title_length"] = short_title_length

    for policy_field in (
        "disc_folder_patterns",
        "generic_containers",
        "codec_rip_keywords",
        "featuring_conjunctions",
        "known_group_artists",
        "noise_genres",
        "outlier_genres",
    ):
        if policy_field in raw_data:
            configured_strings = raw_data[policy_field]
            if not isinstance(configured_strings, list) or any(
                not isinstance(entry, str) for entry in configured_strings
            ):
                raise TypeError(f"{policy_field} must be an array of strings")
            if any(not entry.strip() for entry in configured_strings):
                raise ValueError(f"{policy_field} cannot contain blank strings")
            normalized_strings = (
                entry.strip()
                if policy_field == "disc_folder_patterns"
                else entry.strip().lower()
                for entry in configured_strings
            )
            kwargs[policy_field] = (
                tuple(normalized_strings)
                if policy_field in ("disc_folder_patterns", "featuring_conjunctions")
                else frozenset(normalized_strings)
            )
    if "genre_aliases" in raw_data:
        aliases = raw_data["genre_aliases"]
        if not isinstance(aliases, dict) or any(
            not isinstance(genre, str) or not isinstance(canonical, str)
            for genre, canonical in aliases.items()
        ):
            raise TypeError("genre_aliases must map names to canonical genre strings")
        kwargs["genre_aliases"] = tuple(
            (genre.lower(), canonical) for genre, canonical in aliases.items()
        )

    for credential, config_names, environment_names in (
        (
            "discogs_token",
            ("discogs_token", "discogs_user_token"),
            ("DISCOGS_TOKEN", "DISCOGS_USER_TOKEN"),
        ),
        (
            "acoustid_api_key",
            ("acoustid_api_key", "acoustid_key"),
            ("ACOUSTID_API_KEY",),
        ),
        (
            "genius_api_token",
            ("genius_api_token", "genius_token"),
            ("GENIUS_API_TOKEN", "GENIUS_TOKEN"),
        ),
        ("lastfm_api_key", ("lastfm_api_key", "lastfm_key"), ("LASTFM_API_KEY",)),
        (
            "musixmatch_token",
            ("musixmatch_token",),
            ("MUSIXMATCH_TOKEN", "MUSIXMATCH_USER_TOKEN"),
        ),
        (
            "fanart_api_key",
            ("fanart_api_key", "fanart_key"),
            ("FANART_API_KEY", "FANART_KEY"),
        ),
        (
            "fanart_client_key",
            ("fanart_client_key", "fanart_client"),
            ("FANART_CLIENT_KEY",),
        ),
    ):
        configured_secret = next(
            (raw_data[name] for name in config_names if raw_data.get(name)), None
        )
        if not configured_secret:
            configured_secret = next(
                (
                    os.environ[name]
                    for name in environment_names
                    if os.environ.get(name)
                ),
                None,
            )
        if configured_secret:
            kwargs[credential] = str(configured_secret).strip()

    shazam_env = os.environ.get("ENABLE_SHAZAM") or os.environ.get(
        "SONORA_ENABLE_SHAZAM"
    )
    if shazam_env is not None:
        normalized_shazam = shazam_env.strip().lower()
        if normalized_shazam not in (
            "0",
            "false",
            "no",
            "off",
            "1",
            "true",
            "yes",
            "on",
        ):
            raise ValueError("ENABLE_SHAZAM must be a boolean")
        kwargs["enable_shazam"] = normalized_shazam in ("1", "true", "yes", "on")
    elif "enable_shazam" in raw_data:
        if not isinstance(raw_data["enable_shazam"], bool):
            raise TypeError("enable_shazam must be a boolean")
        kwargs["enable_shazam"] = raw_data["enable_shazam"]

    return SonoraConfig(**kwargs)


def clear_config_cache() -> None:
    get_config.cache_clear()
    get_artist_split_pattern.cache_clear()
    get_disambiguation_pattern.cache_clear()
    get_duplicate_feat_pattern.cache_clear()
    get_bracket_feat_pattern.cache_clear()
    get_balanced_feat_pattern.cache_clear()
    get_feat_tokens_pattern.cache_clear()

    from sonora.core.cache import clear_normalization_cache

    clear_normalization_cache()


@functools.cache
def get_artist_split_pattern() -> re.Pattern[str]:
    config = get_config()
    parts: list[str] = []
    for c in config.featuring_conjunctions:
        escaped = re.escape(c)
        if c in ("feat", "ft", "vs"):
            parts.append(rf"\s+{escaped}\.?\s+")
        elif c == "w/":
            parts.append(rf"\s+{escaped}(?!\s*[oO](?:ut)?\b)\s*")
        else:
            parts.append(rf"\s+{escaped}\s+")
    parts.extend(
        [
            r"\s+[xX\u00d7]\s+",
            r"\s*&\s*",
            r"\s*,\s*",
            r"\s*;\s*",
            r"\s+/\s+",
        ]
    )
    return re.compile("|".join(parts), re.IGNORECASE)


@functools.cache
def get_disambiguation_pattern() -> re.Pattern[str]:
    config = get_config()
    words: list[str] = []
    for c in config.featuring_conjunctions:
        escaped = re.escape(c)
        if c in ("feat", "ft", "vs"):
            words.append(rf"{escaped}\.?")
        elif c == "w/":
            words.append(rf"{escaped}(?!\s*[oO](?:ut)?\b)")
        elif c.endswith("."):
            words.append(rf"{escaped}")
        else:
            words.append(rf"{escaped}\b")
    words_expr = "|".join(words)
    return re.compile(
        rf"\s*\([^()]{{1,40}}\)(?=\s*(?:[,;/&+-\\]|\b(?:{words_expr})|$))",
        re.IGNORECASE,
    )


@functools.cache
def get_duplicate_feat_pattern() -> re.Pattern[str]:
    config = get_config()
    words = "|".join(
        rf"{re.escape(c)}\.?" if c in ("feat", "ft") else re.escape(c)
        for c in config.featuring_conjunctions
        if c not in ("with", "and", "vs")
    )
    return re.compile(
        rf"\s+(?:{words})\s+([A-Za-z0-9\s\.\'\-]+?)(?=\s*[\(\[\{{]\s*(?:{words})\s+\1[\)\]\}}])",
        re.IGNORECASE,
    )


def _featuring_token_expression() -> str:
    words: list[str] = []
    for c in get_config().featuring_conjunctions:
        if c in ("with", "and", "vs"):
            continue
        escaped = re.escape(c)
        if c in ("feat", "ft"):
            words.append(rf"{escaped}\.?")
        elif c == "w/":
            words.append(rf"{escaped}(?!\s*[oO](?:ut)?\b)")
        elif c.endswith("."):
            words.append(rf"{escaped}")
        else:
            words.append(rf"{escaped}\b")
    return "|".join(words)


@functools.cache
def get_bracket_feat_pattern() -> re.Pattern[str]:
    return re.compile(
        rf"\s*[\(\[\{{]\s*(?:{_featuring_token_expression()})\s+.*?[\)\]\}}]",
        re.IGNORECASE,
    )


@functools.cache
def get_balanced_feat_pattern() -> re.Pattern[str]:
    return re.compile(rf"^(?:{_featuring_token_expression()})\s+(.*)$", re.IGNORECASE)


@functools.cache
def get_feat_tokens_pattern() -> re.Pattern[str]:
    config = get_config()
    dotted = [c for c in config.featuring_conjunctions if c in ("feat", "ft", "vs")]
    regular = [
        c
        for c in config.featuring_conjunctions
        if c not in ("feat", "ft", "vs", "w.", "w/", "w")
    ]

    parts: list[str] = [
        r"[,;]",
        r"\s+[&+]|\s*[&+](?=\s)",
        r"\bw/(?!\s*[oO](?:ut)?\b)\s*",
        r"\bw\.\s*",
    ]
    if dotted:
        dot_clean = [c.rstrip(".") for c in dotted]
        dot_expr = "|".join(re.escape(c) for c in dot_clean)
        parts.append(rf"\b(?:{dot_expr})\.?(?!\w)")
    if regular:
        reg_expr = "|".join(re.escape(c) for c in regular)
        parts.append(rf"\b(?:{reg_expr})\b")

    return re.compile("|".join(parts), re.IGNORECASE)
