import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sonora.audio.bpm import calculate_bpm
from sonora.core import config
from sonora.core.artists import get_primary_artist
from sonora.core.cache import clear_normalization_cache
from sonora.core.config import (
    SonoraConfig,
    _default_policy,
    clear_config_cache,
    get_config,
)
from sonora.core.constants import DIRS
from sonora.core.models import TrackInfo
from sonora.core.state import tagging_state_status
from sonora.core.text import (
    is_noise_genre,
    is_valid_uuid,
    normalize_country_name,
    normalize_date,
    normalize_genre,
    normalize_language_name,
    normalize_legacy_diacritics,
    normalize_script_name,
    normalize_str,
    preserve_unicode_repertoire,
)
from sonora.core.titles import (
    clean_title,
    extract_title_features,
    score_artist_title_match,
)
from tests.support import IsolatedTestCase

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class TestConfigurationFiles(IsolatedTestCase):
    def test_environment_search_uses_audio_parent_and_deduplicates_default_paths(
        self,
    ) -> None:
        audio_path = self.test_directory / "recording.flac"
        audio_path.touch()
        environment_path = self.test_directory / ".env"
        environment_path.write_text("SONORA_TEST=fixture")
        with patch("sonora.core.config.load_dotenv") as load:
            config.load_app_environment(audio_path)
        load.assert_any_call(dotenv_path=environment_path.resolve())
        with patch.dict(os.environ, {"HOME": str(Path.home())}, clear=True):
            self.assertIsNone(config._discover_config_file())

    def test_json_and_toml_root_and_sections(self) -> None:
        payloads = (
            (".json", '{"enable_shazam": false}'),
            (".json", '{"sonora": {"enable_shazam": false}}'),
            (".toml", "enable_shazam = false"),
            (".toml", "[sonora]\nenable_shazam = false"),
        )
        for suffix, content in payloads:
            with self.subTest(suffix=suffix, content=content):
                path = self.test_directory / f"config{suffix}"
                path.write_text(content)
                self.assertEqual(
                    config._parse_config_data(path), {"enable_shazam": False}
                )

    def test_nonobject_json_and_sections_fail_fast(self) -> None:
        path = self.test_directory / "config.json"
        payloads: tuple[object, ...] = ([], 3, {"sonora": []})
        for payload in payloads:
            path.write_text(json.dumps(payload))
            with self.subTest(payload=payload), self.assertRaises(TypeError):
                config._parse_config_data(path)

    def test_unsupported_and_invalid_files_are_reported_or_empty(self) -> None:
        unsupported = self.test_directory / "settings.txt"
        unsupported.write_text("settings")
        self.assertEqual(config._parse_config_data(unsupported), {})
        path = self.test_directory / "config.json"
        path.write_text("broken json")
        with patch("sonora.core.logger.LOG.warning") as warning:
            self.assertEqual(config._parse_config_data(path), {})
        warning.assert_called_once()
        self.assertIn(str(path), warning.call_args.args[0])

    def test_toml_imports_follow_python_version(self) -> None:
        path = self.test_directory / "config.toml"
        path.write_text("[sonora]\nenable_shazam = false")
        for python_version in ((3, 10, 0), (3, 11, 0)):
            with self.subTest(python_version=python_version):
                module_name = "sonora.core.configuration_import_contract"
                specification = importlib.util.spec_from_file_location(
                    module_name, config.__file__
                )
                assert specification is not None and specification.loader is not None
                module = importlib.util.module_from_spec(specification)
                with (
                    patch.dict(
                        sys.modules,
                        {module_name: module, "tomli": tomllib, "tomllib": tomllib},
                    ),
                    patch.object(sys, "version_info", python_version),
                ):
                    specification.loader.exec_module(module)
                self.assertIs(module.tomllib, tomllib)
                self.assertEqual(
                    module._parse_config_data(path), {"enable_shazam": False}
                )

    def test_environment_discovery_is_deduplicated_and_survives_io_errors(self) -> None:
        path = self.test_directory / ".env"
        path.write_text("SONORA_TEST=true")
        seen: set[Path] = set()
        with patch("sonora.core.config.load_dotenv") as load:
            config._safe_load_dotenv(path, seen)
            config._safe_load_dotenv(path, seen)
        load.assert_called_once_with(dotenv_path=path.resolve())
        self.assertEqual(seen, {path.resolve()})
        with patch.object(Path, "resolve", side_effect=OSError("locked path")):
            config._safe_load_dotenv(path, set())
            config.load_app_environment(path)

    def test_discovery_prefers_user_configuration(self) -> None:
        path = DIRS.user_config_path / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("enable_shazam = false")
        self.assertEqual(config._discover_config_file(), path)
        with patch.object(Path, "is_file", return_value=False):
            self.assertIsNone(config._discover_config_file())
        with patch.object(
            Path, "resolve", side_effect=PermissionError("locked directory")
        ):
            self.assertIsNone(config._discover_config_file())

    def test_invalid_scalar_policies_do_not_silently_use_defaults(self) -> None:
        invalid_settings: tuple[dict[str, object], ...] = (
            {"title_match_threshold": True},
            {"title_match_threshold": "not a number"},
            {"short_title_length": 8.5},
            {"generic_containers": [""]},
            {"genre_aliases": []},
        )
        for values in invalid_settings:
            config.clear_config_cache()
            with (
                self.subTest(values=values),
                patch(
                    "sonora.core.config._discover_config_file",
                    return_value=Path("config.json"),
                ),
                patch("sonora.core.config._parse_config_data", return_value=values),
                self.assertRaises((TypeError, ValueError)),
            ):
                config.get_config()

    def test_optional_octave_policy_and_environment_credentials(self) -> None:
        values = {
            "bpm_octave_min": None,
            "bpm_octave_max": None,
            "short_title_length": 9,
            "noise_genres": ["Placeholder"],
            "outlier_genres": ["Other"],
            "known_group_artists": ["Example & Company"],
            "codec_rip_keywords": ["Codec"],
            "featuring_conjunctions": ["with"],
            "genre_aliases": {"test style": "Rock"},
        }
        credentials = {
            "DISCOGS_TOKEN": "discogs",
            "ACOUSTID_API_KEY": "acoustid",
            "GENIUS_API_TOKEN": "genius",
            "LASTFM_API_KEY": "lastfm",
            "MUSIXMATCH_TOKEN": "musixmatch",
            "FANART_API_KEY": "fanart",
            "FANART_CLIENT_KEY": "client",
        }
        config.clear_config_cache()
        with (
            patch.dict(os.environ, credentials),
            patch(
                "sonora.core.config._discover_config_file",
                return_value=Path("config.json"),
            ),
            patch("sonora.core.config._parse_config_data", return_value=values),
        ):
            configuration = config.get_config()
        self.assertEqual(configuration.discogs_token, "discogs")
        self.assertEqual(configuration.acoustid_api_key, "acoustid")
        self.assertEqual(configuration.genius_api_token, "genius")
        self.assertEqual(configuration.lastfm_api_key, "lastfm")
        self.assertEqual(configuration.musixmatch_token, "musixmatch")
        self.assertEqual(configuration.fanart_api_key, "fanart")
        self.assertEqual(configuration.fanart_client_key, "client")
        self.assertIsNone(configuration.bpm_octave_min)
        self.assertEqual(configuration.short_title_length, 9)
        self.assertEqual(configuration.genre_aliases, (("test style", "Rock"),))

    def test_file_boolean_and_invalid_environment_boolean(self) -> None:
        config.clear_config_cache()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("sonora.core.config.load_app_environment"),
            patch(
                "sonora.core.config._discover_config_file",
                return_value=Path("config.json"),
            ),
            patch(
                "sonora.core.config._parse_config_data",
                return_value={"enable_shazam": False},
            ),
        ):
            self.assertFalse(config.get_config().enable_shazam)
        config.clear_config_cache()
        with (
            patch.dict(os.environ, {"ENABLE_SHAZAM": "sometimes"}),
            self.assertRaisesRegex(ValueError, "ENABLE_SHAZAM"),
        ):
            config.get_config()

    def test_feature_patterns_support_empty_custom_token_sets(self) -> None:
        with patch(
            "sonora.core.config.get_config",
            return_value=config.SonoraConfig(featuring_conjunctions=()),
        ):
            self.assertIsNone(config.get_feat_tokens_pattern().search("feat Guest"))
            self.assertIsNone(config.get_feat_tokens_pattern().search("with Guest"))
        config.clear_config_cache()
        with patch(
            "sonora.core.config.get_config",
            return_value=config.SonoraConfig(featuring_conjunctions=("with",)),
        ):
            self.assertIsNotNone(config.get_feat_tokens_pattern().search("with Guest"))


class TestTextContracts(IsolatedTestCase):
    def test_unicode_quality_and_missing_titles(self) -> None:
        self.assertEqual(preserve_unicode_repertoire("Cafe", "Café"), "Café")
        self.assertEqual(extract_title_features(None), ("", []))
        self.assertEqual(
            score_artist_title_match(
                "Artist", "Song (Album Version)", "Artist", "Song"
            ),
            100,
        )

    def test_invalid_dates_and_identifiers_never_become_metadata(self) -> None:
        for date_value in ("1800-01-01", "1800", "2200-01-01", "2200"):
            with self.subTest(date_value=date_value):
                self.assertIsNone(normalize_date(date_value))
        for identifier in (
            " ",
            "x" * 36,
            ";;/",
            "11111111-2222-3333-4444-555555555555;wrong",
        ):
            with self.subTest(identifier=identifier):
                self.assertFalse(is_valid_uuid(identifier, allow_multivalue=True))
        self.assertTrue(is_noise_genre(" "))

    def test_country_fuzzy_lookup_empty_and_successful_results(self) -> None:
        normalize_country_name.cache_clear()
        with (
            patch("pycountry.countries.lookup", side_effect=LookupError),
            patch("pycountry.historic_countries.lookup", side_effect=LookupError),
            patch("pycountry.countries.search_fuzzy", return_value=[]),
        ):
            self.assertEqual(
                normalize_country_name("unrecognized place"), "unrecognized place"
            )
        normalize_country_name.cache_clear()
        from types import SimpleNamespace

        with (
            patch("pycountry.countries.lookup", side_effect=LookupError),
            patch("pycountry.historic_countries.lookup", side_effect=LookupError),
            patch(
                "pycountry.countries.search_fuzzy",
                return_value=[SimpleNamespace(name="United Kingdom")],
            ),
        ):
            self.assertEqual(normalize_country_name("England"), "United Kingdom")

    def test_explicit_album_artist_is_preserved_in_reports(self) -> None:
        self.assertEqual(
            TrackInfo(
                Path("track.flac"), artist="Guest", album_artist="Primary"
            ).to_dict()["album_artist"],
            "Primary",
        )

    def test_utility_cache_clear_evicts_populated_results(self) -> None:
        get_primary_artist("Artist")
        clean_title("Song")
        normalize_str("Artist")
        normalize_date("2020")
        normalize_genre("rock")
        normalize_country_name("US")
        normalize_language_name("eng")
        normalize_script_name("Latn")
        preserve_unicode_repertoire("Cafe", "Café")
        normalize_legacy_diacritics("Cafe")
        self.assertGreater(clean_title.cache_info().currsize, 0)
        self.assertGreater(normalize_country_name.cache_info().currsize, 0)
        clear_normalization_cache()
        self.assertEqual(clean_title.cache_info().currsize, 0)
        self.assertEqual(normalize_country_name.cache_info().currsize, 0)
        self.assertEqual(normalize_str.cache_info().currsize, 0)
        self.assertEqual(normalize_language_name.cache_info().currsize, 0)


class TestPolicyConfiguration(IsolatedTestCase):
    def test_defaults_are_loaded_once(self) -> None:
        _default_policy.cache_clear()
        original_read = Path.read_text
        with patch.object(
            Path, "read_text", autospec=True, wraps=Path.read_text
        ) as read_text:
            read_text.side_effect = original_read
            SonoraConfig()
            SonoraConfig()
        self.assertEqual(read_text.call_count, 1)

    def test_all_string_arrays_reject_invalid_types(self) -> None:
        for policy in (
            "disc_folder_patterns",
            "generic_containers",
            "codec_rip_keywords",
            "featuring_conjunctions",
            "known_group_artists",
            "noise_genres",
            "outlier_genres",
        ):
            for invalid in ("string", [1], [True], [None]):
                with self.subTest(policy=policy, value=invalid):
                    clear_config_cache()
                    with (
                        patch(
                            "sonora.core.config._discover_config_file",
                            return_value=Path("config.json"),
                        ),
                        patch(
                            "sonora.core.config._parse_config_data",
                            return_value={policy: invalid},
                        ),
                        self.assertRaisesRegex(TypeError, policy),
                    ):
                        get_config()

    def test_configuration_requires_real_boolean(self) -> None:
        value: object
        for value in ("false", 0, 1, [], None):
            with self.subTest(value=value):
                clear_config_cache()
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch("sonora.core.config.load_app_environment"),
                    patch(
                        "sonora.core.config._discover_config_file",
                        return_value=Path("config.json"),
                    ),
                    patch(
                        "sonora.core.config._parse_config_data",
                        return_value={"enable_shazam": value},
                    ),
                    self.assertRaisesRegex(TypeError, "enable_shazam"),
                ):
                    get_config()

    def test_environment_boolean_and_valid_arrays(self) -> None:
        for environment_value, expected in (("false", False), ("true", True)):
            clear_config_cache()
            with (
                patch.dict(os.environ, {"ENABLE_SHAZAM": environment_value}),
                patch(
                    "sonora.core.config._discover_config_file",
                    return_value=Path("config.json"),
                ),
                patch(
                    "sonora.core.config._parse_config_data",
                    return_value={
                        "disc_folder_patterns": ["^disk[0-9]+$"],
                        "generic_containers": ["Downloads"],
                    },
                ),
            ):
                configuration = get_config()
                self.assertIs(configuration.enable_shazam, expected)
                self.assertTrue(configuration.is_disc_folder("disk2"))
                self.assertTrue(configuration.is_generic_container("downloads"))

    def test_matching_policies_validate_bounds(self) -> None:
        invalid_configs: tuple[Callable[[], SonoraConfig], ...] = (
            lambda: SonoraConfig(version_mismatch_penalty=-1),
            lambda: SonoraConfig(remix_modifier_mismatch_penalty=101),
            lambda: SonoraConfig(title_score_weight=1.1),
            lambda: SonoraConfig(single_duration_tolerance_seconds=float("inf")),
        )
        for invalid_config in invalid_configs:
            with self.assertRaises(ValueError):
                invalid_config()

    def test_fingerprint_is_stable_in_separate_interpreters(self) -> None:
        script = (
            "from sonora.core.config import SonoraConfig; "
            "from sonora.core.state import tagging_state_status; "
            "print(tagging_state_status(SonoraConfig(),fetch_bpm=False,fetch_key=False,"
            "fetch_replaygain=False,fetch_lyrics=False,fetch_itunes_art=False))"
        )
        statuses = []
        for seed in ("1", "2", "3"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            )
            statuses.append(result.stdout.strip())
        self.assertEqual(len(set(statuses)), 1)

    def test_fingerprint_changes_for_meaningful_policy_changes(self) -> None:
        flags = {
            "fetch_bpm": False,
            "fetch_key": False,
            "fetch_replaygain": False,
            "fetch_lyrics": False,
            "fetch_itunes_art": False,
        }
        original = tagging_state_status(SonoraConfig(), **flags)
        self.assertNotEqual(
            original,
            tagging_state_status(SonoraConfig(), **{**flags, "fetch_bpm": True}),
        )
        self.assertNotEqual(
            original,
            tagging_state_status(
                dataclasses.replace(SonoraConfig(), title_match_threshold=90), **flags
            ),
        )


class TestConfigurablePolicies(IsolatedTestCase):
    def test_invalid_policy_bounds(self) -> None:
        invalid_configs: tuple[Callable[[], SonoraConfig], ...] = (
            lambda: SonoraConfig(title_match_threshold=84),
            lambda: SonoraConfig(short_title_match_threshold=94),
            lambda: SonoraConfig(short_title_length=7),
            lambda: SonoraConfig(acoustid_confidence_threshold=1.1),
            lambda: SonoraConfig(key_confidence_threshold=-0.1),
            lambda: SonoraConfig(bpm_search_min=0),
            lambda: SonoraConfig(bpm_search_max=float("inf")),
            lambda: SonoraConfig(bpm_search_min=200, bpm_search_max=60),
            lambda: SonoraConfig(bpm_octave_min=75),
            lambda: SonoraConfig(bpm_octave_min=75, bpm_octave_max=140),
        )
        for construct_config in invalid_configs:
            with (
                self.subTest(construct_config=construct_config),
                self.assertRaises(ValueError),
            ):
                construct_config()

    def test_configuration_loads_new_policies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                "[sonora]\ntitle_match_threshold = 91\nbpm_search_min = 40\nbpm_search_max = 240\nbpm_octave_min = 75\nbpm_octave_max = 190\n",
                encoding="utf-8",
            )
            with patch(
                "sonora.core.config._discover_config_file", return_value=config_path
            ):
                clear_config_cache()
                config = get_config()
            self.assertEqual(config.title_match_threshold, 91)
            self.assertEqual((config.bpm_search_min, config.bpm_search_max), (40, 240))
            self.assertEqual((config.bpm_octave_min, config.bpm_octave_max), (75, 190))

    def test_invalid_numeric_configuration(self) -> None:
        for configured in (True, "nan", "not a number"):
            with (
                self.subTest(configured=configured),
                patch(
                    "sonora.core.config._discover_config_file",
                    return_value=Path("config.json"),
                ),
                patch(
                    "sonora.core.config._parse_config_data",
                    return_value={"bpm_search_min": configured},
                ),
            ):
                clear_config_cache()
                with self.assertRaises((TypeError, ValueError)):
                    get_config()

    def test_matching_policy_changes_acceptance(self) -> None:
        self.assertGreater(
            score_artist_title_match(
                "Artist", "An ordinary song", "Artist", "An ordinary songs"
            ),
            0,
        )
        with patch(
            "sonora.core.titles.get_config",
            return_value=SonoraConfig(title_match_threshold=99),
        ):
            self.assertEqual(
                score_artist_title_match(
                    "Artist", "An ordinary song", "Artist", "An ordinary songs"
                ),
                0,
            )
            self.assertEqual(
                score_artist_title_match(
                    "Artist", "An ordinary song", "Artist", "An ordinary song"
                ),
                100,
            )
        self.assertEqual(
            score_artist_title_match("Artist", "Intro", "Artist", "Outro"), 0
        )

    def test_slow_tempo_not_doubled_unless_configured(self) -> None:
        sample_rate = 22050
        pulse_samples = int(sample_rate * 60 / 70)
        audio_samples = np.zeros(sample_rate * 20)
        for pulse_start in range(0, len(audio_samples), pulse_samples):
            pulse_length = min(256, len(audio_samples) - pulse_start)
            audio_samples[pulse_start : pulse_start + pulse_length] = np.hanning(
                pulse_length
            )
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "70bpm.wav"
            audio_path.touch()
            with patch(
                "sonora.audio.bpm.load_audio", return_value=(audio_samples, sample_rate)
            ):
                measured_bpm = calculate_bpm(audio_path)
                assert measured_bpm is not None
                self.assertAlmostEqual(measured_bpm, 70, delta=2)
                with patch(
                    "sonora.audio.bpm.get_config",
                    return_value=SonoraConfig(bpm_octave_min=75, bpm_octave_max=190),
                ):
                    folded_bpm = calculate_bpm(audio_path)
                assert folded_bpm is not None
                self.assertAlmostEqual(folded_bpm, 140, delta=4)
