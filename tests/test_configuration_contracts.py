import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from sonora.core import config
from sonora.core.artists import get_primary_artist
from sonora.core.constants import DIRS
from sonora.core.models import TrackInfo
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
from sonora.core.utils import clear_utils_cache
from tests.support import IsolatedTestCase


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
            0,
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
        clear_utils_cache()
        self.assertEqual(clean_title.cache_info().currsize, 0)
        self.assertEqual(normalize_country_name.cache_info().currsize, 0)
        self.assertEqual(normalize_str.cache_info().currsize, 0)
        self.assertEqual(normalize_language_name.cache_info().currsize, 0)
