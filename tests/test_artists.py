import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from sonora.core import artists
from sonora.core.artists import (
    extract_featured_artist_tokens,
    get_primary_artist,
    is_single_group_artist,
    normalize_artist_name,
    normalize_featured_artists,
)
from sonora.core.config import SonoraConfig, clear_config_cache
from sonora.core.constants import DIRS
from sonora.core.http import SESSION
from sonora.core.state import get_library_state
from sonora.services import artist_resolution, musicbrainz
from sonora.services.artist_resolution import resolve_artist_name
from tests.support import IsolatedTestCase


class TestArtistNormalization(IsolatedTestCase):
    def test_duplicate_alias_search_paths_are_loaded_once(self) -> None:
        config_root = self.test_directory / "home" / ".config"
        alias_path = config_root / "sonora" / "aliases.json"
        alias_path.parent.mkdir(parents=True)
        alias_path.write_text('{"Legal Name": "Stage"}')
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_root)}):
            self.assertEqual(artists._load_user_overrides(), {"legal name": "Stage"})

    def test_aliases_control_offline_resolution_and_registered_groups(self) -> None:
        alias_path = DIRS.user_config_path / "aliases.json"
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        alias_path.write_text(
            json.dumps({"Legal Name": "Stage", "Alpha & Beta": "Ensemble"})
        )
        self.assertEqual(artists.normalize_artist_name("Legal Name"), "Stage")
        self.assertTrue(artists.is_single_group_artist("Alpha & Beta"))
        self.assertEqual(
            artists.extract_featured_artist_tokens("Legal Name"),
            ["Stage"],
        )
        with patch.object(
            Path, "home", return_value=DIRS.user_config_path.parent.parent
        ):
            artists._load_user_overrides.cache_clear()
            self.assertEqual(artists._load_user_overrides()["legal name"], "Stage")

    def test_artist_casing_retains_distinct_names_and_reference_acronyms(self) -> None:
        self.assertEqual(artists.harmonize_artist_casing("First", "Second"), "First")
        self.assertEqual(artists.harmonize_artist_casing("ABC", "abc"), "abc")
        self.assertEqual(artists.harmonize_artist_casing("Artist", "artist"), "artist")

    def test_algorithmic_artist_normalization_rules(self) -> None:
        cases = {
            # Acronyms (consonant-only)
            "pnl": "PNL",
            "mgl": "MGL",
            "dmx": "DMX",
            "ctc": "CTC",
            "dzws": "DZWS",
            # Dotted initials
            "m.g.l.": "M.G.L.",
            "m.g.l": "M.G.L.",
            "t.i.": "T.I.",
            "t.i": "T.I.",
            "d.o.c.": "D.O.C.",
            # Currency signs and leetspeak ($)
            "asap rocky": "A$AP Rocky",
            "a$ap ferg": "A$AP Ferg",
            "asap mob": "A$AP Mob",
            "joey badass": "Joey Bada$$",
            "currensy": "Curren$y",
            "suicideboys": "$uicideboy$",
            "ty dolla sign": "Ty Dolla $ign",
            # Romanian hip-hop band formats & diacritics
            "satra benz": "ȘATRA B.E.N.Z.",
            "bug mafia": "B.U.G. Mafia",
            "parazitii": "Paraziții",
            "subcarpati": "Subcarpați",
            "sisu tudor": "Sișu Tudor",
            "dragonu ak47": "Dragonu'",
            "deliric 1": "Deliric",
            "albert nbn": "Albert Nbn",
            # Known algorithmic stylizations
            "6lack": "6LACK",
            "nf": "NF",
            "idk": "IDK",
            "atb": "ATB",
            "nosfe": "NOSFE",
            "rava": "RAVA",
            "renvto": "RENVTØ",
            "noua unspe": "NOUA UNSPE",
            "911": "NOUA UNSPE",
            # Normal names preserved
            "Kendrick Lamar": "Kendrick Lamar",
            "Mario": "Mario",
            "The Game": "The Game",
            "Eminem": "Eminem",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(artists.normalize_artist_name(raw), expected)

    def test_persisted_sqlite_artist_aliases_and_precedence(self) -> None:

        state_mgr = get_library_state()
        state_mgr.record_artist_alias("k-dot", "Kendrick Lamar", source="musicbrainz")

        artists.clear_artist_cache()
        # Persisted alias used when no user override exists
        self.assertEqual(artists.normalize_artist_name("K-Dot"), "Kendrick Lamar")

        # User override takes precedence over persisted alias
        alias_path = DIRS.user_config_path / "aliases.json"
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        alias_path.write_text(json.dumps({"k-dot": "Custom King"}))
        artists.clear_artist_cache()
        self.assertEqual(artists.normalize_artist_name("K-Dot"), "Custom King")

    def test_artist_resolution_persists_discovered_aliases_to_library_state(
        self,
    ) -> None:

        mb_payload = [
            {
                "name": "Kendrick Lamar",
                "alias-list": [
                    {"alias": "K-Dot", "type": "Artist name"},
                    {"alias": "Kung Fu Kenny", "type": "Artist name"},
                ],
            }
        ]
        with (
            patch.object(
                musicbrainz, "search_musicbrainz_artists", return_value=mb_payload
            ),
            patch.object(artist_resolution, "SESSION") as mock_session,
        ):
            mock_session.get.return_value.status_code = 200
            mock_session.get.return_value.json.return_value = {"data": []}

            resolved = artist_resolution.resolve_artist_name("k-dot")
            self.assertEqual(resolved, "Kendrick Lamar")

            state_mgr = get_library_state()
            self.assertEqual(state_mgr.get_artist_alias("k-dot"), "Kendrick Lamar")
            self.assertEqual(
                state_mgr.get_artist_alias("kung fu kenny"), "Kendrick Lamar"
            )

    def test_user_alias_files_empty_malformed_and_nonobject_content(self) -> None:
        path = DIRS.user_config_path / "aliases.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        for content, expected in (
            ("", {}),
            ("broken", {}),
            ("[]", {}),
            (json.dumps({"Legal Name": "Stage Name"}), {"legal name": "Stage Name"}),
        ):
            artists._load_user_overrides.cache_clear()
            path.write_text(content)
            with self.subTest(content=content):
                self.assertEqual(artists._load_user_overrides(), expected)

    def test_artist_resolution_obeys_alias_and_punctuation_precedence(self) -> None:
        cases: tuple[
            tuple[str, list[dict[str, object]], list[dict[str, object]], str], ...
        ] = (
            ("Primary", [{"name": "Primary!"}], [], "Primary"),
            ("Primary", [], [{"name": "Primary!"}], "Primary"),
            ("Primary", [], [{"name": "Primary"}], "Primary"),
            ("Primary", [], [{"name": "Wrong"}], "Primary"),
            (
                "Legal Name",
                [
                    {
                        "name": "Ensemble",
                        "alias-list": [
                            {},
                            {"alias": "Unrelated"},
                            {"alias": "Legal Name"},
                        ],
                    }
                ],
                [],
                "Ensemble",
            ),
            ("Primary", [{"name": "Primary"}], [], "Primary"),
            ("Singer", [{"name": "SINGER"}], [], "Singer"),
            ("ABC", [{"name": "Abc"}], [], "ABC"),
            ("Primary", [{"name": "P.r.i.m.a.r.y"}], [], "Primary"),
            ("primary", [], [{"name": "Primary"}], "Primary"),
            ("Singer", [], [{"name": "SINGER"}], "Singer"),
            ("ABC", [], [{"name": "Abc"}], "ABC"),
            ("primary", [], [{"name": "P.r.i.m.a.r.y"}, {}], "primary"),
            (
                "Legal Name",
                [
                    {
                        "name": "Stage",
                        "alias-list": [{"alias": "Legal Name", "type": "Legal name"}],
                    }
                ],
                [],
                "Stage",
            ),
            (
                "Alias",
                [
                    {
                        "name": "Stage",
                        "alias-list": [{"alias": "Alias", "type": "Search hint"}],
                    }
                ],
                [],
                "Alias",
            ),
            (
                "Alias",
                [
                    {
                        "name": "Stage",
                        "alias-list": [{"alias": "Alias", "type": "Legal name"}],
                    }
                ],
                [],
                "Alias",
            ),
            (
                "Alias",
                [{"name": "Stage Name", "alias-list": [{"alias": "Alias"}]}],
                [],
                "Alias",
            ),
            (
                "Legal Name",
                [
                    {
                        "name": "Legal Name Ensemble",
                        "alias-list": [{"alias": "Legal Name"}],
                    }
                ],
                [],
                "Legal Name Ensemble",
            ),
            ("Alias", [{"name": "Stage", "alias-list": ["Alias"]}], [], "Stage"),
            ("Nobody", [{}, {"name": "Unrelated"}], [], "Nobody"),
        )
        for query, mb_candidates, deezer_candidates, expected in cases:
            with (
                self.subTest(
                    query=query, candidates=mb_candidates, deezer=deezer_candidates
                ),
                patch.object(artist_resolution, "get_cached_api", return_value=None),
                patch.object(
                    musicbrainz,
                    "search_musicbrainz_artists",
                    return_value=mb_candidates,
                ),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(200, json={"data": deezer_candidates}),
                ),
            ):
                self.assertEqual(artist_resolution.resolve_artist_name(query), expected)

    def test_cached_allcaps_artist_names_skip_provider_lookups(self) -> None:
        for name in ("RAVA", "BLANCO", "ELIO"):
            with (
                self.subTest(artist=name),
                patch.object(artist_resolution, "get_cached_api", return_value=name),
                patch.object(
                    musicbrainz, "search_musicbrainz_artists", return_value=[]
                ) as search,
                patch.object(
                    SESSION, "get", return_value=httpx.Response(404)
                ) as request,
            ):
                self.assertEqual(artist_resolution.resolve_artist_name(name), name)
                search.assert_not_called()
                request.assert_not_called()

    def test_joint_main_artists_are_not_repeated_as_features(self) -> None:
        self.assertEqual(
            artists.extract_featured_artist_tokens(
                ["Dora Gaitanovici", "Dani Bușoi", "Guest"],
                primary_artist="Dora Gaitanovici & Dani Bușoi",
            ),
            ["Guest"],
        )
        self.assertEqual(
            artists.extract_featured_artist_tokens(
                ["Primary", "Guest"], primary_artist="Primary feat. Guest"
            ),
            ["Guest"],
        )
        policy = SonoraConfig(known_group_artists=frozenset({"Alpha & Beta"}))
        with patch("sonora.core.artists.get_config", return_value=policy):
            self.assertEqual(
                artists.extract_featured_artist_tokens(
                    ["Alpha & Beta", "Alpha", "Guest"],
                    primary_artist="Alpha & Beta",
                ),
                ["Alpha", "Guest"],
            )

    def test_artist_provider_failures_preserve_clean_input(self) -> None:
        with (
            patch.object(artist_resolution, "get_cached_api", return_value=None),
            patch.object(
                musicbrainz,
                "search_musicbrainz_artists",
                side_effect=OSError("offline"),
            ),
            patch.object(SESSION, "get", side_effect=OSError("offline")),
        ):
            self.assertEqual(artist_resolution.resolve_artist_name("Singer"), "Singer")
            self.assertFalse(artists.is_single_group_artist("Alpha & Beta"))

    def test_artist_resolution_exposes_malformed_deezer_responses(self) -> None:
        payload: object
        for payload in ([], {"data": "wrong"}, {"data": [None]}):
            with (
                self.subTest(payload=payload),
                patch(
                    "sonora.services.musicbrainz.search_musicbrainz_artists",
                    return_value=[],
                ),
                patch("sonora.services.artist_resolution.SESSION.get") as request,
            ):
                request.return_value.status_code = 200
                request.return_value.json.return_value = payload
                with self.assertRaises(TypeError):
                    artist_resolution.resolve_artist_name("Artist Alpha")

    def test_normalize_featured_artists(self) -> None:
        self.assertIsNone(normalize_featured_artists(None))
        self.assertIsNone(normalize_featured_artists(""))
        self.assertIsNone(normalize_featured_artists([]))

        # Composite conjunction string deduplication
        self.assertEqual(
            extract_featured_artist_tokens(
                "Project Pat, ScHoolboy Q, Project Pat & ScHoolboy Q",
                primary_artist="21 Savage",
            ),
            ["Project Pat", "ScHoolboy Q"],
        )
        self.assertEqual(
            normalize_featured_artists(
                "Project Pat, ScHoolboy Q, Project Pat & ScHoolboy Q",
                primary_artist="21 Savage",
            ),
            "Project Pat, ScHoolboy Q",
        )
        self.assertEqual(
            normalize_featured_artists(
                ["Gunna, Lil Baby", "Gunna & Lil Baby"],
                primary_artist="21 Savage",
            ),
            "Gunna, Lil Baby",
        )

        # Conjunction variants ('&' vs 'si' / 'și')
        self.assertEqual(
            normalize_featured_artists(
                "Roxana & Puya, Roxana si Puya",
                primary_artist="B.U.G. Mafia",
            ),
            "Roxana, Puya",
        )
        self.assertEqual(
            normalize_featured_artists(
                "Nico, Luchian, Luchian si Nico",
                primary_artist="B.U.G. Mafia",
            ),
            "Nico, Luchian",
        )

        # Unicode diacritics / canonical repertoire reconciliation
        self.assertEqual(
            normalize_featured_artists("Dj Flama, DJ Flamă"),
            "DJ Flamă",
        )

        # Registered duo/group entity preservation
        with patch("sonora.core.artists.is_single_group_artist", return_value=True):
            self.assertEqual(
                normalize_featured_artists("Agnes, Vargas & Lagola"),
                "Agnes, Vargas & Lagola",
            )

        # Primary artist self-feature exclusion
        self.assertEqual(
            normalize_featured_artists(
                "21 Savage, Project Pat, ScHoolboy Q",
                primary_artist="21 Savage",
            ),
            "Project Pat, ScHoolboy Q",
        )

        # Multilingual with / w. / w/ conjunctions
        self.assertEqual(
            normalize_featured_artists(
                "Post Malone w. Swae Lee", primary_artist="Post Malone"
            ),
            "Swae Lee",
        )
        self.assertEqual(
            normalize_featured_artists("Drake w/ 21 Savage", primary_artist="Drake"),
            "21 Savage",
        )
        # Without (w/o) negative lookahead guarantee
        self.assertEqual(
            extract_featured_artist_tokens(
                "21 Savage w/o Metro Boomin",
                primary_artist="21 Savage",
            ),
            ["21 Savage w/o Metro Boomin"],
        )
        self.assertEqual(
            normalize_featured_artists(
                "21 Savage w/o Metro Boomin",
                primary_artist="21 Savage",
            ),
            "21 Savage w/o Metro Boomin",
        )

    @patch("sonora.services.artist_resolution.get_cached_api", return_value=None)
    def test_resolve_artist_name_exact_match(self, mock_get_cache: MagicMock) -> None:

        with patch("sonora.services.musicbrainz.search_musicbrainz_artists") as mock_mb:
            # Simulate MusicBrainz returning Enrico Rava with score 100 and RAVA with score 90
            mock_mb.return_value = [
                {"name": "Enrico Rava", "ext:score": "100", "id": "uuid-1"},
                {"name": "RAVA", "ext:score": "90", "id": "uuid-2"},
            ]
            # Must prioritize exact match RAVA over higher-scored Enrico Rava
            self.assertEqual(resolve_artist_name("RAVA"), "RAVA")
            self.assertEqual(resolve_artist_name("rava"), "RAVA")

    @patch("sonora.services.artist_resolution.get_cached_api", return_value=None)
    @patch("sonora.services.artist_resolution.set_cached_api")
    def test_resolve_artist_name_preserves_titlecase_over_allcaps(
        self, mock_set_cache: MagicMock, mock_get_cache: MagicMock
    ) -> None:

        with patch("sonora.services.musicbrainz.search_musicbrainz_artists") as mock_mb:
            # MusicBrainz returns CRUSH (Group, score 100)
            mock_mb.return_value = [
                {"name": "CRUSH", "type": "Group", "ext:score": "100", "id": "uuid-1"},
            ]
            # TitleCase "Crush" must not be converted to screaming "CRUSH"
            self.assertEqual(resolve_artist_name("Crush"), "Crush")

    def test_offline_resolution_ignores_corrupt_cached_expansion(self) -> None:

        with (
            patch(
                "sonora.services.artist_resolution.get_cached_api",
                return_value="Gruppe Drei",
            ),
            patch("sonora.services.artist_resolution.set_cached_api") as mock_set_cache,
        ):
            # Local normalization must not consult or mutate a provider cache.
            resolved_artist_name = normalize_artist_name("Drei")
            mock_set_cache.assert_not_called()
            self.assertEqual(resolved_artist_name, "Drei")

    @patch("sonora.services.artist_resolution.get_cached_api", return_value=None)
    def test_offline_artist_resolution_does_not_query_providers(
        self, mock_get_cache: MagicMock
    ) -> None:

        with patch("sonora.services.musicbrainz.search_musicbrainz_artists") as mock_mb:
            result = normalize_artist_name("Unknown Indierock Band")
            self.assertEqual(result, "Unknown Indierock Band")
            mock_mb.assert_not_called()

            primary = get_primary_artist("New Artist feat. Guest Singer")
            self.assertEqual(primary, "New Artist")
            mock_mb.assert_not_called()

    def test_is_single_group_artist(self) -> None:
        with patch(
            "sonora.core.artists.get_config",
            return_value=SonoraConfig(
                known_group_artists=frozenset({"Above & Beyond"})
            ),
        ):
            self.assertTrue(is_single_group_artist("Above & Beyond"))
            self.assertFalse(is_single_group_artist("Drake & 21 Savage"))
            self.assertFalse(is_single_group_artist("SingleArtist"))
            self.assertFalse(is_single_group_artist(""))
            self.assertFalse(is_single_group_artist(None))

    def test_get_primary_artist(self) -> None:
        groups = {"Above & Beyond", "Alan & Kepa", "Play & Win", "Simon & Garfunkel"}
        with patch(
            "sonora.core.artists.get_config",
            return_value=SonoraConfig(known_group_artists=frozenset(groups)),
        ):
            for group in groups:
                with self.subTest(group=group):
                    self.assertEqual(get_primary_artist(group), group)
            self.assertEqual(get_primary_artist("Play&Win"), "Play & Win")
            self.assertEqual(
                get_primary_artist("21 Savage feat. Metro Boomin"), "21 Savage"
            )
            self.assertEqual(get_primary_artist("Drake & Future"), "Drake")
            self.assertEqual(get_primary_artist("Samurai & El Nino"), "Samurai")
            self.assertEqual(get_primary_artist(""), "Unknown")
            self.assertEqual(get_primary_artist(None), "Unknown")

    def test_registered_comma_artist_preserves_its_name(self) -> None:
        self.assertEqual(get_primary_artist("Beach, Please!"), "Beach, Please!")
        self.assertEqual(
            extract_featured_artist_tokens("Beach, Please!"), ["Beach, Please!"]
        )
        self.assertEqual(get_primary_artist("Artist, Guest"), "Artist")
        self.assertEqual(
            extract_featured_artist_tokens("Artist, Guest"), ["Artist", "Guest"]
        )

    def test_load_user_overrides_corrupt_json(self) -> None:
        from sonora.core.artists import _load_user_overrides

        _load_user_overrides.cache_clear()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.stat", return_value=MagicMock(st_size=100)),
            patch("pathlib.Path.read_text", return_value="INVALID JSON {{"),
        ):
            overrides = _load_user_overrides()
            self.assertEqual(overrides, {})
        _load_user_overrides.cache_clear()

    def test_featured_artist_deduplication_uses_configured_threshold(self) -> None:
        for threshold, expected in ((88.0, []), (100.0, ["Example Artists"])):
            with (
                self.subTest(threshold=threshold),
                patch(
                    "sonora.core.artists.get_config",
                    return_value=SonoraConfig(
                        featured_artist_duplicate_threshold=threshold
                    ),
                ),
            ):
                self.assertEqual(
                    extract_featured_artist_tokens("Example Artists", "Example Artist"),
                    expected,
                )
                self.assertEqual(
                    extract_featured_artist_tokens(None, "Example Artist"), []
                )
                self.assertEqual(
                    extract_featured_artist_tokens(
                        "Different Performer", "Example Artist"
                    ),
                    ["Different Performer"],
                )
        for invalid in (-1.0, 101.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                SonoraConfig(featured_artist_duplicate_threshold=invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                SonoraConfig(group_member_match_threshold=invalid)

    def test_config_refresh_invalidates_derived_artist_classifications(self) -> None:
        configured_groups: dict[str, object] = {"known_group_artists": []}
        with (
            patch("sonora.core.config.load_app_environment"),
            patch(
                "sonora.core.config._discover_config_file",
                return_value=Path("config.json"),
            ),
            patch(
                "sonora.core.config._parse_config_data", return_value=configured_groups
            ),
        ):
            clear_config_cache()
            self.assertFalse(is_single_group_artist("Example & Ensemble"))
            configured_groups["known_group_artists"] = ["Example & Ensemble"]
            clear_config_cache()
            self.assertTrue(is_single_group_artist("Example & Ensemble"))
            configured_groups["known_group_artists"] = []
            clear_config_cache()
            self.assertFalse(is_single_group_artist("Example & Ensemble"))
