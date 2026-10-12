from pathlib import Path
from unittest.mock import MagicMock, patch

from sonora.core.artists import (
    extract_featured_artist_tokens,
    get_primary_artist,
    is_single_group_artist,
    normalize_featured_artists,
)
from sonora.core.text import (
    clean_disambiguation,
    clean_unicode_punct,
    is_valid_uuid,
    normalize_date,
    normalize_genre,
    normalize_str,
    safe_float,
    safe_int,
    safe_int_pair,
    sanitize_name,
)
from sonora.core.titles import (
    clean_title,
    extract_title_features,
    score_artist_title_match,
)
from sonora.core.utils import (
    extract_disc_number_from_folder,
    group_files_by_album_root,
    group_files_by_parent,
    parse_track_filename,
    relocate_companion_lyrics,
    safe_case_rename,
)
from tests.support import IsolatedTestCase


class TestCoreUtils(IsolatedTestCase):
    def test_normalize_str_contracts(self) -> None:
        cases = (
            ("Hello World", "hello world"),
            ("$tring!", "string"),
            ("Beyoncé", "beyonce"),
            ("Mötley Crüe", "motley crue"),
            (None, ""),
            ("", ""),
            ("   ", ""),
            ("A$AP Rocky!", "asap rocky"),
            ("Beyoncé - Nöél", "beyonce noel"),
            ("Nimeni Altu'", "nimeni altu"),
            ("Satra B.E.N.Z.", "satra b e n z"),
            ("O.$.O.D. IV", "o s o d iv"),
            ("M.G.L.", "m g l"),
            ("DEBANDADĂ FAC", "debandada fac"),
            ("Nopți prea lungi", "nopti prea lungi"),
        )
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(normalize_str(original), expected)

    def test_sanitize_name_contracts(self) -> None:
        cases = (
            ("AC/DC", "AC_DC"),
            ("Artist: Album?", "Artist Album"),
            ("Track 01.flac.", "Track 01.flac"),
            ("", "Unknown"),
            (None, "Unknown"),
            ("Artist / Title <HQ>:", "Artist _ Title HQ"),
            ("CON", "CON_"),
            ("NUL", "NUL_"),
            ("Nimeni Altu'", "Nimeni Altu'"),
            ("M.G.L.", "M.G.L"),
            ("Satra B.E.N.Z. - O.$.O.D. IV", "Satra B.E.N.Z. - O.$.O.D. IV"),
        )
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(sanitize_name(original), expected)

    def test_is_valid_uuid(self) -> None:
        self.assertTrue(is_valid_uuid("c8b03190-306c-4125-9b32-3f9d86d60a12"))
        self.assertTrue(is_valid_uuid("C8B03190-306C-4125-9B32-3F9D86D60A12"))
        self.assertFalse(is_valid_uuid("not-a-uuid"))
        self.assertFalse(
            is_valid_uuid("c8b03190306c41259b323f9d86d60a12")
        )  # 32 chars without hyphens
        self.assertFalse(is_valid_uuid("urn:uuid:c8b03190-306c-4125-9b32-3f9d86d60a12"))
        self.assertFalse(is_valid_uuid("{c8b03190-306c-4125-9b32-3f9d86d60a12}"))
        self.assertFalse(is_valid_uuid(None))
        self.assertFalse(is_valid_uuid(""))

    def test_extract_series_number_all_formats(self) -> None:
        from sonora.core.titles import extract_series_number

        # Arabic numbers
        self.assertEqual(extract_series_number("Savage Mode 2"), 2)
        self.assertEqual(extract_series_number("Part 3"), 3)
        self.assertEqual(extract_series_number("Vol. 4"), 4)
        self.assertEqual(extract_series_number("O.$.O.D. 2"), 2)

        # Roman numerals
        self.assertEqual(extract_series_number("Savage Mode II"), 2)
        self.assertEqual(extract_series_number("Act IV"), 4)
        self.assertEqual(extract_series_number("O.$.O.D. IV"), 4)
        self.assertEqual(extract_series_number("Chapter IX"), 9)
        self.assertEqual(extract_series_number("Volume XIV"), 14)
        self.assertEqual(extract_series_number("Rapocalipsa I"), 1)
        self.assertEqual(extract_series_number("Part XX"), 20)
        self.assertEqual(extract_series_number("Part XXI"), 21)
        self.assertEqual(extract_series_number("Part XXV"), 25)
        self.assertEqual(extract_series_number("Volume XXX"), 30)
        self.assertEqual(extract_series_number("Part L"), 50)

        # Number words (1..20)
        self.assertEqual(extract_series_number("Volume One"), 1)
        self.assertEqual(extract_series_number("Part Three"), 3)
        self.assertEqual(extract_series_number("Act Seven"), 7)
        self.assertEqual(extract_series_number("Chapter Twelve"), 12)
        self.assertEqual(extract_series_number("Book Twenty"), 20)

        # None / Edge cases
        self.assertIsNone(extract_series_number("Savage Mode"))
        self.assertIsNone(extract_series_number("Live in Paris"))
        self.assertIsNone(extract_series_number(""))
        self.assertIsNone(extract_series_number(None))

    def test_clean_title_remaster_and_features(self) -> None:
        self.assertEqual(clean_title("In the End (2020 Remaster)"), "In the End")
        self.assertEqual(clean_title("Rockstar (feat. 21 Savage)"), "Rockstar")
        self.assertEqual(clean_title("Song [Explicit]"), "Song")
        self.assertEqual(clean_title("Track (Deluxe Edition)"), "Track")
        self.assertEqual(clean_title("Video [Official Music Video]"), "Video")
        self.assertEqual(clean_title("Audio (Official Audio)"), "Audio")
        self.assertEqual(clean_title("Clean [Clean Version]"), "Clean")
        self.assertEqual(clean_title("Parody [Parody]"), "Parody")
        self.assertEqual(clean_title("Spaces\u200b\u00a0Track"), "Spaces Track")
        self.assertEqual(
            clean_title("Melodie cu Vlad Dobrescu (feat. Vlad Dobrescu)"),
            "Melodie",
        )
        self.assertEqual(
            clean_title('R.I.P ROACH "EAST SIDE SOULJA"'),
            'R.I.P ROACH "EAST SIDE SOULJA"',
        )
        self.assertEqual(clean_title("Mr. Right Now feat. Drake"), "Mr. Right Now")
        self.assertEqual(clean_title("Stay With Me"), "Stay With Me")
        self.assertEqual(clean_title("Ball w/o You"), "Ball w/o You")
        self.assertEqual(
            clean_title("Solo (Wideboys remix) (feat. Demi Lovato)"),
            "Solo (Wideboys remix)",
        )
        self.assertEqual(clean_title(""), "")

    def test_extract_title_features(self) -> None:
        clean_t, feats = extract_title_features("Rockstar (feat. 21 Savage)")
        self.assertEqual(clean_t, "Rockstar")
        self.assertEqual(feats, ["21 Savage"])

        clean_t2, feats2 = extract_title_features(
            "Mr. Right Now feat. Drake", primary_artist="21 Savage"
        )
        self.assertEqual(clean_t2, "Mr. Right Now")
        self.assertEqual(feats2, ["Drake"])

        clean_t3, feats3 = extract_title_features("Stay With Me")
        self.assertEqual(clean_t3, "Stay With Me")
        self.assertEqual(feats3, [])

        clean_t4, feats4 = extract_title_features("Ball w/o You")
        self.assertEqual(clean_t4, "Ball w/o You")
        self.assertEqual(feats4, [])

        clean_t5, feats5 = extract_title_features(
            "Solo (Wideboys remix) (feat. Demi Lovato)"
        )
        self.assertEqual(clean_t5, "Solo (Wideboys remix)")
        self.assertEqual(feats5, ["Demi Lovato"])

    def test_clean_disambiguation(self) -> None:
        self.assertEqual(clean_disambiguation("Armin (ROU)"), "Armin")
        self.assertEqual(clean_disambiguation("IDK (ROU)"), "IDK")
        self.assertEqual(clean_disambiguation("Swisher (ROU)"), "Swisher")
        self.assertEqual(clean_disambiguation("Ortega (ROU)"), "Ortega")
        self.assertEqual(clean_disambiguation("Jony (10)"), "Jony")
        self.assertEqual(clean_disambiguation("Artist (USA)"), "Artist")
        self.assertEqual(clean_disambiguation("Band (UK)"), "Band")
        self.assertEqual(
            clean_disambiguation("Rafoo, BITTNER, Armin (ROU), ASSAF (ROU), RAVi"),
            "Rafoo, BITTNER, Armin, ASSAF, RAVi",
        )
        self.assertEqual(clean_disambiguation("Normal Artist"), "Normal Artist")
        self.assertEqual(clean_disambiguation(""), "")
        self.assertEqual(clean_disambiguation(None), "")

    def test_safe_parsers(self) -> None:
        self.assertEqual(safe_int(12), 12)
        self.assertEqual(safe_int("12"), 12)
        self.assertEqual(safe_int("04/10"), 4)
        self.assertIsNone(safe_int(None))
        self.assertIsNone(safe_int(""))
        self.assertIsNone(safe_int("invalid"))
        self.assertEqual(safe_float(4.5), 4.5)
        self.assertEqual(safe_float("4.5"), 4.5)
        self.assertEqual(safe_float("-2.5 dB"), -2.5)
        self.assertIsNone(safe_float(None))
        self.assertIsNone(safe_float("bad"))

    def test_clean_title_featuring_removal(self) -> None:
        self.assertEqual(
            clean_title("Melodie cu Vlad Dobrescu (feat. Vlad Dobrescu)"),
            "Melodie",
        )
        self.assertEqual(
            clean_title("Piesa cu Nane [feat. Nane]"),
            "Piesa",
        )
        self.assertEqual(
            clean_title("Dans cu Lupii (feat. Vlad Dobrescu)"),
            "Dans cu Lupii",
        )
        self.assertEqual(
            clean_title("În golul tău (feat. RAVA)"),
            "În golul tău",
        )
        self.assertEqual(
            clean_title("Capitanu' (feat. Amuly & RAVA)"),
            "Capitanu'",
        )
        self.assertEqual(
            clean_title("ZODIAC (feat. NOUA UNSPE & RAVA & BITTNER)"),
            "ZODIAC",
        )
        self.assertEqual(
            clean_title(
                "MĂ AGITĂ (feat. RAVA & Armin & Ravisval & Super ED & Armin & RAVA)"
            ),
            "MĂ AGITĂ",
        )
        self.assertEqual(
            clean_title("AX8 (feat. Lentile Blur, Ares (ROU))"),
            "AX8",
        )
        self.assertEqual(
            clean_title(
                "Undeva-n Balkani (Double L remix) (feat. Viper, Reptile (Rapper))"
            ),
            "Undeva-n Balkani (Double L remix)",
        )
        self.assertEqual(
            clean_title("Lately (feat. Bugus & Musa (KingofBabylon))"),
            "Lately",
        )
        self.assertEqual(
            clean_title("sex with my ex (og version) (feat. my ex)"),
            "sex with my ex (og version)",
        )
        self.assertEqual(
            clean_title("sex with my ex (feat. my ex)"),
            "sex with my ex",
        )
        self.assertEqual(
            clean_title("ball w/o you"),
            "ball w/o you",
        )
        self.assertEqual(
            clean_title("Track (w/o Drums)"),
            "Track (w/o Drums)",
        )
        self.assertEqual(
            clean_title("Song (w/ Drake)"),
            "Song",
        )
        self.assertEqual(clean_title(""), "")
        self.assertEqual(clean_title(None), "")

    def test_extract_title_features_with_user_aliases(self) -> None:
        alias_path = Path.home() / ".config" / "sonora" / "aliases.json"
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        alias_path.write_text('{"RAVi": "Ravisval"}', encoding="utf-8")
        clean_t, feats = extract_title_features("SEMAKA (feat. RAVi & Armin)")
        self.assertEqual(clean_t, "SEMAKA")
        self.assertEqual(feats, ["Ravisval", "Armin"])

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
        with patch("sonora.core.utils.is_single_group_artist", return_value=True):
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

    def test_safe_int_pair(self) -> None:
        self.assertEqual(safe_int_pair("1/2"), (1, 2))
        self.assertEqual(safe_int_pair("01/12"), (1, 12))
        self.assertEqual(safe_int_pair("2"), (2, None))
        self.assertEqual(safe_int_pair(1), (1, None))
        self.assertEqual(safe_int_pair(None), (None, None))
        self.assertEqual(safe_int_pair(""), (None, None))
        self.assertEqual(safe_int_pair("invalid/val"), (None, None))

    def test_parse_track_filename(self) -> None:
        self.assertEqual(parse_track_filename("1-01 - Title.flac"), (1, 1, "Title"))
        self.assertEqual(
            parse_track_filename("2-12 - Song Name.mp3"), (2, 12, "Song Name")
        )
        self.assertEqual(
            parse_track_filename("CD 1 - 01 - Title.flac"), (1, 1, "Title")
        )
        self.assertEqual(
            parse_track_filename("Disc 02 - 05 - Great Track.flac"),
            (2, 5, "Great Track"),
        )
        self.assertEqual(parse_track_filename("01 - Title.flac"), (None, 1, "Title"))
        self.assertEqual(parse_track_filename("Title.flac"), (None, None, "Title"))

    def test_extract_disc_number_from_folder(self) -> None:
        self.assertEqual(extract_disc_number_from_folder("CD 1"), 1)
        self.assertEqual(extract_disc_number_from_folder("Disc 2"), 2)
        self.assertEqual(extract_disc_number_from_folder("Disc 03"), 3)
        self.assertEqual(extract_disc_number_from_folder("Side A"), 1)
        self.assertEqual(extract_disc_number_from_folder("Side B"), 2)
        self.assertIsNone(extract_disc_number_from_folder("Regular Album"))

    def test_group_files_by_album_root(self) -> None:
        file1 = Path("/library/Artist - Album/CD 1/01 - Song.flac")
        file2 = Path("/library/Artist - Album/CD 2/01 - Song.flac")
        file3 = Path("/library/Other Artist - Other Album/01 - Track.flac")
        grouped = group_files_by_album_root([file1, file2, file3])

        album_key = Path("/library/Artist - Album")
        other_key = Path("/library/Other Artist - Other Album")

        self.assertIn(album_key, grouped)
        self.assertEqual(grouped[album_key], [file1, file2])
        self.assertIn(other_key, grouped)
        self.assertEqual(grouped[other_key], [file3])

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

    def test_normalize_genre_mapping_and_filtering(self) -> None:
        self.assertEqual(normalize_genre("Hip Hop"), "Hip-Hop/Rap")
        self.assertEqual(normalize_genre("Rap"), "Hip-Hop/Rap")
        self.assertEqual(normalize_genre("Trap"), "Hip-Hop/Rap")
        self.assertEqual(normalize_genre("Rap/Hip Hop"), "Hip-Hop/Rap")
        self.assertEqual(normalize_genre("Pop Rap"), "Hip-Hop/Rap")
        self.assertEqual(normalize_genre("Alternativă"), "Alternative")
        self.assertEqual(normalize_genre("Electronic"), "Electronic")
        self.assertEqual(normalize_genre("Electronica"), "Electronic")
        self.assertEqual(normalize_genre("Euro House"), "House")
        self.assertEqual(normalize_genre("Contemporary R&B"), "R&B/Soul")
        self.assertEqual(normalize_genre("French Pop"), "Pop")
        self.assertEqual(normalize_genre("Synthpop"), "Synth-pop")
        self.assertIsNone(normalize_genre("Billboard Top 40"))  # Blacklisted
        self.assertIsNone(normalize_genre("Unknown"))  # Blacklisted
        self.assertIsNone(normalize_genre("Fitness & Workout"))  # Blacklisted
        self.assertIsNone(normalize_genre("12345"))  # Digits
        self.assertIsNone(normalize_genre(""))
        self.assertIsNone(normalize_genre(None))

    def test_clean_unicode_punct(self) -> None:
        self.assertEqual(clean_unicode_punct("Connect‐R"), "Connect-R")
        self.assertEqual(clean_unicode_punct("K‐POP"), "K-POP")
        self.assertEqual(clean_unicode_punct("Saint‐Tropez"), "Saint-Tropez")
        self.assertEqual(clean_unicode_punct("Făt‐Frumos"), "Făt-Frumos")
        self.assertEqual(clean_unicode_punct("\u200bits me"), "its me")
        self.assertEqual(clean_unicode_punct(""), "")
        self.assertEqual(clean_unicode_punct(None), "")

    def test_normalize_date(self) -> None:
        self.assertEqual(normalize_date("2024-07-12"), "2024-07-12")
        self.assertEqual(normalize_date("2003"), "2003")
        self.assertIsNone(normalize_date("0"))
        self.assertIsNone(normalize_date("0000"))
        self.assertIsNone(normalize_date("None"))
        self.assertIsNone(normalize_date("null"))
        self.assertIsNone(normalize_date(""))
        self.assertIsNone(normalize_date(None))
        self.assertIsNone(normalize_date("1899"))
        self.assertIsNone(normalize_date("2099"))

    @patch("sonora.core.artists.get_cached_api", return_value=None)
    def test_resolve_artist_name_exact_match(self, mock_get_cache: MagicMock) -> None:
        from sonora.core.artists import resolve_artist_name

        with patch("sonora.services.musicbrainz.search_musicbrainz_artists") as mock_mb:
            # Simulate MusicBrainz returning Enrico Rava with score 100 and RAVA with score 90
            mock_mb.return_value = [
                {"name": "Enrico Rava", "ext:score": "100", "id": "uuid-1"},
                {"name": "RAVA", "ext:score": "90", "id": "uuid-2"},
            ]
            # Must prioritize exact match RAVA over higher-scored Enrico Rava
            self.assertEqual(resolve_artist_name("RAVA"), "RAVA")
            self.assertEqual(resolve_artist_name("rava"), "RAVA")

    @patch("sonora.core.artists.get_cached_api", return_value=None)
    @patch("sonora.core.artists.set_cached_api")
    def test_resolve_artist_name_preserves_titlecase_over_allcaps(
        self, mock_set_cache: MagicMock, mock_get_cache: MagicMock
    ) -> None:
        from sonora.core.artists import resolve_artist_name

        resolve_artist_name.cache_clear()
        with patch("sonora.services.musicbrainz.search_musicbrainz_artists") as mock_mb:
            # MusicBrainz returns CRUSH (Group, score 100)
            mock_mb.return_value = [
                {"name": "CRUSH", "type": "Group", "ext:score": "100", "id": "uuid-1"},
            ]
            # TitleCase "Crush" must not be converted to screaming "CRUSH"
            self.assertEqual(resolve_artist_name("Crush"), "Crush")

    def test_resolve_artist_name_heals_corrupted_cached_expansion(self) -> None:
        from sonora.core.artists import resolve_artist_name

        resolve_artist_name.cache_clear()
        with (
            patch("sonora.core.artists.get_cached_api", return_value="Gruppe Drei"),
            patch("sonora.core.artists.set_cached_api") as mock_set_cache,
        ):
            # When cache has corrupt multi-word expansion "Gruppe Drei" for single word "Drei"
            # It must purge cache and return "Drei" (when network is disabled)
            resolved_artist_name = resolve_artist_name("Drei", allow_network=False)
            mock_set_cache.assert_not_called()
            self.assertEqual(resolved_artist_name, "Drei")

    @patch("sonora.core.artists.get_cached_api", return_value=None)
    def test_resolve_artist_name_allow_network_false(
        self, mock_get_cache: MagicMock
    ) -> None:
        from sonora.core.artists import get_primary_artist, resolve_artist_name

        resolve_artist_name.cache_clear()
        with patch("sonora.services.musicbrainz.search_musicbrainz_artists") as mock_mb:
            # When allow_network is False, MusicBrainz search must NEVER be called
            result = resolve_artist_name("Unknown Indierock Band", allow_network=False)
            self.assertEqual(result, "Unknown Indierock Band")
            mock_mb.assert_not_called()

            # Similarly, get_primary_artist must never invoke network
            primary = get_primary_artist("New Artist feat. Guest Singer")
            self.assertEqual(primary, "New Artist")
            mock_mb.assert_not_called()

    def test_match_score_series_and_version_disambiguation(self) -> None:

        # Series volumes must never match each other
        self.assertEqual(
            score_artist_title_match(
                "B.U.G. Mafia",
                "Viața noastră vol. 1",
                "B.U.G. Mafia",
                "Viața noastră vol. 2",
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "Chase Atlantic", "Part One", "Chase Atlantic", "Part Two"
            ),
            0.0,
        )

        # Versions vs originals
        self.assertEqual(
            score_artist_title_match(
                "Deliric",
                "Deliric X Silent Strike (Instrumentals)",
                "Deliric",
                "Deliric X Silent Strike",
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "Dennis Lloyd",
                "Alien (Acoustic)",
                "Dennis Lloyd",
                "Alien (Topic Remix)",
            ),
            0.0,
        )
        self.assertGreaterEqual(
            score_artist_title_match(
                "Dennis Lloyd", "Alien (Acoustic)", "Dennis Lloyd", "Alien Acoustic"
            ),
            90.0,
        )
        self.assertGreaterEqual(
            score_artist_title_match(
                "B.U.G. Mafia",
                "Viața noastră vol. 1",
                "B.U.G. Mafia",
                "Viața Noastră, Vol. 1",
            ),
            90.0,
        )

    @patch("sonora.services.musicbrainz.search_musicbrainz_artists")
    def test_is_single_group_artist(self, mock_search: MagicMock) -> None:
        is_single_group_artist.cache_clear()
        mock_search.return_value = [
            {
                "name": "Above & Beyond",
                "type": "Group",
                "ext:score": "100",
            }
        ]
        self.assertTrue(is_single_group_artist("Above & Beyond"))

        is_single_group_artist.cache_clear()
        mock_search.return_value = [
            {"name": "Drake", "type": "Person", "ext:score": "80"}
        ]
        self.assertFalse(is_single_group_artist("Drake & 21 Savage"))
        self.assertFalse(is_single_group_artist("SingleArtist"))
        self.assertFalse(is_single_group_artist(""))
        self.assertFalse(is_single_group_artist(None))

    @patch("sonora.services.musicbrainz.search_musicbrainz_artists")
    def test_get_primary_artist(self, mock_search: MagicMock) -> None:
        is_single_group_artist.cache_clear()
        mock_search.return_value = [
            {
                "name": "Above & Beyond",
                "type": "Group",
                "ext:score": "100",
            }
        ]
        self.assertEqual(
            get_primary_artist("Above & Beyond", allow_network=True),
            "Above & Beyond",
        )

        is_single_group_artist.cache_clear()
        mock_search.return_value = [
            {
                "name": "Alan & Kepa",
                "type": "Group",
                "ext:score": "100",
            }
        ]
        self.assertEqual(
            get_primary_artist("Alan & Kepa", allow_network=True), "Alan & Kepa"
        )

        is_single_group_artist.cache_clear()
        mock_search.return_value = [
            {
                "name": "Play & Win",
                "type": "Group",
                "ext:score": "100",
            }
        ]
        self.assertEqual(
            get_primary_artist("Play & Win", allow_network=True), "Play & Win"
        )
        self.assertEqual(
            get_primary_artist("Play&Win", allow_network=True), "Play & Win"
        )

        is_single_group_artist.cache_clear()
        mock_search.return_value = [
            {
                "name": "Simon & Garfunkel",
                "type": "Group",
                "ext:score": "100",
            }
        ]
        self.assertEqual(
            get_primary_artist("Simon & Garfunkel", allow_network=True),
            "Simon & Garfunkel",
        )

        is_single_group_artist.cache_clear()
        mock_search.return_value = []
        self.assertEqual(
            get_primary_artist("21 Savage feat. Metro Boomin"), "21 Savage"
        )
        self.assertEqual(get_primary_artist("Drake & Future"), "Drake")
        self.assertEqual(get_primary_artist("Samurai & El Nino"), "Samurai")
        self.assertEqual(get_primary_artist(""), "Unknown")
        self.assertEqual(get_primary_artist(None), "Unknown")

    def test_group_files_by_parent(self) -> None:
        f1 = Path("/tmp/album1/01.flac")
        f2 = Path("/tmp/album1/02.flac")
        f3 = Path("/tmp/album2/01.flac")
        grouped = group_files_by_parent([f1, f2, f3])
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[Path("/tmp/album1")], [f1, f2])
        self.assertEqual(grouped[Path("/tmp/album2")], [f3])

    def test_safe_case_rename_and_relocate_companion_lyrics(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audio = tmp_path / "Song.flac"
            audio.write_bytes(b"audio")
            lrc = tmp_path / "Song.lrc"
            lrc.write_text("[00:01.00] test", encoding="utf-8")

            target_audio = tmp_path / "01 - Song.flac"
            safe_case_rename(audio, target_audio)
            self.assertTrue(target_audio.exists())
            self.assertFalse(audio.exists())

            moved_lyrics = relocate_companion_lyrics(audio, target_audio)
            self.assertEqual(len(moved_lyrics), 1)
            target_lrc = tmp_path / "01 - Song.lrc"
            self.assertTrue(target_lrc.exists())
            self.assertFalse(lrc.exists())
