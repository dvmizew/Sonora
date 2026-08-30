import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest
from unittest.mock import MagicMock, patch

from sonora.core.models import TrackInfo
from sonora.core.utils import (
    clean_disambiguation,
    clean_title,
    deduplicate_title_features,
    get_primary_artist,
    group_files_by_parent,
    is_single_group_artist,
    is_valid_uuid,
    normalize_genre,
    normalize_str,
    relocate_companion_lyrics,
    safe_case_rename,
    sanitize_name,
)


class TestCoreUtils(unittest.TestCase):
    def test_normalize_str_basic(self):
        self.assertEqual(normalize_str("Hello World"), "hello world")
        self.assertEqual(normalize_str("$tring!"), "string")

    def test_normalize_str_diacritics_and_symbols(self):
        self.assertEqual(normalize_str("Beyoncé"), "beyonce")
        self.assertEqual(normalize_str("Mötley Crüe"), "motley crue")

    def test_sanitize_name_filesystem_chars(self):
        self.assertEqual(sanitize_name("AC/DC"), "AC_DC")
        self.assertEqual(sanitize_name("Artist: Album?"), "Artist Album")
        self.assertEqual(sanitize_name("Track 01.flac."), "Track 01.flac")

    def test_sanitize_name_empty(self):
        self.assertEqual(sanitize_name(""), "Unknown")
        self.assertEqual(sanitize_name(None), "Unknown")

    def test_normalize_str_edge_cases(self):
        self.assertEqual(normalize_str(None), "")
        self.assertEqual(normalize_str(""), "")
        self.assertEqual(normalize_str("   "), "")
        self.assertEqual(normalize_str("A$AP Rocky!"), "asap rocky")
        self.assertEqual(normalize_str("Beyoncé - Nöél"), "beyonce noel")
        self.assertEqual(normalize_str("Nimeni Altu'"), "nimeni altu")
        self.assertEqual(normalize_str("Satra B.E.N.Z."), "satra b e n z")
        self.assertEqual(normalize_str("O.$.O.D. IV"), "o s o d iv")
        self.assertEqual(normalize_str("M.G.L."), "m g l")
        self.assertEqual(normalize_str("DEBANDADĂ FAC"), "debandada fac")
        self.assertEqual(normalize_str("Nopți prea lungi"), "nopti prea lungi")

    def test_sanitize_name_complex_edge_cases(self):
        self.assertEqual(sanitize_name("Artist / Title <HQ>:"), "Artist _ Title HQ")
        self.assertEqual(sanitize_name("CON"), "CON_")
        self.assertEqual(sanitize_name("NUL"), "NUL_")
        self.assertEqual(sanitize_name("Nimeni Altu'"), "Nimeni Altu'")
        self.assertEqual(sanitize_name("M.G.L."), "M.G.L")
        self.assertEqual(
            sanitize_name("Satra B.E.N.Z. - O.$.O.D. IV"),
            "Satra B.E.N.Z. - O.$.O.D. IV",
        )

    def test_is_valid_uuid(self):
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

    def test_extract_series_number_all_formats(self):
        from sonora.core.utils import extract_series_number

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

    def test_clean_title_remaster_and_features(self):
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
        self.assertEqual(clean_title(""), "")

    def test_clean_disambiguation(self):
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

    def test_deduplicate_title_features(self):
        self.assertEqual(
            deduplicate_title_features(
                "Melodie cu Vlad Dobrescu (feat. Vlad Dobrescu)"
            ),
            "Melodie (feat. Vlad Dobrescu)",
        )
        self.assertEqual(
            deduplicate_title_features("Piesa cu Nane [feat. Nane]"),
            "Piesa [feat. Nane]",
        )
        self.assertEqual(
            deduplicate_title_features("Dans cu Lupii (feat. Vlad Dobrescu)"),
            "Dans cu Lupii (feat. Vlad Dobrescu)",
        )
        self.assertEqual(
            deduplicate_title_features("În golul tău (feat. RAVA)"),
            "În golul tău (feat. RAVA)",
        )
        self.assertEqual(
            deduplicate_title_features(
                "Capitanu' (feat. Amuly & RAVA)", primary_artist="Tussin"
            ),
            "Capitanu' (feat. Amuly & RAVA)",
        )
        self.assertEqual(
            deduplicate_title_features(
                "ZODIAC (feat. NOUA UNSPE & RAVA & BITTNER)",
                primary_artist="NOUA UNSPE",
            ),
            "ZODIAC (feat. RAVA & BITTNER)",
        )
        self.assertEqual(
            deduplicate_title_features(
                "MĂ AGITĂ (feat. RAVA & Armin & Ravisval & Super ED & Armin & RAVA)",
                primary_artist="4 226",
            ),
            "MĂ AGITĂ (feat. RAVA, Armin, Ravisval & Super ED)",
        )
        self.assertEqual(deduplicate_title_features(""), "")
        self.assertEqual(deduplicate_title_features(None), "")

    def test_deduplicate_title_features_with_user_aliases(self):
        with patch(
            "sonora.core.utils._load_user_overrides",
            return_value={"ravi": "Ravisval"},
        ):
            self.assertEqual(
                deduplicate_title_features("SEMAKA (feat. RAVi & Armin)"),
                "SEMAKA (feat. Ravisval & Armin)",
            )

    def test_load_user_overrides_corrupt_json(self):
        from sonora.core.utils import _load_user_overrides

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

    def test_normalize_genre_mapping_and_filtering(self):
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

    def test_resolve_artist_name_exact_match(self):
        from sonora.core.utils import resolve_artist_name

        with patch("sonora.core.utils.musicbrainzngs.search_artists") as mock_mb:
            # Simulate MusicBrainz returning Enrico Rava with score 100 and RAVA with score 90
            mock_mb.return_value = {
                "artist-list": [
                    {"name": "Enrico Rava", "ext:score": "100", "id": "uuid-1"},
                    {"name": "RAVA", "ext:score": "90", "id": "uuid-2"},
                ]
            }
            # Must prioritize exact match RAVA over higher-scored Enrico Rava
            self.assertEqual(resolve_artist_name("RAVA"), "RAVA")
            self.assertEqual(resolve_artist_name("rava"), "RAVA")

    def test_match_score_series_and_version_disambiguation(self):
        from sonora.core.utils import match_score

        # Series volumes must never match each other
        self.assertEqual(
            match_score(
                "B.U.G. Mafia",
                "Viața noastră vol. 1",
                "B.U.G. Mafia",
                "Viața noastră vol. 2",
            ),
            0.0,
        )
        self.assertEqual(
            match_score("Chase Atlantic", "Part One", "Chase Atlantic", "Part Two"),
            0.0,
        )

        # Versions vs originals
        self.assertEqual(
            match_score(
                "Deliric",
                "Deliric X Silent Strike (Instrumentals)",
                "Deliric",
                "Deliric X Silent Strike",
            ),
            0.0,
        )
        self.assertEqual(
            match_score(
                "Dennis Lloyd",
                "Alien (Acoustic)",
                "Dennis Lloyd",
                "Alien (Topic Remix)",
            ),
            0.0,
        )
        self.assertGreaterEqual(
            match_score(
                "Dennis Lloyd", "Alien (Acoustic)", "Dennis Lloyd", "Alien Acoustic"
            ),
            90.0,
        )
        self.assertGreaterEqual(
            match_score(
                "B.U.G. Mafia",
                "Viața noastră vol. 1",
                "B.U.G. Mafia",
                "Viața Noastră, Vol. 1",
            ),
            90.0,
        )

    @patch("sonora.core.utils.musicbrainzngs.search_artists")
    def test_is_single_group_artist(self, mock_search):
        is_single_group_artist.cache_clear()
        mock_search.return_value = {
            "artist-list": [
                {
                    "name": "Above & Beyond",
                    "type": "Group",
                    "ext:score": "100",
                }
            ]
        }
        self.assertTrue(is_single_group_artist("Above & Beyond"))

        is_single_group_artist.cache_clear()
        mock_search.return_value = {
            "artist-list": [{"name": "Drake", "type": "Person", "ext:score": "80"}]
        }
        self.assertFalse(is_single_group_artist("Drake & 21 Savage"))
        self.assertFalse(is_single_group_artist("SingleArtist"))
        self.assertFalse(is_single_group_artist(""))
        self.assertFalse(is_single_group_artist(None))

    @patch("sonora.core.utils.musicbrainzngs.search_artists")
    def test_get_primary_artist(self, mock_search):
        is_single_group_artist.cache_clear()
        mock_search.return_value = {
            "artist-list": [
                {
                    "name": "Above & Beyond",
                    "type": "Group",
                    "ext:score": "100",
                }
            ]
        }
        self.assertEqual(get_primary_artist("Above & Beyond"), "Above & Beyond")

        is_single_group_artist.cache_clear()
        mock_search.return_value = {
            "artist-list": [
                {
                    "name": "Alan & Kepa",
                    "type": "Group",
                    "ext:score": "100",
                }
            ]
        }
        self.assertEqual(get_primary_artist("Alan & Kepa"), "Alan & Kepa")

        is_single_group_artist.cache_clear()
        mock_search.return_value = {
            "artist-list": [
                {
                    "name": "Play & Win",
                    "type": "Group",
                    "ext:score": "100",
                }
            ]
        }
        self.assertEqual(get_primary_artist("Play & Win"), "Play & Win")
        self.assertEqual(get_primary_artist("Play&Win"), "Play & Win")

        is_single_group_artist.cache_clear()
        mock_search.return_value = {
            "artist-list": [
                {
                    "name": "Simon & Garfunkel",
                    "type": "Group",
                    "ext:score": "100",
                }
            ]
        }
        self.assertEqual(get_primary_artist("Simon & Garfunkel"), "Simon & Garfunkel")

        is_single_group_artist.cache_clear()
        mock_search.return_value = {"artist-list": []}
        self.assertEqual(
            get_primary_artist("21 Savage feat. Metro Boomin"), "21 Savage"
        )
        self.assertEqual(get_primary_artist("Drake & Future"), "Drake")
        self.assertEqual(get_primary_artist("Samurai & El Nino"), "Samurai")
        self.assertEqual(get_primary_artist(""), "Unknown")
        self.assertEqual(get_primary_artist(None), "Unknown")

    def test_group_files_by_parent(self):
        f1 = Path("/tmp/album1/01.flac")
        f2 = Path("/tmp/album1/02.flac")
        f3 = Path("/tmp/album2/01.flac")
        grouped = group_files_by_parent([f1, f2, f3])
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[Path("/tmp/album1")], [f1, f2])
        self.assertEqual(grouped[Path("/tmp/album2")], [f3])

    def test_safe_case_rename_and_relocate_companion_lyrics(self):
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


class TestCoreModels(unittest.TestCase):
    def test_track_info_serialization(self):
        track = TrackInfo(
            file_path=Path("/music/song.flac"),
            artist="Beyoncé",
            title="HALO",
            album="I Am... Sasha Fierce",
            track_number=1,
            bpm=120.0,
        )
        data = track.to_dict()
        self.assertEqual(data["artist"], "Beyoncé")
        self.assertEqual(data["title"], "HALO")
        self.assertEqual(data["track_number"], 1)
        self.assertEqual(data["bpm"], 120.0)

        empty_track = TrackInfo(file_path=Path("/music/empty.flac"))
        empty_data = empty_track.to_dict()
        self.assertEqual(empty_data["artist"], "Unknown Artist")
        self.assertIsNone(empty_data["track_number"])
        self.assertIsNone(empty_data["bpm"])


if __name__ == "__main__":
    unittest.main()
