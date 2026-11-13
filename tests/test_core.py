import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx

from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core import (
    audio_files,
    library_paths,
    sidecars,
)
from sonora.core.audio_files import group_files_by_parent
from sonora.core.cancellation import InterruptedOperationError, is_interruption
from sonora.core.config import (
    SonoraConfig,
    _parse_config_data,
    clear_config_cache,
    get_artist_split_pattern,
    get_balanced_feat_pattern,
    get_bracket_feat_pattern,
    get_config,
    get_disambiguation_pattern,
    get_duplicate_feat_pattern,
    get_feat_tokens_pattern,
)
from sonora.core.constants import RATE_LIMIT_DISCOGS_UNAUTHENTICATED
from sonora.core.http import RetryTransport, _retry_delay
from sonora.core.library_paths import (
    extract_disc_number_from_folder,
    get_single_release_title,
    group_files_by_album_root,
    is_in_singles_hierarchy,
    parse_track_filename,
    safe_case_rename,
)
from sonora.core.models import TrackInfo
from sonora.core.rate_limit import RateLimiter
from sonora.core.retry import retry_network_call
from sonora.core.sidecars import relocate_companion_lyrics
from sonora.core.text import (
    clean_disambiguation,
    clean_unicode_punct,
    is_valid_uuid,
    normalize_country_name,
    normalize_date,
    normalize_genre,
    normalize_language_name,
    normalize_legacy_diacritics,
    normalize_script_name,
    normalize_str,
    preserve_unicode_repertoire,
    safe_float,
    safe_int,
    safe_int_pair,
    sanitize_name,
)
from sonora.core.titles import (
    clean_title,
    extract_title_features,
    extract_version_modifier,
    score_artist_title_match,
)
from sonora.modules.backup import backup_library_tags, restore_library_tags
from sonora.modules.checker import check_library
from sonora.modules.organizer import organize_library_singles
from sonora.modules.renamer import rename_directory_files
from sonora.modules.tagger import tag_album_folder
from sonora.modules.track_processing import normalize_library
from tests.fixtures import write_silent_wav
from tests.module_support import OfflinePipelineTestCase
from tests.support import IsolatedTestCase


class TestCoreModels(IsolatedTestCase):
    def test_track_info_serialization(self) -> None:
        track = TrackInfo(
            file_path=Path("/music/song.flac"),
            artist="Beyoncé",
            title="HALO",
            album="I Am... Sasha Fierce",
            track_number=1,
            bpm=120.0,
        )
        serialized_track = track.to_dict()
        self.assertEqual(serialized_track["artist"], "Beyoncé")
        self.assertEqual(serialized_track["title"], "HALO")
        self.assertEqual(serialized_track["track_number"], 1)
        self.assertEqual(serialized_track["bpm"], 120.0)

        empty_track = TrackInfo(file_path=Path("/music/empty.flac"))
        empty_data = empty_track.to_dict()
        self.assertEqual(empty_data["artist"], "Unknown Artist")
        self.assertIsNone(empty_data["track_number"])
        self.assertIsNone(empty_data["bpm"])


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

    def test_edition_markers_do_not_strip_remixer_names(self) -> None:
        for version in (
            "Monoir Remix",
            "Monoir Remix Radio Edit",
            "Monolink Remix",
            "Stereophonics Remix",
        ):
            title = f"I Loved You ({version})"
            with self.subTest(version=version):
                self.assertEqual(clean_title(title), title)
        self.assertEqual(clean_title("Song (Mono Version)"), "Song")
        self.assertEqual(clean_title("Song (Stereo Mix)"), "Song")

    def test_identical_recording_titles_ignore_remaster_capitalization(self) -> None:
        self.assertGreater(
            score_artist_title_match(
                "Creed",
                "What If (remastered 2024)",
                "Creed",
                "What If (Remastered 2024)",
            ),
            0,
        )

    def test_dotted_initialisms_match_without_merging_unrelated_artist_names(
        self,
    ) -> None:
        for artist, candidate in (
            ("C.T.C.", "CTC"),
            ("CTC", "C.T.C."),
            ("M.G.L.", "MGL"),
        ):
            with self.subTest(artist=artist, candidate=candidate):
                self.assertGreater(
                    score_artist_title_match(artist, "Intro", candidate, "Intro"), 0
                )
        for artist, candidate in (("C.T.C.", "CTL"),):
            with self.subTest(artist=artist, candidate=candidate):
                self.assertEqual(
                    score_artist_title_match(artist, "Intro", candidate, "Intro"), 0
                )

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
        for cleaner in (clean_unicode_punct, clean_title):
            with self.subTest(cleaner=cleaner.__name__):
                self.assertEqual(cleaner("MAI TII MINTE\u2060?"), "MAI TII MINTE?")
                self.assertEqual(cleaner("می\u200cروم"), "می\u200cروم")
                self.assertEqual(cleaner("👩\u200d🎤"), "👩\u200d🎤")
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

    def test_group_files_by_parent(self) -> None:
        f1 = Path("/tmp/album1/01.flac")
        f2 = Path("/tmp/album1/02.flac")
        f3 = Path("/tmp/album2/01.flac")
        grouped = group_files_by_parent([f1, f2, f3])
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[Path("/tmp/album1")], [f1, f2])
        self.assertEqual(grouped[Path("/tmp/album2")], [f3])

    def test_safe_case_rename_and_relocate_companion_lyrics(self) -> None:
        audio = self.test_directory / "Song.flac"
        audio.write_bytes(b"audio")
        lrc = self.test_directory / "Song.lrc"
        lrc.write_text("[00:01.00] test", encoding="utf-8")

        target_audio = self.test_directory / "01 - Song.flac"
        safe_case_rename(audio, target_audio)
        self.assertTrue(target_audio.exists())
        self.assertFalse(audio.exists())

        moved_lyrics = relocate_companion_lyrics(audio, target_audio)
        self.assertEqual(len(moved_lyrics), 1)
        target_lrc = self.test_directory / "01 - Song.lrc"
        self.assertTrue(target_lrc.exists())
        self.assertFalse(lrc.exists())

    def test_numeric_tags_reject_booleans_nondecimal_digits_and_overflow(self) -> None:
        for value in (None, True, False, "²", "not a number"):
            with self.subTest(value=value):
                self.assertIsNone(safe_int(value))
                self.assertEqual(safe_int_pair(value), (None, None))
                self.assertIsNone(safe_float(value))
        self.assertIsNone(safe_float(10**1000))
        self.assertEqual(safe_int("٣"), 3)
        self.assertEqual(safe_int_pair("03/12"), (3, 12))
        self.assertEqual(safe_float("-7.5 dB"), -7.5)

    def test_configured_disc_folders_without_numbers_have_no_invented_index(
        self,
    ) -> None:
        with patch.object(
            library_paths,
            "get_config",
            return_value=SonoraConfig(disc_folder_patterns=("Bonus",)),
        ):
            self.assertIsNone(library_paths.extract_disc_number_from_folder("Bonus"))

    def test_hard_link_aliases_are_safe_rename_targets(self) -> None:
        source = self.test_directory / "recording.flac"
        source.write_bytes(b"recording")
        alternate = self.test_directory / "alternate.flac"
        alternate.hardlink_to(source)
        self.assertEqual(library_paths.safe_case_rename(source, alternate), alternate)
        self.assertEqual(alternate.read_bytes(), b"recording")
        other_directory = self.test_directory / "other"
        other_directory.mkdir()
        same_name = other_directory / source.name
        same_name.hardlink_to(source)
        self.assertEqual(library_paths.safe_case_rename(source, same_name), same_name)
        self.assertEqual(same_name.read_bytes(), b"recording")

    def test_companion_lyrics_can_be_relocated_in_a_preview(self) -> None:
        source = self.test_directory / "recording.flac"
        lyric = source.with_suffix(".lrc")
        lyric.write_text("lyrics")
        destination = self.test_directory / "renamed.flac"
        self.assertEqual(
            sidecars.relocate_companion_lyrics(source, destination, dry_run=True),
            [destination.with_suffix(".lrc")],
        )
        self.assertEqual(lyric.read_text(), "lyrics")

    def test_missing_directory_and_companion_lyric_search_fallbacks(self) -> None:
        self.assertEqual(
            audio_files.find_audio_files(self.test_directory / "absent"), []
        )
        audio = self.test_directory / "original.flac"
        lyric = self.test_directory / "02 - Different Name.lrc"
        lyric.touch()
        self.assertEqual(sidecars.find_companion_lyrics(audio, track_number=2), [lyric])
        self.assertEqual(
            sidecars.find_companion_lyrics(
                audio, track_number="invalid", in_singles=True
            ),
            [lyric],
        )
        (self.test_directory / "other.lrc").touch()
        self.assertEqual(sidecars.find_companion_lyrics(audio, in_singles=True), [])
        with patch.object(
            Path, "iterdir", side_effect=PermissionError("locked directory")
        ):
            self.assertEqual(
                sidecars.find_companion_lyrics(audio, track_number=2, in_singles=True),
                [],
            )

    def test_artwork_moves_preserve_existing_destinations_and_dry_run(self) -> None:
        source = self.test_directory / "source"
        target = self.test_directory / "target"
        source.mkdir()
        target.mkdir()
        cover = source / "cover.jpg"
        cover.write_bytes(b"artwork")
        self.assertEqual(sidecars.relocate_companion_artwork(source, source), [])
        self.assertEqual(
            sidecars.relocate_companion_artwork(source, target, dry_run=True),
            [target / "cover.jpg"],
        )
        self.assertTrue(cover.exists())
        with patch.object(
            sidecars,
            "safe_case_rename",
            side_effect=PermissionError("destination locked"),
        ):
            self.assertEqual(sidecars.relocate_companion_artwork(source, target), [])
        target_cover = target / "cover.jpg"
        path_exists = Path.exists

        def create_destination(path: Path) -> bool:
            if path == target_cover and not path_exists(path):
                path.write_bytes(b"existing artwork")
                return False
            return path_exists(path)

        with patch.object(Path, "exists", create_destination):
            self.assertEqual(sidecars.relocate_companion_artwork(source, target), [])
        self.assertEqual(cover.read_bytes(), b"artwork")
        self.assertEqual(target_cover.read_bytes(), b"existing artwork")
        target_cover.unlink()
        self.assertEqual(
            sidecars.relocate_companion_artwork(source, target),
            [target / "cover.jpg"],
        )
        self.assertEqual((target / "cover.jpg").read_bytes(), b"artwork")

    def test_safe_rename_handles_identity_permissions_and_collision(self) -> None:
        source = self.test_directory / "source.flac"
        source.write_bytes(b"recording")
        self.assertEqual(library_paths.safe_case_rename(source, source), source)
        with (
            patch.object(os, "access", return_value=False),
            self.assertRaises(PermissionError),
        ):
            library_paths.safe_case_rename(source, self.test_directory / "target.flac")
        destination = self.test_directory / "destination.flac"
        destination.touch()
        with self.assertRaises(FileExistsError):
            library_paths.safe_case_rename(source, destination)
        folder = self.test_directory / "folder"
        folder.mkdir()
        renamed = folder.with_name("renamed")
        self.assertEqual(library_paths.safe_case_rename(folder, renamed), renamed)
        self.assertTrue(renamed.is_dir())

    def test_missing_companions_and_existing_lyrics_do_not_overwrite(self) -> None:
        source = self.test_directory / "source.flac"
        destination = self.test_directory / "destination.flac"
        source_lrc = source.with_suffix(".lrc")
        destination_lrc = destination.with_suffix(".lrc")
        source_lrc.write_text("original")
        destination_lrc.write_text("destination")
        self.assertEqual(
            sidecars.relocate_companion_lyrics(
                source,
                destination,
                sidecars=[source_lrc, self.test_directory / "absent.lrc"],
            ),
            [],
        )
        self.assertEqual(destination_lrc.read_text(), "destination")


class TestCoreConfig(IsolatedTestCase):
    def test_credentials_keep_config_alias_and_environment_precedence(self) -> None:
        for credential, alias, environment_names in (
            (
                "discogs_token",
                "discogs_user_token",
                ("DISCOGS_TOKEN", "DISCOGS_USER_TOKEN"),
            ),
            ("acoustid_api_key", "acoustid_key", ("ACOUSTID_API_KEY",)),
            ("genius_api_token", "genius_token", ("GENIUS_API_TOKEN", "GENIUS_TOKEN")),
            ("lastfm_api_key", "lastfm_key", ("LASTFM_API_KEY",)),
            (
                "musixmatch_token",
                "musixmatch_token",
                ("MUSIXMATCH_TOKEN", "MUSIXMATCH_USER_TOKEN"),
            ),
            ("fanart_api_key", "fanart_key", ("FANART_API_KEY", "FANART_KEY")),
            ("fanart_client_key", "fanart_client", ("FANART_CLIENT_KEY",)),
        ):
            for configured, expected in (
                (
                    {credential: " canonical ", alias: " alias "}
                    if credential != alias
                    else {credential: " canonical "},
                    "canonical",
                ),
                ({alias: " alias "}, "alias"),
                ({}, "environment"),
            ):
                with (
                    self.subTest(credential=credential, configured=configured),
                    patch("sonora.core.config.load_app_environment"),
                    patch(
                        "sonora.core.config._discover_config_file",
                        return_value=Path("config.json"),
                    ),
                    patch(
                        "sonora.core.config._parse_config_data", return_value=configured
                    ),
                    patch.dict(
                        os.environ,
                        {name: " environment " for name in environment_names},
                        clear=True,
                    ),
                ):
                    clear_config_cache()
                    self.assertEqual(getattr(get_config(), credential), expected)

    def test_default_config_properties(self) -> None:
        cfg = get_config()
        self.assertTrue(cfg.is_disc_folder("CD1"))
        self.assertTrue(cfg.is_disc_folder("Disc 2"))
        self.assertTrue(cfg.is_disc_folder("side 1"))
        self.assertFalse(cfg.is_disc_folder("Great Album"))

        self.assertTrue(cfg.is_generic_container("flac"))
        self.assertTrue(cfg.is_generic_container("singles"))
        self.assertTrue(cfg.is_generic_container("downloads"))
        self.assertFalse(cfg.is_generic_container("Pink Floyd"))

    def test_custom_config_instantiation(self) -> None:
        custom_cfg = SonoraConfig(
            disc_folder_patterns=(r"^part\s*\d+$",),
            generic_containers=frozenset({"custom_box"}),
            featuring_conjunctions=("feat", "ft", "avec"),
        )
        self.assertTrue(custom_cfg.is_disc_folder("part 1"))
        self.assertFalse(custom_cfg.is_disc_folder("CD1"))
        self.assertTrue(custom_cfg.is_generic_container("custom_box"))
        self.assertFalse(custom_cfg.is_generic_container("flac"))

    def test_config_parsing_toml_and_json(self) -> None:
        toml_path = self.test_directory / "config.toml"
        toml_path.write_text(
            '[sonora]\nartist_match_threshold = 92.5\ndisc_folder_patterns = ["^disc_\\\\d+$"]\n',
            encoding="utf-8",
        )
        parsed_toml = _parse_config_data(toml_path)
        self.assertEqual(parsed_toml.get("artist_match_threshold"), 92.5)
        self.assertEqual(parsed_toml.get("disc_folder_patterns"), ["^disc_\\d+$"])

        json_path = self.test_directory / "config.json"
        json_path.write_text(
            '{"artist_match_threshold": 88.0, "featuring_conjunctions": ["feat", "ft"]}',
            encoding="utf-8",
        )
        parsed_json = _parse_config_data(json_path)
        self.assertEqual(parsed_json.get("artist_match_threshold"), 88.0)

        empty_path = self.test_directory / "empty.toml"
        empty_path.write_text("", encoding="utf-8")
        self.assertEqual(_parse_config_data(empty_path), {})

        corrupt_path = self.test_directory / "corrupt.toml"
        corrupt_path.write_text("invalid [ [ toml", encoding="utf-8")
        self.assertEqual(_parse_config_data(corrupt_path), {})

    def test_dynamic_pattern_generation(self) -> None:
        split_pat = get_artist_split_pattern()
        self.assertEqual(
            split_pat.split("Artist A feat. Artist B"), ["Artist A", "Artist B"]
        )
        self.assertEqual(
            split_pat.split("Artist A & Artist B"), ["Artist A", "Artist B"]
        )

        bracket_pat = get_bracket_feat_pattern()
        self.assertEqual(bracket_pat.sub("", "Title (feat. Artist)").strip(), "Title")
        self.assertEqual(bracket_pat.sub("", "Title [con Artist]").strip(), "Title")

        balanced_pat = get_balanced_feat_pattern()
        self.assertIsNotNone(balanced_pat.match("feat. Someone"))
        self.assertIsNotNone(balanced_pat.match("con Alguien"))
        self.assertIsNone(balanced_pat.match("Random Word"))

        tokens_pat = get_feat_tokens_pattern()
        self.assertEqual(
            [
                t.strip()
                for t in tokens_pat.split("Artist A & Artist B feat. Artist C")
                if t.strip()
            ],
            ["Artist A", "Artist B", "Artist C"],
        )

        disambig_pat = get_disambiguation_pattern()
        self.assertEqual(
            disambig_pat.sub("", "Armin (ROU) feat. ASSAF (ROU)").strip(),
            "Armin feat. ASSAF",
        )

        dup_pat = get_duplicate_feat_pattern()
        self.assertEqual(
            dup_pat.sub("", "Track feat. Artist (feat. Artist)"),
            "Track (feat. Artist)",
        )

    def test_normalize_country_name(self) -> None:
        self.assertEqual(normalize_country_name("US"), "United States")
        self.assertEqual(normalize_country_name("USA"), "United States")
        self.assertEqual(normalize_country_name("RO"), "Romania")
        self.assertEqual(normalize_country_name("ROU"), "Romania")
        self.assertEqual(normalize_country_name("GB"), "United Kingdom")
        self.assertEqual(normalize_country_name("GBR"), "United Kingdom")
        self.assertEqual(normalize_country_name("DE"), "Germany")
        self.assertEqual(normalize_country_name("United States"), "United States")
        self.assertEqual(normalize_country_name("Romania"), "Romania")
        self.assertEqual(normalize_country_name("XW"), "Worldwide")
        self.assertEqual(normalize_country_name("worldwide"), "Worldwide")
        self.assertEqual(normalize_country_name("XE"), "Europe")
        self.assertEqual(
            normalize_country_name("SU"),
            "USSR, Union of Soviet Socialist Republics",
        )
        self.assertEqual(
            normalize_country_name("SUN"),
            "USSR, Union of Soviet Socialist Republics",
        )
        self.assertIsNone(normalize_country_name(None))
        self.assertIsNone(normalize_country_name(""))
        self.assertIsNone(normalize_country_name("   "))
        self.assertEqual(
            normalize_country_name("CustomFictionalCountryXYZ"),
            "CustomFictionalCountryXYZ",
        )

    def test_normalize_language_name(self) -> None:
        self.assertEqual(normalize_language_name("en"), "English")
        self.assertEqual(normalize_language_name("EN"), "English")
        self.assertEqual(normalize_language_name("eng"), "English")
        self.assertEqual(normalize_language_name("English"), "English")
        self.assertEqual(normalize_language_name("ro"), "Romanian")
        self.assertEqual(normalize_language_name("ron"), "Romanian")
        self.assertEqual(normalize_language_name("Romanian"), "Romanian")
        self.assertEqual(normalize_language_name("de"), "German")
        self.assertEqual(normalize_language_name("deu"), "German")
        self.assertEqual(normalize_language_name("ger"), "German")
        self.assertEqual(normalize_language_name("fre"), "French")
        self.assertEqual(normalize_language_name("ja"), "Japanese")
        self.assertEqual(normalize_language_name("jpn"), "Japanese")
        self.assertIsNone(normalize_language_name(None))
        self.assertIsNone(normalize_language_name(""))
        self.assertIsNone(normalize_language_name("   "))
        self.assertEqual(
            normalize_language_name("KlingonXYZ"),
            "KlingonXYZ",
        )
        self.assertEqual(normalize_language_name("unknown"), "unknown")

    def test_normalize_script_name(self) -> None:
        self.assertEqual(normalize_script_name("Latn"), "Latin")
        self.assertEqual(normalize_script_name("latin"), "Latin")
        self.assertEqual(normalize_script_name("Latin"), "Latin")
        self.assertEqual(normalize_script_name("Cyrl"), "Cyrillic")
        self.assertEqual(normalize_script_name("Cyrillic"), "Cyrillic")
        self.assertEqual(normalize_script_name("Arab"), "Arabic")
        self.assertIsNone(normalize_script_name(None))
        self.assertIsNone(normalize_script_name(""))
        self.assertIsNone(normalize_script_name("   "))
        self.assertEqual(normalize_script_name("UnknownScriptXYZ"), "UnknownScriptXYZ")

    def test_fanart_and_shazam_config(self) -> None:
        cfg = SonoraConfig(
            fanart_api_key="test_api_key",
            fanart_client_key="test_client_key",
            enable_shazam=False,
        )
        self.assertEqual(cfg.fanart_api_key, "test_api_key")
        self.assertEqual(cfg.fanart_client_key, "test_client_key")
        self.assertFalse(cfg.enable_shazam)

    def test_clean_unicode_punct_spaces_before_comma(self) -> None:
        self.assertEqual(
            clean_unicode_punct("Stres feat. Anuryh , Boier Bibescu , SHIFT"),
            "Stres feat. Anuryh, Boier Bibescu, SHIFT",
        )
        self.assertEqual(clean_unicode_punct("Artist , Title"), "Artist, Title")
        self.assertEqual(clean_unicode_punct("None"), "None")
        self.assertEqual(clean_unicode_punct(None), "")

    def test_extract_version_modifier(self) -> None:
        self.assertEqual(
            extract_version_modifier("Breaking Me (RetroVision remix)"),
            "retrovision remix",
        )
        self.assertEqual(
            extract_version_modifier("Breaking Me (HUGEL remix)"),
            "hugel remix",
        )
        self.assertEqual(
            extract_version_modifier("Electrified [Extended Version]"),
            "extended version",
        )
        self.assertIsNone(extract_version_modifier("Standard Title"))
        self.assertIsNone(extract_version_modifier("Title (feat. Someone)"))

    def test_preserve_unicode_repertoire(self) -> None:
        self.assertEqual(
            preserve_unicode_repertoire("Bombe în rai", "Bombe in rai"),
            "Bombe în rai",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Bombe in rai", "Bombe în rai"),
            "Bombe în rai",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Andreea Bănică", "Andreea Banica"),
            "Andreea Bănică",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Fără Lacrimi", "Fara Lacrimi"),
            "Fără Lacrimi",
        )
        # International languages (Turkish, Icelandic, German)
        self.assertEqual(
            preserve_unicode_repertoire("Barış Manço", "Baris Manco"),
            "Barış Manço",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Sigur Rós", "Sigur Ros"),
            "Sigur Rós",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Mötley Crüe", "Motley Crue"),
            "Mötley Crüe",
        )
        # Diacritic density: string with more accents preferred over partial
        self.assertEqual(
            preserve_unicode_repertoire("Andreea Bănică", "Andreea Bănica"),
            "Andreea Bănică",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Andreea Bănica", "Andreea Bănică"),
            "Andreea Bănică",
        )
        self.assertEqual(preserve_unicode_repertoire(None, "Candidate"), "Candidate")
        self.assertEqual(preserve_unicode_repertoire("Current", None), "Current")
        self.assertEqual(preserve_unicode_repertoire("Title A", "Title B"), "Title B")

    def test_normalize_legacy_diacritics(self) -> None:
        # T-cedilla is unconditionally normalized to standard comma-below
        self.assertEqual(
            normalize_legacy_diacritics("Ţi-O Dau La Muie"),
            "Ți-O Dau La Muie",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Cât Poţi Tu de Tare"),
            "Cât Poți Tu de Tare",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Mii De Feţe"),
            "Mii De Fețe",
        )
        self.assertEqual(
            normalize_legacy_diacritics("1 TEL. DISTANŢĂ"),
            "1 TEL. DISTANȚĂ",
        )
        self.assertEqual(
            normalize_legacy_diacritics("ROTAŢIE"),
            "ROTAȚIE",
        )

        # Romanian S-cedilla normalized to S-comma below when Romanian markers or words exist
        self.assertEqual(
            normalize_legacy_diacritics("... Şi Cui Îi Pasă?"),
            "... Și Cui Îi Pasă?",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Bag Pula-N Lume Şi V-o Fac Cadou"),
            "Bag Pula-N Lume Și V-o Fac Cadou",
        )
        self.assertEqual(
            normalize_legacy_diacritics("BUCUREŞTI"),
            "BUCUREȘTI",
        )
        self.assertEqual(
            normalize_legacy_diacritics("AM ŞOFER"),
            "AM ȘOFER",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Dani Buşoi"),
            "Dani Bușoi",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Maşina Timpului"),
            "Mașina Timpului",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Leş Elephants Bizarres"),
            "Leș Elephants Bizarres",
        )

        # Turkish S-cedilla is preserved
        self.assertEqual(
            normalize_legacy_diacritics("Barış Manço"),
            "Barış Manço",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Şımarık"),
            "Şımarık",
        )

        # Integration with clean_unicode_punct and preserve_unicode_repertoire
        self.assertEqual(
            clean_unicode_punct("... Şi Cui Îi Pasă?"),
            "... Și Cui Îi Pasă?",
        )
        self.assertEqual(
            preserve_unicode_repertoire("... Si Cui Ii Pasa?", "... Şi Cui Îi Pasă?"),
            "... Și Cui Îi Pasă?",
        )

        # Empty and None handling
        self.assertEqual(normalize_legacy_diacritics(""), "")
        self.assertEqual(normalize_legacy_diacritics(None), "")

    def test_match_score_penalties_and_remixes(self) -> None:
        # Stopword immunity: "The" in band names must not produce high match scores
        self.assertEqual(
            score_artist_title_match(
                "The Motans", "Sangria", "The Re-Tards", "Sangria"
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "The Motans",
                "Sangria",
                "But Then the Self in the Tree Fell Asleep",
                "Sangria",
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "The Motans", "Flagrant", "Audra The Rapper feat. Satasha", "Flagrant"
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "The Motans", "Sativa", "The Body Rampant", "Sativa"
            ),
            0.0,
        )

        # Remix modifier penalty: Different remixers must score 0.0
        self.assertEqual(
            score_artist_title_match(
                "Topic & A7S",
                "Breaking Me (RetroVision remix)",
                "Topic & A7S",
                "Breaking Me (HUGEL remix)",
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "Topic & A7S",
                "Breaking Me (Riton remix)",
                "Topic & A7S",
                "Breaking Me (HUGEL remix)",
            ),
            0.0,
        )

        # Identical remix modifier should match perfectly
        self.assertGreaterEqual(
            score_artist_title_match(
                "Topic & A7S",
                "Breaking Me (RetroVision remix)",
                "Topic & A7S",
                "Breaking Me (RetroVision remix)",
            ),
            95.0,
        )

    def test_is_interruption(self) -> None:
        self.assertTrue(is_interruption(KeyboardInterrupt()))
        self.assertTrue(is_interruption(InterruptedOperationError()))
        self.assertFalse(is_interruption(RuntimeError("release unlocked lock")))

        nested_exc = RuntimeError("wrapper error")
        nested_exc.__context__ = KeyboardInterrupt()
        self.assertTrue(is_interruption(nested_exc))

        self.assertFalse(is_interruption(RuntimeError("unrelated database error")))
        self.assertFalse(is_interruption(ValueError("invalid value")))
        self.assertFalse(is_interruption(OSError("file not found")))

    def test_is_in_singles_hierarchy(self) -> None:
        self.assertTrue(
            is_in_singles_hierarchy(
                Path("/home/music/Artist/Singles/Artist - Song/01 - Song.flac")
            )
        )
        self.assertTrue(
            is_in_singles_hierarchy(Path("/home/music/Singles/Artist - Song"))
        )
        self.assertTrue(
            is_in_singles_hierarchy(
                Path("/home/music/FLAC/Downloads/song.flac"),
                root_dir=Path("/home/music/FLAC"),
            )
        )
        self.assertFalse(
            is_in_singles_hierarchy(
                Path("/home/music/FLAC/Artist/Album/01 - Song.flac"),
                root_dir=Path("/home/music/FLAC"),
            )
        )
        self.assertFalse(
            is_in_singles_hierarchy(
                Path("/home/music/FLAC/21 Savage/21 Savage - Savage Mode")
            )
        )

    def test_get_single_release_title(self) -> None:
        info1 = TrackInfo(
            file_path=Path("dummy.flac"), title="Minim Doi", album="Track 2"
        )
        self.assertEqual(get_single_release_title(info1), "Minim Doi")

        info2 = TrackInfo(
            file_path=Path("dummy.flac"),
            title="Electrified (extended version)",
            album="O Fortuna (extended version)",
        )
        self.assertEqual(
            get_single_release_title(info2), "Electrified (extended version)"
        )

        info3 = TrackInfo(
            file_path=Path("dummy.flac"),
            title="I Loved You",
            album="I Loved You (Monoir Remix)",
        )
        self.assertEqual(get_single_release_title(info3), "I Loved You (Monoir Remix)")

        info4 = TrackInfo(
            file_path=Path("dummy.flac"),
            title="Tren De Noapte",
            album="Track 2 (feat. Joyce Meyer)",
        )
        self.assertEqual(get_single_release_title(info4), "Tren De Noapte")


class TestCoreCancellation(OfflinePipelineTestCase):
    def test_backup_handles_keyboard_interrupt(self) -> None:
        wav = self.tmp_path / "test.wav"
        write_silent_wav(wav)
        with (
            patch(
                "sonora.modules.backup._read_track_for_backup",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            backup_library_tags(self.tmp_path, max_threads=2)

    def test_restore_handles_keyboard_interrupt(self) -> None:
        backup_file = self.tmp_path / "backup.json"
        backup_file.write_text(
            '{"schema":"sonora_backup_v2","root":"/tmp","tracks":{"test.wav":{"tags":{},"complex_tags":{}}}}',
            encoding="utf-8",
        )
        with (
            patch(
                "sonora.modules.backup._restore_single_track",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            restore_library_tags(backup_file, max_threads=2)

    def test_organize_handles_keyboard_interrupt(self) -> None:
        wav1 = self.tmp_path / "test1.wav"
        wav2 = self.tmp_path / "test2.wav"
        write_silent_wav(wav1)
        write_silent_wav(wav2)
        with (
            patch(
                "sonora.modules.organizer.read_track_metadata",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            organize_library_singles(
                self.tmp_path, self.tmp_path / "Singles", max_threads=2
            )

    def test_renamer_handles_keyboard_interrupt(self) -> None:
        wav1 = self.tmp_path / "test1.wav"
        wav2 = self.tmp_path / "test2.wav"
        write_silent_wav(wav1)
        write_silent_wav(wav2)
        with (
            patch(
                "sonora.modules.renamer.read_track_metadata",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            rename_directory_files(self.tmp_path, max_threads=2)

    def test_checker_handles_keyboard_interrupt(self) -> None:
        wav = self.tmp_path / "test.wav"
        write_silent_wav(wav)
        with (
            patch(
                "sonora.modules.checker._check_single_file",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            check_library(self.tmp_path, max_threads=2)

    def test_replaygain_handles_keyboard_interrupt(self) -> None:
        wav1 = self.tmp_path / "test1.wav"
        wav2 = self.tmp_path / "test2.wav"
        write_silent_wav(wav1)
        write_silent_wav(wav2)
        with (
            patch(
                "sonora.audio.replaygain._measure_track_loudness",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            calculate_album_replaygain([wav1, wav2], max_threads=2)

    def test_normalize_handles_keyboard_interrupt(self) -> None:
        wav = self.tmp_path / "test.wav"
        write_silent_wav(wav)
        with (
            patch(
                "sonora.modules.track_processing.normalize_single_track",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            normalize_library(self.tmp_path, max_threads=2)

    def test_tagger_handles_keyboard_interrupt(self) -> None:
        wav = self.tmp_path / "test.wav"
        write_silent_wav(wav)
        with (
            patch(
                "sonora.modules.tagger.process_single_track",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_album_details", return_value=None
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            tag_album_folder(
                self.tmp_path,
                max_threads=2,
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )


class TestCoreHttpAndRetry(IsolatedTestCase):
    def test_invalid_attempt_budget_is_rejected(self) -> None:
        for attempts in (0, -1):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                RetryTransport(max_retries=attempts)

    def test_discogs_rate_limit_depends_on_authentication(self) -> None:
        limiter = Mock()
        for headers, interval in (
            ({}, RATE_LIMIT_DISCOGS_UNAUTHENTICATED),
            ({"Authorization": "Discogs token=fixture"}, None),
        ):
            limiter.reset_mock()
            with (
                self.subTest(headers=headers),
                patch("sonora.core.http._HOST_LIMITERS", {"api.discogs.com": limiter}),
                patch(
                    "httpx.HTTPTransport.handle_request",
                    return_value=httpx.Response(200),
                ),
                RetryTransport() as transport,
            ):
                response = transport.handle_request(
                    httpx.Request(
                        "GET", "https://api.discogs.com/test", headers=headers
                    )
                )
            self.assertEqual(response.status_code, 200)
            limiter.wait.assert_called_once_with(interval)

    def test_long_retry_after_returns_response_without_sleeping(self) -> None:
        response = httpx.Response(429, headers={"Retry-After": "61"})
        with (
            patch("httpx.HTTPTransport.handle_request", return_value=response) as send,
            patch("sonora.core.http.time.sleep") as sleep,
            RetryTransport() as transport,
        ):
            self.assertIs(
                transport.handle_request(httpx.Request("GET", "https://example.com")),
                response,
            )
        send.assert_called_once()
        sleep.assert_not_called()

    def test_exhausted_transport_failure_preserves_original_exception(self) -> None:
        error = httpx.ReadTimeout("socket timed out")
        with (
            patch("httpx.HTTPTransport.handle_request", side_effect=error) as send,
            patch("sonora.core.http.time.sleep") as sleep,
            RetryTransport(max_retries=2, backoff_factor=0.25) as transport,
            self.assertRaises(httpx.ReadTimeout) as raised,
        ):
            transport.handle_request(httpx.Request("GET", "https://example.com"))
        self.assertIs(raised.exception, error)
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once_with(0.25)

    def test_dns_lookup_failures_are_not_retried(self) -> None:
        for code in (socket.EAI_AGAIN, socket.EAI_NONAME):
            with self.subTest(code=code):
                dns_error = socket.gaierror(code, "DNS lookup failed")
                error = httpx.ConnectError("connection failed")
                error.__cause__ = dns_error
                with (
                    patch("sonora.core.retry._DNS_UNAVAILABLE_UNTIL", 0.0),
                    patch(
                        "httpx.HTTPTransport.handle_request", side_effect=error
                    ) as send,
                    patch("sonora.core.http.time.sleep") as sleep,
                    RetryTransport() as transport,
                    self.assertRaises(httpx.ConnectError) as raised,
                ):
                    transport.handle_request(
                        httpx.Request("GET", "https://example.com")
                    )
                self.assertIs(raised.exception, error)
                send.assert_called_once()
                sleep.assert_not_called()
        send.assert_called_once()
        sleep.assert_not_called()

    def test_unknown_host_does_not_suspend_other_http_requests(self) -> None:
        import socket

        request = httpx.Request("GET", "https://example.com")
        error = httpx.ConnectError("connection failed", request=request)
        error.__cause__ = socket.gaierror(socket.EAI_NONAME, "Unknown host")
        responses = [error, httpx.Response(200, request=request)]
        with (
            patch("sonora.core.retry._DNS_UNAVAILABLE_UNTIL", 0.0),
            patch("httpx.HTTPTransport.handle_request", side_effect=responses) as send,
            patch("sonora.core.http.time.sleep") as sleep,
            RetryTransport(max_retries=1) as transport,
        ):
            with self.assertRaises(httpx.ConnectError):
                transport.handle_request(request)
            response = transport.handle_request(request)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(send.call_count, 2)
            sleep.assert_not_called()

    def test_retry_dates_support_naive_utc_and_timezone_aware_deadlines(self) -> None:
        instant = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
        for deadline in (
            instant + timedelta(seconds=7),
            (instant + timedelta(seconds=7)).replace(tzinfo=None),
        ):
            with (
                self.subTest(deadline=deadline),
                patch("sonora.core.http.parsedate_to_datetime", return_value=deadline),
                patch("sonora.core.http.datetime", wraps=datetime) as clock,
            ):
                clock.now.return_value = instant
                self.assertEqual(_retry_delay("HTTP-date", 1.5), 7)

    def test_lrclib_workers_wait_for_each_response_before_next_request(self) -> None:
        import threading

        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        active = 0
        maximum = 0
        lock = threading.Lock()
        request = httpx.Request("GET", "https://lrclib.net/api/get")

        def send(request: httpx.Request) -> httpx.Response:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.01)
            finally:
                with lock:
                    active -= 1
            return httpx.Response(200, request=request)

        with (
            RetryTransport() as transport,
            patch.object(_HOST_LIMITERS["lrclib.net"], "wait") as pace_request,
            patch.object(httpx.HTTPTransport, "handle_request", side_effect=send),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            responses = list(pool.map(transport.handle_request, [request] * 4))
        self.assertEqual([response.status_code for response in responses], [200] * 4)
        self.assertEqual(maximum, 1)
        self.assertEqual(pace_request.call_count, 4)
        self.assertEqual(_HOST_LIMITERS["lrclib.net"].interval, 0.3)

    def test_provider_cooldown_is_shared_and_cannot_be_shortened(self) -> None:
        from sonora.core.coordination import wait_for_provider

        with (
            patch(
                "sonora.core.coordination.get_cache_dir",
                return_value=self.test_directory,
            ),
            patch("sonora.core.coordination.time.time", return_value=100.0),
            patch("sonora.core.coordination.time.sleep") as sleep,
        ):
            wait_for_provider("lrclib.net", 0, defer_seconds=30)
            wait_for_provider("lrclib.net", 0, defer_seconds=1)
            sleep.assert_not_called()
            self.assertEqual(wait_for_provider("lrclib.net", 0.25), 30)
            sleep.assert_called_once_with(30)

    def test_long_provider_cooldown_defers_workers_without_waiting(self) -> None:
        from sonora.core.coordination import (
            ProviderDeferredError,
            wait_for_provider,
        )

        with (
            patch(
                "sonora.core.coordination.get_cache_dir",
                return_value=self.test_directory,
            ),
            patch("sonora.core.coordination.time.time", return_value=100.0),
            patch("sonora.core.coordination.time.sleep") as sleep,
        ):
            wait_for_provider("lrclib.net", 0, defer_seconds=3600)
            with self.assertRaises(ProviderDeferredError):
                wait_for_provider("lrclib.net", 0.25)
        sleep.assert_not_called()

    def test_retry_transport_does_not_retry_a_deferred_provider(self) -> None:
        import httpx

        from sonora.core.coordination import ProviderDeferredError
        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        with (
            patch.object(
                _HOST_LIMITERS["lrclib.net"],
                "wait",
                side_effect=ProviderDeferredError("provider cooling down"),
            ),
            patch.object(httpx.HTTPTransport, "handle_request") as send,
            RetryTransport(max_retries=3) as transport,
            self.assertRaises(ProviderDeferredError),
        ):
            transport.handle_request(httpx.Request("GET", "https://lrclib.net/api"))
        send.assert_not_called()

    def test_final_rate_limit_response_sets_shared_cooldown(self) -> None:
        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        request = httpx.Request("GET", "https://lrclib.net/api/get")
        response = httpx.Response(429, headers={"Retry-After": "30"}, request=request)
        with (
            RetryTransport(max_retries=1) as transport,
            patch.object(RateLimiter, "_disabled", False),
            patch.object(_HOST_LIMITERS["lrclib.net"], "wait"),
            patch.object(httpx.HTTPTransport, "handle_request", return_value=response),
            patch("sonora.core.http.wait_for_provider") as defer,
            patch("sonora.core.http.SystemRandom") as random,
        ):
            random.return_value.uniform.return_value = 3.0
            self.assertIs(transport.handle_request(request), response)
            defer.assert_called_once_with("lrclib.net", 0, defer_seconds=33)

    def test_retry_network_call_success_on_first_try(self) -> None:

        mock_func = MagicMock(return_value="ok")
        result = retry_network_call(mock_func)
        self.assertEqual(result, "ok")
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_network_call_retries_and_succeeds(self) -> None:

        limiter = RateLimiter(interval_seconds=0)
        mock_func = MagicMock(
            side_effect=[OSError("Connection lost"), OSError("Timeout"), "recovered"]
        )
        with patch("time.sleep") as mock_sleep:
            result = retry_network_call(
                mock_func,
                max_retries=3,
                initial_delay=0.01,
                rate_limiter=limiter,
            )
        self.assertEqual(result, "recovered")
        self.assertEqual(mock_func.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_retry_network_call_exhausts_retries(self) -> None:

        mock_func = MagicMock(side_effect=OSError("Permanent failure"))
        with patch("time.sleep"), self.assertRaises(OSError):
            retry_network_call(mock_func, max_retries=3, initial_delay=0.01)
        self.assertEqual(mock_func.call_count, 3)

    def test_retry_network_call_skips_non_retryable(self) -> None:

        mock_func = MagicMock(side_effect=ValueError("Invalid parameter"))
        with patch("time.sleep"), self.assertRaises(ValueError):
            retry_network_call(
                mock_func,
                max_retries=3,
                transient_exceptions=(OSError,),
            )
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_network_call_skips_dns_lookup_failures(self) -> None:
        import socket

        from musicbrainzngs import NetworkError

        for code in (socket.EAI_AGAIN, socket.EAI_NONAME):
            with self.subTest(code=code):
                failure = NetworkError(
                    "MusicBrainz network error",
                    cause=socket.gaierror(code, "DNS lookup failed"),
                )
                mock_func = MagicMock(side_effect=failure)
                with (
                    patch("time.sleep") as sleep,
                    patch(
                        "sonora.core.retry._DNS_UNAVAILABLE_UNTIL",
                        0.0,
                        create=True,
                    ),
                    self.assertRaises(NetworkError),
                ):
                    retry_network_call(mock_func, transient_exceptions=(NetworkError,))
                mock_func.assert_called_once()
                sleep.assert_not_called()

    def test_dns_failure_suspends_repeated_network_calls_briefly(self) -> None:
        import socket

        first = MagicMock(side_effect=socket.gaierror(socket.EAI_AGAIN, "DNS failed"))
        later = MagicMock(return_value="unexpected")
        with self.assertRaises(socket.gaierror):
            retry_network_call(first)
        with self.assertRaises(socket.gaierror):
            retry_network_call(later)
        first.assert_called_once()
        later.assert_not_called()

    def test_retry_network_call_rechecks_dns_after_rate_limit_wait(self) -> None:
        import socket

        from sonora.core.retry import _suspend_dns_requests

        limiter = RateLimiter(interval_seconds=0)
        request = MagicMock(return_value="unexpected")
        with (
            patch("sonora.core.retry._DNS_UNAVAILABLE_UNTIL", 0.0),
            patch.object(limiter, "wait", side_effect=_suspend_dns_requests),
            self.assertRaises(socket.gaierror),
        ):
            retry_network_call(request, rate_limiter=limiter)
        request.assert_not_called()

    def test_permanent_dns_failure_does_not_suspend_other_lookups(self) -> None:
        import socket

        failure = MagicMock(
            side_effect=socket.gaierror(socket.EAI_NONAME, "Unknown host")
        )
        available = MagicMock(return_value="available")
        with self.assertRaises(socket.gaierror):
            retry_network_call(failure)
        self.assertEqual(retry_network_call(available), "available")
        failure.assert_called_once()
        available.assert_called_once()

    def test_dns_cooldown_expires(self) -> None:
        from sonora.core.retry import _dns_requests_suspended, _suspend_dns_requests

        with (
            patch("sonora.core.retry.time.monotonic", side_effect=[5.0, 34.9, 35.0]),
        ):
            _suspend_dns_requests()
            self.assertTrue(_dns_requests_suspended())
            self.assertFalse(_dns_requests_suspended())

    def test_http_transport_skips_requests_during_dns_cooldown(self) -> None:
        import socket

        import httpx

        from sonora.core.http import RetryTransport

        request = httpx.Request("GET", "https://example.com")
        failure = httpx.ConnectError("DNS failed", request=request)
        failure.__cause__ = socket.gaierror(socket.EAI_AGAIN, "DNS failed")
        with (
            RetryTransport() as transport,
            patch.object(
                httpx.HTTPTransport, "handle_request", side_effect=failure
            ) as send,
        ):
            with self.assertRaises(httpx.ConnectError):
                transport.handle_request(request)
            with self.assertRaises(httpx.ConnectError):
                transport.handle_request(request)
        self.assertEqual(send.call_count, 1)

    def test_http_transport_rechecks_dns_after_rate_limit_wait(self) -> None:
        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport
        from sonora.core.retry import _suspend_dns_requests

        request = httpx.Request("GET", "https://lrclib.net/api/get")
        with (
            RetryTransport() as transport,
            patch("sonora.core.retry._DNS_UNAVAILABLE_UNTIL", 0.0),
            patch.object(
                _HOST_LIMITERS["lrclib.net"],
                "wait",
                side_effect=lambda *_args: _suspend_dns_requests(),
            ),
            patch.object(httpx.HTTPTransport, "handle_request") as send,
            self.assertRaises(httpx.ConnectError),
        ):
            transport.handle_request(request)
        send.assert_not_called()

    def test_http_transport_does_not_retry_after_another_request_suspends_dns(
        self,
    ) -> None:
        import httpx

        from sonora.core.http import RetryTransport
        from sonora.core.retry import _suspend_dns_requests

        request = httpx.Request("GET", "https://example.com")

        def fail_during_dns_outage(request: httpx.Request) -> httpx.Response:
            _suspend_dns_requests()
            raise httpx.ConnectTimeout("resolver unavailable", request=request)

        with (
            RetryTransport(max_retries=2) as transport,
            patch.object(
                httpx.HTTPTransport,
                "handle_request",
                side_effect=fail_during_dns_outage,
            ) as send,
            patch("sonora.core.http.time.sleep") as sleep,
            self.assertRaises(httpx.ConnectTimeout),
        ):
            transport.handle_request(request)

        send.assert_called_once_with(request)
        sleep.assert_not_called()

    def test_retry_network_call_respects_retry_condition(self) -> None:

        mock_func = MagicMock(side_effect=OSError("Forbidden"))
        with patch("time.sleep"), self.assertRaises(OSError):
            retry_network_call(
                mock_func,
                max_retries=3,
                retry_condition=lambda err: "Forbidden" not in str(err),
            )
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_transport_retries_on_429_and_503(self) -> None:
        import httpx

        from sonora.core.http import RetryTransport

        resp_429 = httpx.Response(
            429,
            headers={"Retry-After": "2"},
            request=httpx.Request("GET", "https://example.com"),
        )
        resp_503 = httpx.Response(
            503, request=httpx.Request("GET", "https://example.com")
        )
        resp_200 = httpx.Response(
            200,
            json={"status": "ok"},
            request=httpx.Request("GET", "https://example.com"),
        )

        with (
            RetryTransport(max_retries=3, backoff_factor=0.01) as transport,
            patch.object(
                httpx.HTTPTransport,
                "handle_request",
                side_effect=[resp_429, resp_503, resp_200],
            ) as mock_handle,
            patch("time.sleep") as mock_sleep,
        ):
            response = transport.handle_request(
                httpx.Request("GET", "https://example.com")
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_handle.call_count, 3)
        mock_sleep.assert_any_call(2.0)

    def test_lrclib_503_without_retry_after_uses_one_second_fallback(self) -> None:
        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        request = httpx.Request("GET", "https://lrclib.net/api/get")
        responses = [
            httpx.Response(503, request=request),
            httpx.Response(200, request=request),
        ]
        with (
            RetryTransport(max_retries=2) as transport,
            patch.object(_HOST_LIMITERS["lrclib.net"], "wait"),
            patch("sonora.core.http.wait_for_provider"),
            patch.object(httpx.HTTPTransport, "handle_request", side_effect=responses),
            patch("sonora.core.http.SystemRandom") as random,
            patch("time.sleep") as sleep,
        ):
            random.return_value.uniform.return_value = 0
            response = transport.handle_request(request)

        self.assertEqual(response.status_code, 200)
        sleep.assert_called_once_with(1.0)

    def test_lrclib_stops_after_one_retry_for_repeated_server_errors(self) -> None:
        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        request = httpx.Request("GET", "https://lrclib.net/api/get")
        responses = [
            httpx.Response(502, request=request),
            httpx.Response(502, request=request),
            httpx.Response(200, request=request),
        ]
        with (
            RetryTransport() as transport,
            patch.object(_HOST_LIMITERS["lrclib.net"], "wait"),
            patch("sonora.core.http.wait_for_provider"),
            patch.object(
                httpx.HTTPTransport, "handle_request", side_effect=responses
            ) as request_handler,
            patch("sonora.core.http.SystemRandom") as random,
            patch("time.sleep"),
        ):
            random.return_value.uniform.return_value = 0
            response = transport.handle_request(request)

        self.assertEqual(response.status_code, 502)
        self.assertEqual(request_handler.call_count, 2)
