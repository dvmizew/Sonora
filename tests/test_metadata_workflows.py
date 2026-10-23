from pathlib import Path
from unittest.mock import MagicMock, patch

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.audio_files import find_audio_files
from sonora.core.models import TrackInfo
from sonora.core.titles import score_artist_title_match, strip_corrupt_brackets
from sonora.modules.matching import is_alien_album_track
from sonora.services.artist_resolution import resolve_artist_name
from tests.fixtures import write_silent_wav
from tests.module_support import ModuleTestCase


class TestMetadataWorkflows(ModuleTestCase):
    @patch("sonora.services.artist_resolution.get_cached_api", return_value=None)
    @patch("sonora.services.musicbrainz.search_musicbrainz_artists")
    def test_resolve_artist_name(
        self, mock_search: MagicMock, mock_get_cache: MagicMock
    ) -> None:
        mock_search.return_value = [
            {
                "name": "M.G.L.",
                "alias-list": [{"alias": "mgl"}],
            },
            {
                "name": "Killa Fonic",
                "alias-list": [],
            },
            {
                "name": "Nane",
                "alias-list": [],
            },
        ]
        self.assertEqual(resolve_artist_name("mgl"), "M.G.L.")
        self.assertEqual(resolve_artist_name("killa fonic"), "Killa Fonic")
        self.assertEqual(resolve_artist_name("nane"), "Nane")

    def test_strip_corrupt_brackets(self) -> None:
        self.assertEqual(
            strip_corrupt_brackets("Song Title [Official Video]"),
            "Song Title",
        )
        self.assertEqual(
            strip_corrupt_brackets("Song Title [FLAC]"),
            "Song Title",
        )
        self.assertEqual(
            strip_corrupt_brackets("Song Title (Remix)"),
            "Song Title (Remix)",
        )
        self.assertEqual(
            strip_corrupt_brackets("Candy (feat. Trippie Redd)"),
            "Candy (feat. Trippie Redd)",
        )
        self.assertEqual(
            strip_corrupt_brackets("Song (2011 Remaster)"),
            "Song",
        )
        self.assertEqual(
            strip_corrupt_brackets(""),
            "",
        )

    def test_is_alien_album_track(self) -> None:
        track_path = Path("/music/21 Savage - Issa Album/03 - Blue Jeans.wav")
        alien_track = TrackInfo(
            file_path=track_path,
            artist="Lana Del Rey",
            title="Blue Jeans",
            album="Born to Die",
        )
        mb_release: dict[str, object] = {
            "title": "Issa Album",
            "album_artist": "21 Savage",
            "tracks_by_disc_and_position": {
                (1, 1): {"title": "Famous", "artist": "21 Savage"},
                (1, 2): {"title": "Bank Account", "artist": "21 Savage"},
                (1, 3): {"title": "Close My Eyes", "artist": "21 Savage"},
            },
        }
        # True positive: Lana Del Rey inside 21 Savage album
        self.assertTrue(
            is_alien_album_track(
                track_info=alien_track,
                file_path=track_path,
                target_album_artist="21 Savage",
                target_album_title="Issa Album",
                album_mb_release_details=mb_release,
            )
        )

        # False positive check: Valid 21 Savage track
        valid_track = TrackInfo(
            file_path=Path("/music/21 Savage - Issa Album/02 - Bank Account.wav"),
            artist="21 Savage",
            title="Bank Account",
            album="Issa Album",
        )
        self.assertFalse(
            is_alien_album_track(
                track_info=valid_track,
                file_path=valid_track.file_path,
                target_album_artist="21 Savage",
                target_album_title="Issa Album",
                album_mb_release_details=mb_release,
            )
        )

        # Unnamed / generic track in album should NOT be marked alien
        generic_track = TrackInfo(
            file_path=Path("/music/21 Savage - Issa Album/01.wav"),
            artist="Unknown Artist",
            title="Track 01",
            album="Unknown Album",
        )
        self.assertFalse(
            is_alien_album_track(
                track_info=generic_track,
                file_path=generic_track.file_path,
                target_album_artist="21 Savage",
                target_album_title="Issa Album",
                album_mb_release_details=mb_release,
            )
        )

        # Featured artist / collaboration track
        collab_track = TrackInfo(
            file_path=Path("/music/Drake & 21 Savage - Her Loss/01 - Rich Flex.wav"),
            artist="21 Savage",
            title="Rich Flex",
        )
        self.assertFalse(
            is_alien_album_track(
                track_info=collab_track,
                file_path=collab_track.file_path,
                target_album_artist="Drake & 21 Savage",
                target_album_title="Her Loss",
            )
        )

        # Corrupted title with verified ISRC match should NOT be marked alien
        healed_track = TrackInfo(
            file_path=Path(
                "/music/James Arthur - Back From the Edge/04 - Lust for Life.wav"
            ),
            artist="James Arthur",
            title="Lust for Life (feat. Lana Del Rey & The Weeknd)",
            album="Back From the Edge",
            isrc="DEE861600588",
            track_number=4,
        )
        mb_release_ja: dict[str, object] = {
            "title": "Back From the Edge",
            "album_artist": "James Arthur",
            "tracks_by_disc_and_position": {
                (1, 4): {
                    "title": "Can I Be Him",
                    "artist": "James Arthur",
                    "isrc": "DEE861600588",
                    "recording_mbid": "c8b03190-306c-4125-9b32-3f9d86d60a12",
                },
            },
        }
        self.assertFalse(
            is_alien_album_track(
                track_info=healed_track,
                file_path=healed_track.file_path,
                target_album_artist="James Arthur",
                target_album_title="Back From the Edge",
                album_mb_release_details=mb_release_ja,
            )
        )

        # Bonus track by the same artist whose title is not in known_titles
        bonus_track = TrackInfo(
            file_path=Path("/music/21 Savage - Issa Album/15 - Bonus Track.wav"),
            artist="21 Savage",
            title="7 Min Freestyle (Bonus)",
            album="Issa Album",
        )
        self.assertFalse(
            is_alien_album_track(
                track_info=bonus_track,
                file_path=bonus_track.file_path,
                target_album_artist="21 Savage",
                target_album_title="Issa Album",
                album_mb_release_details=mb_release,
            )
        )

        # Conflicting track by the same artist whose title is not on the album release
        conflicting_track = TrackInfo(
            file_path=Path("/music/XXXTENTACION - Revenge/04 - Revenge.flac"),
            artist="XXXTENTACION",
            title="Revenge",
            album="Revenge",
            track_number=4,
            isrc="USUYG1156895",
        )
        mb_release_rev: dict[str, object] = {
            "title": "Revenge",
            "album_artist": "XXXTENTACION",
            "tracks_by_disc_and_position": {
                (1, 1): {
                    "title": "Look at Me!",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1131080",
                },
                (1, 2): {
                    "title": "I don't wanna do this anymore",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142046",
                },
                (1, 3): {
                    "title": "Looking for a Star",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142047",
                },
                (1, 4): {
                    "title": "KING",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142049",
                },
                (1, 5): {
                    "title": "Slipknot",
                    "artist": "XXXTENTACION feat. Kin$oul & Killstation",
                    "isrc": "USUYG1142050",
                },
                (1, 6): {
                    "title": "YuNg BrAtZ",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142051",
                },
                (1, 7): {
                    "title": 'R.I.P ROACH "EAST SIDE SOULJA"',
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142052",
                },
            },
        }
        self.assertTrue(
            is_alien_album_track(
                track_info=conflicting_track,
                file_path=conflicting_track.file_path,
                target_album_artist="XXXTENTACION",
                target_album_title="Revenge",
                album_mb_release_details=mb_release_rev,
            )
        )

    def test_symfonium_extended_tags(self) -> None:
        wav_path = self.tmp_path / "extended.wav"
        write_silent_wav(wav_path)

        track_metadata = read_track_metadata(wav_path)
        track_metadata.artist_sort = "Artist, Test"
        track_metadata.album_artist_sort = "Artist, Test Album"
        track_metadata.total_tracks = 12
        track_metadata.total_discs = 2
        track_metadata.release_type = "Album"
        track_metadata.release_status = "Official"
        track_metadata.release_country = "US"
        track_metadata.musicbrainz_trackid = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        track_metadata.musicbrainz_albumid = "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
        track_metadata.label = "Test Label"
        track_metadata.barcode = "123456789012"

        write_track_metadata(track_metadata)

        reloaded = read_track_metadata(wav_path)
        self.assertEqual(
            reloaded.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        self.assertEqual(
            reloaded.musicbrainz_albumid, "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
        )

    def test_find_audio_files_ignores_hidden_directories(self) -> None:
        normal_wav = self.tmp_path / "normal.wav"
        write_silent_wav(normal_wav)

        hidden_dir = self.tmp_path / ".venv" / "lib"
        hidden_dir.mkdir(parents=True, exist_ok=True)
        hidden_wav = hidden_dir / "test.wav"
        write_silent_wav(hidden_wav)

        dot_wav = self.tmp_path / "._hidden.flac"
        dot_wav.write_bytes(b"dummy")

        files = find_audio_files(self.tmp_path)
        self.assertEqual(files, [normal_wav])

        # When include_hidden=True, hidden files are included
        all_files = find_audio_files(self.tmp_path, include_hidden=True)
        self.assertIn(hidden_wav, all_files)
        self.assertIn(dot_wav, all_files)

    def test_match_score_short_artist_name_isolation(self) -> None:
        # Short artist names must not match candidates containing them as substrings
        self.assertEqual(score_artist_title_match("Ian", "Song", "Brian", "Song"), 0.0)
        self.assertEqual(
            score_artist_title_match("Ian", "Song", "Arkanian", "Song"), 0.0
        )
        self.assertEqual(
            score_artist_title_match("Ian", "Song", "Christian", "Song"), 0.0
        )
        self.assertEqual(
            score_artist_title_match("Tyga", "Song", "Tyga Ichinose", "Song"), 0.0
        )
        self.assertEqual(
            score_artist_title_match("Aaryan Shah", "Song", "Kavinsky", "Song"), 0.0
        )
        # Collaborations and features on the same primary artist should match
        self.assertGreaterEqual(
            score_artist_title_match(
                "21 Savage & Metro Boomin", "Song", "21 Savage", "Song"
            ),
            80.0,
        )
        self.assertGreaterEqual(
            score_artist_title_match(
                "Kendrick Lamar feat. Travis Scott", "Song", "Kendrick Lamar", "Song"
            ),
            80.0,
        )
