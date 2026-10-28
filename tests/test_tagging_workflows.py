import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from sonora.audio.metadata import write_track_metadata
from sonora.core.models import TrackInfo
from sonora.modules.catalog_enrichment import (
    _enrich_deezer,
    _enrich_shazam,
)
from sonora.modules.matching import _resolve_album_track_position
from sonora.modules.metadata_consensus import apply_metadata_mapping
from sonora.modules.musicbrainz_enrichment import _enrich_musicbrainz
from sonora.modules.tagger import (
    _AlbumMetadataContext,
    _resolve_album_folder_identity,
    process_single_track,
    tag_album_folder,
)
from sonora.services.shazam import ShazamTrackInfo
from tests.fixtures import write_silent_wav
from tests.module_support import ModuleTestCase


class TestTaggingWorkflows(ModuleTestCase):
    @patch("sonora.audio.metadata.write_track_metadata")
    @patch("sonora.services.lyrics.fetch_synced_lyrics")
    @patch("sonora.services.musicbrainz.fetch_track_mbid")
    @patch("sonora.audio.metadata.read_track_metadata")
    def test_process_single_track(
        self,
        mock_read: Any,
        mock_mbid: Any,
        mock_lyrics: Any,
        mock_write: Any,
    ) -> None:
        wav_file = self.tmp_path / "song.wav"
        write_silent_wav(wav_file)

        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="nane", title="Piesa"
        )
        mock_mbid.return_value = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        mock_lyrics.return_value = "[00:01.00] Vers"

        with (
            patch("sonora.audio.art.process_album_cover_art", return_value=None),
            patch(
                "sonora.services.theaudiodb.fetch_theaudiodb_track_details",
                return_value=None,
            ),
            patch(
                "sonora.services.genius.fetch_genius_song_details", return_value=None
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_track_details", return_value=None
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_album_details", return_value=None
            ),
            patch(
                "sonora.services.musicbrainz.fetch_musicbrainz_recording_details",
                return_value={
                    "artist": "Nane",
                    "title": "Piesa",
                    "track_number": 3,
                    "date": "2021-04-30",
                    "isrc": "ROGRA2101575",
                    "advisory": "Explicit",
                },
            ),
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                return_value="Nane",
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_track_metadata", return_value=None
            ),
        ):
            track_metadata = process_single_track(wav_file, fetch_bpm=False)

        self.assertEqual(track_metadata.artist, "Nane")
        self.assertEqual(
            track_metadata.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        mock_write.assert_called_once()

    @patch("sonora.services.itunes.fetch_itunes_album_details", return_value=None)
    @patch("sonora.modules.tagger.process_single_track")
    def test_tag_album_folder(self, mock_process: Any, mock_itunes: MagicMock) -> None:
        album_dir = self.tmp_path / "album"
        audio_file_1 = album_dir / "01.wav"
        audio_file_2 = album_dir / "02.wav"
        write_silent_wav(audio_file_1)
        write_silent_wav(audio_file_2)

        mock_process.side_effect = [
            TrackInfo(file_path=audio_file_1, artist="Artist", title="T1"),
            TrackInfo(file_path=audio_file_2, artist="Artist", title="T2"),
        ]

        with (
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
        ):
            results = tag_album_folder(album_dir, max_threads=2, fetch_itunes_art=False)
        self.assertEqual(len(results), 2)

    def test_resolve_album_folder_identity(self) -> None:
        album_dir = self.tmp_path / "Artist - Great Album"
        f1 = album_dir / "01 - Song.wav"
        f2 = album_dir / "02 - Other.wav"
        write_silent_wav(f1)
        write_silent_wav(f2)

        with patch("sonora.audio.metadata.read_track_metadata") as mock_read:
            # Track 1 has corrupted metadata (e.g. wrong album/artist)
            mock_read.side_effect = [
                TrackInfo(file_path=f1, artist="Corrupt Artist", album="Corrupt Album"),
                TrackInfo(file_path=f2, artist="Artist", album="Great Album"),
            ]
            artist, album = _resolve_album_folder_identity(album_dir, [f1, f2])
            self.assertEqual(artist, "Artist")
            self.assertEqual(album, "Great Album")

        # Case 2: Multi-disc directory structure: Artist / Album / CD1
        disc_dir = self.tmp_path / "Pink Floyd" / "The Wall" / "CD1"
        f_disc = disc_dir / "01.wav"
        write_silent_wav(f_disc)
        with patch("sonora.audio.metadata.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=f_disc, artist="Pink Floyd", album="The Wall"
            )
            artist, album = _resolve_album_folder_identity(disc_dir, [f_disc])
            self.assertEqual(artist, "Pink Floyd")
            self.assertEqual(album, "The Wall")

        # Case 3: Singles folder should not resolve album as 'Singles'
        singles_dir = self.tmp_path / "3 Doors Down" / "Singles"
        f_single = singles_dir / "01.wav"
        write_silent_wav(f_single)
        with patch("sonora.audio.metadata.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=f_single, artist="3 Doors Down", album="Unknown Album"
            )
            artist, album = _resolve_album_folder_identity(singles_dir, [f_single])
            self.assertEqual(artist, "3 Doors Down")
            self.assertIsNone(album)

    def test_resolve_album_track_position(self) -> None:
        mb_release: dict[str, object] = {
            "title": "That's My Name",
            "album_artist": "Akcent",
            "tracks_by_disc_and_position": {
                (1, 1): {"title": "Kylie", "artist": "Akcent", "isrc": "RO001"},
                (1, 2): {
                    "title": "That's My Name",
                    "artist": "Akcent",
                    "isrc": "RO002",
                },
                (1, 3): {"title": "My Passion", "artist": "Akcent", "isrc": "RO003"},
                (1, 4): {"title": "Kamelia", "artist": "Akcent", "isrc": "RO004"},
                (1, 5): {"title": "Te Quiero", "artist": "Akcent", "isrc": "RO005"},
                (1, 6): {"title": "Amor Gitana", "artist": "Akcent", "isrc": "RO006"},
            },
        }

        # Case 1: Misnumbered file on disk (named 06 - That's My Name.flac, track_number=6)
        # Should resolve to position 2 by title match (and not get poisoned by track 6 Amor Gitana)
        track_misnumbered = TrackInfo(
            file_path=Path("/music/Akcent - That's My Name/06 - That's My Name.flac"),
            artist="Akcent",
            title="That's My Name",
            track_number=6,
            album="That's My Name",
        )
        pos = _resolve_album_track_position(
            track_info=track_misnumbered,
            file_path=track_misnumbered.file_path,
            album_mb_release_details=mb_release,
        )
        self.assertEqual(pos, 2)

        # A stored identifier can fill a missing title, but cannot erase meaningful text.
        track_corrupted = TrackInfo(
            file_path=Path("/music/Akcent - That's My Name/02 - Leaked Audio.flac"),
            artist="Akcent",
            title="Untitled",
            isrc="RO002",
            track_number=2,
            album="That's My Name",
        )
        pos = _resolve_album_track_position(
            track_info=track_corrupted,
            file_path=track_corrupted.file_path,
            album_mb_release_details=mb_release,
        )
        self.assertEqual(pos, 2)

        # Case 3: Alien track with completely different song matches none of the album tracks
        track_alien = TrackInfo(
            file_path=Path(
                "/music/Akcent - That's My Name/07 - Drake - Hotline Bling.flac"
            ),
            artist="Drake",
            title="Hotline Bling",
            track_number=7,
            album="Views",
        )
        pos = _resolve_album_track_position(
            track_info=track_alien,
            file_path=track_alien.file_path,
            album_mb_release_details=mb_release,
        )
        self.assertIsNone(pos)

    def test_enrich_musicbrainz_candidate_rec_mismatch_prevents_metadata_poisoning(
        self,
    ) -> None:
        track = TrackInfo(
            file_path=Path("/music/XXXTENTACION - Revenge/04 - Revenge.flac"),
            artist="XXXTENTACION",
            title="Revenge",
            album="Revenge",
            isrc="USUYG1156895",
            track_number=4,
        )
        mb_release: dict[str, object] = {
            "title": "Revenge",
            "album_artist": "XXXTENTACION",
            "tracks_by_disc_and_position": {
                (1, 4): {
                    "title": "KING",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142049",
                    "recording_mbid": "548be1ad-0000-0000-0000-000000000000",
                    "producers": "King Yosef",
                },
            },
        }
        _enrich_musicbrainz(
            track_info=track,
            album_mbid="9d90908d-0a74-4df0-8e76-13c6473b33e5",
            album_track_mbids={(1, 4): "548be1ad-0000-0000-0000-000000000000"},
            album_mb_release_details=mb_release,
            force=True,
        )
        # Position 4 in album is 'KING'; track is 'Revenge'.
        # Track must retain its own title and ISRC, and must not adopt 'KING' MBID or credits.
        self.assertEqual(track.title, "Revenge")
        self.assertEqual(track.isrc, "USUYG1156895")
        self.assertIsNone(track.musicbrainz_trackid)
        self.assertIsNone(track.producers)

    def test_enrich_musicbrainz_heals_corrupted_title_via_isrc(self) -> None:
        track = TrackInfo(
            file_path=Path("dummy.wav"),
            artist="James Arthur",
            title="Untitled",
            album="Back From the Edge",
            isrc="DEE861600588",
            track_number=4,
            musicbrainz_trackid="56a71671-a4cd-4fe7-96aa-1aadab6cd94e",
        )
        mb_release: dict[str, object] = {
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
        with patch(
            "sonora.services.musicbrainz.fetch_musicbrainz_recording_details",
            return_value={
                "artist": "James Arthur",
                "title": "A Different Recording",
                "isrc": "USAAA2400001",
            },
        ) as stored_lookup:
            _enrich_musicbrainz(
                track_info=track,
                album_mbid="c6b7d374-1425-4414-ba65-2ddcb4d98196",
                album_track_mbids={(1, 4): "c8b03190-306c-4125-9b32-3f9d86d60a12"},
                album_mb_release_details=mb_release,
                force=False,
            )
        stored_lookup.assert_called_once_with("56a71671-a4cd-4fe7-96aa-1aadab6cd94e")
        self.assertEqual(track.title, "Can I Be Him")
        self.assertEqual(
            track.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )

    @patch("sonora.audio.metadata.write_track_metadata")
    @patch("sonora.audio.metadata.read_track_metadata")
    def test_process_single_track_shields_alien_track(
        self, mock_read: Any, mock_write: Any
    ) -> None:
        track_file = self.tmp_path / "03 - Blue Jeans.wav"
        write_silent_wav(track_file)
        mock_read.return_value = TrackInfo(
            file_path=track_file,
            artist="Lana Del Rey",
            title="Blue Jeans",
            album="Born to Die",
            track_number=3,
        )
        mbids = {(1, 1): "uuid-1", (1, 2): "uuid-2", (1, 3): "uuid-3"}
        mb_release: dict[str, object] = {
            "title": "Issa Album",
            "album_artist": "21 Savage",
            "tracks_by_disc_and_position": {
                (1, 1): {"title": "Famous", "artist": "21 Savage"},
                (1, 2): {"title": "Bank Account", "artist": "21 Savage"},
                (1, 3): {"title": "Close My Eyes", "artist": "21 Savage"},
            },
        }
        with (
            patch("sonora.modules.tagger._enrich_musicbrainz"),
            patch("sonora.modules.tagger._enrich_acoustid"),
            patch("sonora.modules.tagger._enrich_deezer"),
            patch("sonora.modules.tagger._enrich_itunes"),
            patch("sonora.modules.tagger._enrich_theaudiodb"),
            patch("sonora.modules.tagger._enrich_genius"),
        ):
            track_metadata = process_single_track(
                track_file,
                album_context=_AlbumMetadataContext(
                    track_mbids=mbids,
                    mb_release_details=mb_release,
                    sample_artist="21 Savage",
                    sample_album="Issa Album",
                ),
                force=True,
                fetch_bpm=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )
            self.assertTrue(track_metadata.is_alien)
            self.assertEqual(track_metadata.artist, "Lana Del Rey")
            self.assertEqual(track_metadata.title, "Blue Jeans")

    def test_tag_album_folder_nonexistent_directory(self) -> None:
        with self.assertRaises(FileNotFoundError):
            tag_album_folder(self.tmp_path / "nonexistent_dir_999")

    @patch("sonora.audio.metadata.write_track_metadata")
    @patch("sonora.services.discogs.search_discogs_release")
    @patch("sonora.services.acoustid.lookup_acoustid")
    @patch("sonora.services.musicbrainz.fetch_track_mbid")
    @patch("sonora.audio.metadata.read_track_metadata")
    def test_process_single_track_acoustid_discogs_fallback(
        self,
        mock_read: Any,
        mock_mbid: Any,
        mock_acoustid: Any,
        mock_discogs: Any,
        mock_write: Any,
    ) -> None:
        wav_file = self.tmp_path / "song.wav"
        write_silent_wav(wav_file)

        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="Artist", title="Title", genre=None
        )
        mock_mbid.return_value = None
        mock_acoustid.return_value = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        mock_discogs.return_value = {"id": 123, "year": 2024}

        with (
            patch("sonora.audio.art.process_album_cover_art", return_value=None),
            patch(
                "sonora.services.theaudiodb.fetch_theaudiodb_track_details",
                return_value=None,
            ),
            patch(
                "sonora.services.genius.fetch_genius_song_details", return_value=None
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_track_details", return_value=None
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.services.musicbrainz.fetch_musicbrainz_recording_details",
                return_value=None,
            ),
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                side_effect=lambda x: str(x).strip(),
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_track_metadata", return_value=None
            ),
        ):
            track_metadata = process_single_track(
                wav_file,
                fetch_bpm=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                acoustid_api_key="acoustid_key",
                discogs_user_token="discogs_token",
            )

        self.assertEqual(
            track_metadata.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        self.assertIsNone(track_metadata.date)
        self.assertNotIn("discogs_release_id", track_metadata.metadata_conflicts)
        mock_acoustid.assert_called_once()
        mock_discogs.assert_called_once()
        mock_write.assert_called_once()

    @patch.dict(os.environ, {"ENABLE_SHAZAM": "true"})
    @patch("sonora.services.shazam.recognize_audio_track")
    def test_enrich_shazam_generic_and_force(self, mock_recognize: MagicMock) -> None:
        audio_file = self.tmp_path / "01.flac"
        audio_file.write_bytes(b"dummy")

        mock_recognize.return_value = ShazamTrackInfo(
            title="Recognized Title",
            artist="Recognized Artist",
            album="Recognized Album",
            genre="Electronic",
            apple_music_id="12345",
            isrc="USABC1234567",
            release_date="2024-01-01",
            label="Recognized Label",
            lyrics="Lyrics Line 1\nLyrics Line 2",
        )

        # Case 1: Generic title and artist gets healed by Shazam
        trk = TrackInfo(file_path=audio_file, title="Track 01", artist="Unknown Artist")
        _enrich_shazam(trk, audio_file, force=False)
        self.assertEqual(trk.title, "Recognized Title")
        self.assertEqual(trk.artist, "Recognized Artist")
        self.assertEqual(trk.album, "Recognized Album")
        self.assertEqual(trk.genre, "Electronic")
        self.assertEqual(trk.itunes_trackid, "12345")
        self.assertEqual(trk.isrc, "USABC1234567")
        self.assertEqual(trk.date, "2024-01-01")
        self.assertEqual(trk.label, "Recognized Label")
        self.assertEqual(trk.lyrics, "Lyrics Line 1\nLyrics Line 2")

        # Case 2: Non-generic title is not overridden when force=False
        trk2 = TrackInfo(
            file_path=audio_file, title="Original Title", artist="Original Artist"
        )
        mock_recognize.reset_mock()
        _enrich_shazam(trk2, audio_file, force=False)
        self.assertEqual(trk2.title, "Original Title")
        self.assertEqual(trk2.artist, "Original Artist")
        mock_recognize.assert_not_called()

        # Forced recognition still requires the existing identity to match.
        _enrich_shazam(trk2, audio_file, force=True)
        self.assertEqual(trk2.title, "Original Title")
        self.assertEqual(trk2.artist, "Original Artist")

    def test_apply_mapping_genre_noise_rejection(self) -> None:
        trk = TrackInfo(
            file_path=Path("/dummy/song.flac"), artist="Artist", title="Title"
        )
        # Spam/noise genres must be rejected and not overwrite genre
        apply_metadata_mapping(trk, {"genre": "Karaoke"}, {"genre": "genre"})
        self.assertEqual(trk.genre, "Karaoke")
        apply_metadata_mapping(trk, {"genre": "Fitness & Workout"}, {"genre": "genre"})
        self.assertEqual(trk.genre, "Karaoke")
        apply_metadata_mapping(trk, {"genre": "Children's Music"}, {"genre": "genre"})
        self.assertEqual(trk.genre, "Karaoke")
        # Valid genres must be normalized and accepted
        apply_metadata_mapping(
            trk, {"genre": "Hip Hop"}, {"genre": "genre"}, force=True
        )
        self.assertEqual(trk.genre, "Hip-Hop/Rap")

    @patch("sonora.modules.tagger._enrich_musicbrainz")
    @patch("sonora.modules.tagger._enrich_acoustid")
    @patch("sonora.modules.tagger._enrich_itunes")
    @patch("sonora.modules.tagger._enrich_deezer")
    @patch("sonora.modules.tagger._enrich_lastfm")
    @patch("sonora.modules.tagger._enrich_discogs")
    @patch("sonora.modules.tagger._enrich_genius")
    @patch("sonora.modules.tagger._enrich_theaudiodb")
    def test_album_context_identity_healing(
        self,
        mock_tadb: MagicMock,
        mock_genius: MagicMock,
        mock_discogs: MagicMock,
        mock_lastfm: MagicMock,
        mock_deezer: MagicMock,
        mock_itunes: MagicMock,
        mock_acoustid: MagicMock,
        mock_mb: MagicMock,
    ) -> None:
        album_dir = self.tmp_path / "Aaryan Shah - The Arrival"
        album_dir.mkdir()
        audio_file = album_dir / "05 - Renegade.wav"
        write_silent_wav(audio_file)

        # Corrupted artist on disk
        trk = TrackInfo(
            file_path=audio_file,
            artist="Unknown Artist",
            title="Renegade",
            track_number=5,
            isrc="TCADM1875890",
        )
        write_track_metadata(trk)

        mock_deezer_details = {
            "title": "The Arrival",
            "tracks_by_disc_and_position": {
                (1, 5): {
                    "title": "Renegade",
                    "artist": "Aaryan Shah",
                    "isrc": "TCADM1875890",
                }
            },
        }

        result = process_single_track(
            file_path=audio_file,
            fetch_bpm=False,
            fetch_key=False,
            fetch_lyrics=False,
            fetch_itunes_art=False,
            album_context=_AlbumMetadataContext(
                deezer_details=mock_deezer_details,
                sample_artist="Aaryan Shah",
                sample_album="The Arrival",
            ),
        )
        # Corrupted artist 'Kavinsky' must be healed to 'Aaryan Shah'
        self.assertEqual(result.artist, "Aaryan Shah")
        self.assertEqual(result.album, "The Arrival")

    def test_enrich_deezer_fallback_preserves_album_track_number(self) -> None:
        """Fallback track search from Deezer (e.g. single track_position=1) must not overwrite album track number."""
        trk = TrackInfo(
            file_path=Path("/music/album/17 - Dip.flac"),
            artist="Tyga",
            title="Dip",
            album="Legendary",
            track_number=17,
            disc_number=1,
        )
        # Album has only track 1 (e.g. standard edition or incomplete match)
        incomplete_album = {
            "title": "Legendary",
            "tracks_by_disc_and_position": {
                (1, 1): {"title": "Too Many", "track_position": 1}
            },
            "tracks_by_title": {"too many": {"title": "Too Many", "track_position": 1}},
        }
        with patch("sonora.services.deezer.fetch_deezer_track_details") as mock_fetch:
            # Fallback single search returns single track with track_position = 1
            mock_fetch.return_value = {
                "title": "Dip",
                "artist": "Tyga",
                "track_position": 1,
                "disk_number": 1,
                "producers": "D.A. Got That Dope",
            }
            _enrich_deezer(
                trk,
                album_deezer_details=incomplete_album,
                has_album_context=True,
                force=True,
            )

        # Track number and disc number must be preserved, while producer credit is enriched
        self.assertEqual(trk.track_number, 17)
        self.assertEqual(trk.disc_number, 1)
        self.assertEqual(trk.producers, "D.A. Got That Dope")

    def test_healer_preserves_credited_collaborative_artist(self) -> None:
        """Legitimate track artist on collaborative/multi-artist album must not be overwritten by album artist."""
        audio_file = self.tmp_path / "2-02 - What's in Your Heart.wav"
        write_silent_wav(audio_file)
        trk = TrackInfo(
            file_path=audio_file,
            artist="Swae Lee",
            title="What's in Your Heart?",
            album="SR3MM",
            track_number=2,
            disc_number=2,
        )
        write_track_metadata(trk)

        mock_mb_release: dict[str, object] = {
            "title": "SR3MM",
            "album_artist": "Rae Sremmurd",
            "tracks_by_disc_and_position": {
                (1, 2): {
                    "title": "What's in Your Heart?",
                    "artist": "Swae Lee",
                }
            },
        }

        with (
            patch("sonora.modules.tagger._enrich_musicbrainz"),
            patch("sonora.modules.tagger._enrich_acoustid"),
            patch("sonora.modules.tagger._enrich_deezer"),
            patch("sonora.modules.tagger._enrich_itunes"),
            patch("sonora.modules.tagger._enrich_theaudiodb"),
            patch("sonora.modules.tagger._enrich_genius"),
        ):
            track_metadata = process_single_track(
                file_path=audio_file,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                album_context=_AlbumMetadataContext(
                    mb_release_details=mock_mb_release,
                    sample_artist="Rae Sremmurd",
                    sample_album="SR3MM",
                ),
                force=True,
            )
        # Swae Lee must be preserved as track artist, Rae Sremmurd set as album artist
        self.assertEqual(track_metadata.artist, "Swae Lee")
        self.assertEqual(track_metadata.album_artist, "Rae Sremmurd")

    def test_album_genre_consensus_harmonization(self) -> None:
        """Album genre consensus must harmonize isolated instrumental/unknown outliers to dominant genre."""
        album_dir = self.tmp_path / "Lewis Capaldi - Divinely Uninspired"
        album_dir.mkdir(parents=True, exist_ok=True)

        f1 = album_dir / "01 - Grace.wav"
        f2 = album_dir / "02 - Bruises.wav"
        f3 = album_dir / "03 - Hold Me While You Wait.wav"
        f4 = album_dir / "04 - Maybe.wav"
        for f in (f1, f2, f3, f4):
            write_silent_wav(f)

        write_track_metadata(
            TrackInfo(
                file_path=f1,
                artist="Lewis Capaldi",
                album="Divinely Uninspired",
                title="Grace",
                genre="Alternative",
                track_number=1,
            )
        )
        write_track_metadata(
            TrackInfo(
                file_path=f2,
                artist="Lewis Capaldi",
                album="Divinely Uninspired",
                title="Bruises",
                genre="Alternative",
                track_number=2,
            )
        )
        write_track_metadata(
            TrackInfo(
                file_path=f3,
                artist="Lewis Capaldi",
                album="Divinely Uninspired",
                title="Hold Me While You Wait",
                genre="Alternative",
                track_number=3,
            )
        )
        # Track 4 is an outlier with 'Instrumental'
        write_track_metadata(
            TrackInfo(
                file_path=f4,
                artist="Lewis Capaldi",
                album="Divinely Uninspired",
                title="Maybe",
                genre="Instrumental",
                track_number=4,
            )
        )

        with (
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
            patch("sonora.audio.art.process_artist_artwork"),
            patch("sonora.audio.replaygain.calculate_album_replaygain"),
            patch("sonora.modules.tagger._enrich_musicbrainz"),
            patch("sonora.modules.tagger._enrich_acoustid"),
            patch("sonora.modules.tagger._enrich_deezer"),
            patch("sonora.modules.tagger._enrich_itunes"),
            patch("sonora.modules.tagger._enrich_theaudiodb"),
            patch("sonora.modules.tagger._enrich_genius"),
        ):
            results = tag_album_folder(
                album_dir,
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )

        # All 4 tracks must now have 'Alternative' genre
        genres = {t.title: t.genre for t in results}
        self.assertEqual(genres["Maybe"], "Alternative")
        self.assertEqual(genres["Grace"], "Alternative")

    def test_album_artist_casing_consensus_harmonization(self) -> None:
        """Album artist and track artist casing must harmonize to canonical consensus."""
        album_dir = self.tmp_path / "Alex - Test Release"
        album_dir.mkdir(parents=True, exist_ok=True)

        track_file1 = album_dir / "01 - Intro.wav"
        track_file2 = album_dir / "02 - Main.wav"
        track_file3 = album_dir / "03 - Outro.wav"
        for track_file in (track_file1, track_file2, track_file3):
            write_silent_wav(track_file)

        write_track_metadata(
            TrackInfo(
                file_path=track_file1,
                artist="Alex",
                album_artist="Alex",
                album="Test Release",
                title="Intro",
                track_number=1,
            )
        )
        write_track_metadata(
            TrackInfo(
                file_path=track_file2,
                artist="ALEX",
                album_artist="Alex",
                album="Test Release",
                title="Main",
                track_number=2,
            )
        )
        write_track_metadata(
            TrackInfo(
                file_path=track_file3,
                artist="Alex",
                album_artist="ALEX",
                album="Test Release",
                title="Outro",
                track_number=3,
            )
        )

        with (
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
            patch("sonora.audio.art.process_artist_artwork"),
            patch("sonora.audio.replaygain.calculate_album_replaygain"),
            patch("sonora.modules.tagger._enrich_musicbrainz"),
            patch("sonora.modules.tagger._enrich_acoustid"),
            patch("sonora.modules.tagger._enrich_deezer"),
            patch("sonora.modules.tagger._enrich_itunes"),
            patch("sonora.modules.tagger._enrich_theaudiodb"),
            patch("sonora.modules.tagger._enrich_genius"),
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                side_effect=lambda name, **kwargs: name,
            ),
        ):
            harmonized_tracks = tag_album_folder(
                album_dir,
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )

        for track_info_result in harmonized_tracks:
            self.assertEqual(track_info_result.artist, "Alex")
            self.assertEqual(track_info_result.album_artist, "Alex")
