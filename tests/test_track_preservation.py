from unittest.mock import MagicMock, patch

from PIL import Image

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.models import TrackInfo
from sonora.modules.tagger import (
    _AlbumMetadataContext,
    process_single_track,
)
from tests.fixtures import write_silent_wav
from tests.module_support import OfflinePipelineTestCase


class TestTrackPreservation(OfflinePipelineTestCase):
    def test_process_single_track_rejects_mismatched_release_title_and_artist_sort(
        self,
    ) -> None:
        wav = self.tmp_path / "decebal.wav"
        write_silent_wav(wav)
        track_metadata = read_track_metadata(wav)
        track_metadata.artist = "RAVA"
        track_metadata.album = "LUCIFER"
        track_metadata.musicbrainz_albumid = "a48c628f-1b6e-4b73-b155-1ccb4cfdb7cd"
        write_track_metadata(track_metadata)

        fake_release = {
            "title": "Katcharpari",
            "artist": "Rava",
            "artist_sort": "Rava, Enrico",
            "date": "1973-01-10",
            "release_country": "Italy",
            "musicbrainz_releasegroupid": "f7b54840-2826-4a25-a821-945a9ee09669",
        }
        with patch(
            "sonora.services.musicbrainz.fetch_musicbrainz_release_details",
            return_value=fake_release,
        ):
            processed = process_single_track(
                wav,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                force=True,
                dry_run=True,
                album_context=_AlbumMetadataContext(
                    sample_album="LUCIFER", sample_artist="RAVA"
                ),
            )
            self.assertIsNone(processed.musicbrainz_albumid)
            self.assertIsNone(processed.musicbrainz_releasegroupid)
            self.assertIsNone(processed.artist_sort)
            self.assertNotEqual(processed.date, "1973-01-10")

    def test_process_single_track_rejects_foreign_artist_sort_containment(self) -> None:
        wav = self.tmp_path / "fratemius.wav"
        write_silent_wav(wav)
        track_metadata = read_track_metadata(wav)
        track_metadata.artist = "Samurai"
        track_metadata.album = "Fratemius High"
        track_metadata.musicbrainz_albumid = "fc48d2ff-667d-468a-a5e2-760f8f5a1bcf"
        write_track_metadata(track_metadata)

        fake_release = {
            "title": "Fly High",
            "artist": "A Samurai In Tokyo",
            "artist_sort": "A Samurai In Tokyo",
            "date": "2023-02-22",
            "musicbrainz_releasegroupid": "d0078d28-0a2a-4025-a3a9-81ee90685e6a",
        }
        with patch(
            "sonora.services.musicbrainz.fetch_musicbrainz_release_details",
            return_value=fake_release,
        ):
            processed = process_single_track(
                wav,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                force=True,
                dry_run=True,
                album_context=_AlbumMetadataContext(
                    sample_album="Fratemius High", sample_artist="Samurai"
                ),
            )
            self.assertIsNone(processed.musicbrainz_albumid)
            self.assertIsNone(processed.musicbrainz_releasegroupid)
            self.assertIsNone(processed.artist_sort)

    def test_process_single_track_preserves_valid_inverted_artist_sort(self) -> None:
        wav = self.tmp_path / "posty.wav"
        write_silent_wav(wav)
        track_metadata = read_track_metadata(wav)
        track_metadata.artist = "Post Malone"
        track_metadata.album = "Hollywood's Bleeding"
        track_metadata.musicbrainz_albumid = "11111111-2222-3333-4444-555555555555"
        write_track_metadata(track_metadata)

        fake_release = {
            "title": "Hollywood's Bleeding",
            "artist": "Post Malone",
            "artist_sort": "Malone, Post",
            "date": "2019-09-06",
            "musicbrainz_releasegroupid": "66666666-7777-8888-9999-000000000000",
        }
        with patch(
            "sonora.services.musicbrainz.fetch_musicbrainz_release_details",
            return_value=fake_release,
        ):
            processed = process_single_track(
                wav,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                force=True,
                dry_run=True,
                album_context=_AlbumMetadataContext(
                    sample_album="Hollywood's Bleeding", sample_artist="Post Malone"
                ),
            )
            self.assertEqual(
                processed.musicbrainz_albumid, "11111111-2222-3333-4444-555555555555"
            )
            self.assertEqual(processed.artist_sort, "Malone, Post")

    @patch("sonora.audio.metadata.write_track_metadata")
    def test_process_single_track_triggers_write_on_art_upgrade(
        self, mock_write: MagicMock
    ) -> None:
        wav = self.tmp_path / "lowres_art.wav"
        write_silent_wav(wav)
        track_metadata = read_track_metadata(wav)
        track_metadata.artist = "Artist"
        track_metadata.title = "Title"
        track_metadata.album = "Album"
        track_metadata.art_width = 300
        track_metadata.art_height = 300
        write_track_metadata(track_metadata)

        cover_img = self.tmp_path / "cover.jpg"
        Image.new("RGB", (800, 800), "red").save(cover_img)

        with (
            patch(
                "sonora.services.musicbrainz.fetch_track_mbid",
                return_value=(None, None),
            ),
            patch("sonora.audio.metadata.read_track_metadata") as mock_read,
        ):
            mock_read.return_value = TrackInfo(
                file_path=wav,
                artist="Artist",
                title="Title",
                album="Album",
                album_artist="Artist",
                track_number=1,
                disc_number=1,
                art_width=300,
                art_height=300,
            )
            process_single_track(
                wav,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=True,
                force=False,
                dry_run=False,
                album_context=_AlbumMetadataContext(
                    cover_path=cover_img, sample_album="Album", sample_artist="Artist"
                ),
            )
            self.assertTrue(mock_write.called)
            called_cover_path = mock_write.call_args[1].get("cover_art_path")
            self.assertEqual(called_cover_path, cover_img)
