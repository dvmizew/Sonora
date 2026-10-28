from pathlib import Path
from unittest.mock import patch

from sonora.core.models import TrackInfo
from sonora.modules import catalog_metadata
from sonora.services import genius
from tests.support import IsolatedTestCase


class TestGeniusLegacyMetadata(IsolatedTestCase):
    def test_repaired_identifier_reports_stale_legacy_fields_without_erasing_them(
        self,
    ) -> None:
        track = TrackInfo(
            Path("introduction.flac"),
            artist="Performer",
            title="Introduction",
            album="Second Album",
            genius_song_id="1",
            comment="Description of the first album",
            composer="Old writer",
            lyricist="Same writer",
            producers="Verified producer",
            featured_artists="Verified guest",
            metadata_sources={
                "composer": "Genius",
                "producers": "MusicBrainz",
                "featured_artists": "Deezer",
            },
        )
        with patch.object(
            genius,
            "fetch_genius_song_details",
            return_value={
                "genius_song_id": "2",
                "writers": "Same writer",
                "description": "Description of the second album",
                "producers": "Another producer",
                "featured_artists": "Another guest",
            },
        ):
            catalog_metadata._update_genius(track, "fixture")
        self.assertEqual(track.genius_song_id, "2")
        self.assertEqual(track.comment, "Description of the first album")
        self.assertEqual(track.composer, "Old writer")
        self.assertIn("comment", track.metadata_conflicts)
        self.assertIn("composer", track.metadata_conflicts)
        self.assertNotIn("lyricist", track.metadata_conflicts)
        self.assertNotIn("producers", track.metadata_conflicts)
        self.assertNotIn("featured_artists", track.metadata_conflicts)

    def test_same_identifier_does_not_flag_existing_personal_comment(self) -> None:
        track = TrackInfo(
            Path("song.flac"), genius_song_id="2", comment="Personal comment"
        )
        with patch.object(
            genius,
            "fetch_genius_song_details",
            return_value={"genius_song_id": "2", "description": "Community text"},
        ):
            catalog_metadata._update_genius(track, "fixture")
        self.assertEqual(track.comment, "Personal comment")
        self.assertNotIn("comment", track.metadata_conflicts)
