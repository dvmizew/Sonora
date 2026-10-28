from pathlib import Path
from unittest.mock import patch

from sonora.core.config import SonoraConfig
from sonora.core.models import TrackInfo
from sonora.modules import catalog_enrichment as catalog
from sonora.modules.metadata_consensus import (
    MetadataProposal,
    merge_provider_proposals,
)
from sonora.services import deezer, genius, itunes
from tests.support import IsolatedTestCase


class TestReportFollowup(IsolatedTestCase):
    def test_apple_id_accepts_feature_credit_placement_but_rejects_duration_conflict(
        self,
    ) -> None:
        for duration, expected in ((343.0, "123"), (332.0, None)):
            with (
                self.subTest(duration=duration),
                patch.object(
                    itunes,
                    "lookup_itunes_track",
                    return_value={
                        "artist": "Performer",
                        "title": "Long Song (feat. Guest)",
                        "duration": duration,
                        "itunes_trackid": "123",
                    },
                ),
                patch.object(itunes, "fetch_itunes_track_metadata", return_value=None),
            ):
                recording = TrackInfo(
                    Path("song.flac"),
                    artist="Performer feat. Guest",
                    title="Long Song",
                    duration=343.0,
                    itunes_trackid="123",
                )
                catalog._enrich_itunes(recording, force=True)
                self.assertEqual(recording.itunes_trackid, expected)

    def test_single_provider_cannot_append_unverified_artist_aliases(self) -> None:
        recording = TrackInfo(Path("song.flac"), featured_artists="Guest")
        merge_provider_proposals(
            recording,
            [
                MetadataProposal(
                    {"featured_artists": ("Guest, Guest Alias", "Genius")}, {}
                )
            ],
        )
        self.assertEqual(recording.featured_artists, "Guest")
        self.assertIn("featured_artists", recording.metadata_conflicts)

    def test_genius_prose_requires_explicit_opt_in_and_preserves_comments(self) -> None:
        for opted_in, existing, expected in (
            (False, None, None),
            (True, None, "Unverified community prose"),
            (True, "Personal comment", "Personal comment"),
        ):
            with (
                self.subTest(opted_in=opted_in, existing=existing),
                patch.object(
                    catalog,
                    "get_config",
                    return_value=SonoraConfig(import_genius_descriptions=opted_in),
                ),
                patch.object(
                    genius,
                    "fetch_genius_song_details",
                    return_value={
                        "artist": "Artist",
                        "title": "Song",
                        "genius_song_id": "123",
                        "description": "Unverified community prose",
                        "writers": "Writer",
                    },
                ) as lookup,
            ):
                recording = TrackInfo(
                    Path("song.flac"),
                    artist="Artist",
                    title="Song",
                    album="Album",
                    comment=existing,
                )
                catalog._enrich_genius(recording, "fixture")
                self.assertEqual(recording.comment, expected)
                self.assertEqual(recording.composer, "Writer")
                lookup.assert_called_once_with(
                    "Artist", "Song", api_token="fixture", album="Album"
                )

    def test_deezer_does_not_replace_album_editions_under_force(self) -> None:
        for edition in ("Deluxe", "Live", "Radio Edit", "Remixes"):
            with (
                self.subTest(edition=edition),
                patch.object(deezer, "fetch_deezer_track_details", return_value=None),
            ):
                recording = TrackInfo(
                    Path("song.flac"),
                    artist="Artist",
                    title="Song",
                    album=f"Album ({edition})",
                )
                catalog._enrich_deezer(recording, {"title": "Album"}, force=True)
                self.assertEqual(recording.album, f"Album ({edition})")
