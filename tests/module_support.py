from unittest.mock import MagicMock, patch

from sonora.core.rate_limit import RateLimiter
from tests.support import IsolatedTestCase


class ModuleTestCase(IsolatedTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        RateLimiter.set_disabled(True)

    @classmethod
    def tearDownClass(cls) -> None:
        RateLimiter.set_disabled(False)

    def setUp(self) -> None:
        artist_lookup = patch(
            "sonora.services.musicbrainz.search_musicbrainz_artists", return_value=[]
        )
        artist_lookup.start()
        self.addCleanup(artist_lookup.stop)
        deezer_artist = patch(
            "sonora.services.artist_resolution.SESSION.get",
            return_value=MagicMock(status_code=404),
        )
        deezer_artist.start()
        self.addCleanup(deezer_artist.stop)
        self.tmp_path = self.test_directory
        patcher = patch("sonora.services.theaudiodb.get_cached_api", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)


class OfflinePipelineTestCase(ModuleTestCase):
    def setUp(self) -> None:
        super().setUp()
        for provider in (
            "sonora.services.musicbrainz.fetch_track_mbid",
            "sonora.services.musicbrainz.search_musicbrainz_release",
            "sonora.services.itunes.fetch_itunes_track_metadata",
            "sonora.services.deezer.fetch_deezer_track_details",
            "sonora.services.deezer.fetch_deezer_album_details",
            "sonora.services.theaudiodb.fetch_theaudiodb_track_details",
        ):
            provider_patch = patch(provider, return_value=None)
            provider_patch.start()
            self.addCleanup(provider_patch.stop)
