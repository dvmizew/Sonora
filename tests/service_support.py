from unittest.mock import patch

from sonora.core.rate_limit import RateLimiter
from tests.support import IsolatedTestCase


class ServiceResponseTestCase(IsolatedTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        RateLimiter.set_disabled(True)

    @classmethod
    def tearDownClass(cls) -> None:
        RateLimiter.set_disabled(False)

    def setUp(self) -> None:
        lrclib_patch = patch("sonora.services.lyrics._query_lrclib", return_value=None)
        lrclib_patch.start()
        self.addCleanup(lrclib_patch.stop)
        services = [
            "sonora.services.itunes",
            "sonora.services.lyrics",
            "sonora.services.musicbrainz",
            "sonora.services.discogs",
            "sonora.services.acoustid",
            "sonora.services.lastfm",
            "sonora.services.genius",
            "sonora.services.deezer",
            "sonora.services.theaudiodb",
            "sonora.services.fanart",
            "sonora.services.shazam",
        ]
        for service in services:
            patcher = patch(f"{service}.get_cached_api", return_value=None)
            patcher.start()
            self.addCleanup(patcher.stop)
