import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sonora.audio.metadata import clear_metadata_cache
from sonora.core.cache import clear_normalization_cache, close_cache, set_ignore_cache
from sonora.core.config import clear_config_cache
from sonora.core.state import reset_library_state


def reject_network(*arguments: object, **keywords: object) -> None:
    raise AssertionError("Unit tests must mock every outbound request")


class IsolatedTestCase(unittest.TestCase):
    def run(
        self, result: unittest.TestResult | None = None
    ) -> unittest.TestResult | None:
        with (
            tempfile.TemporaryDirectory(prefix="sonora-test-") as test_directory,
            contextlib.ExitStack() as resources,
        ):
            self.test_directory = Path(test_directory)
            resources.enter_context(
                patch.object(Path, "home", return_value=self.test_directory / "home")
            )
            resources.enter_context(
                patch.dict(
                    os.environ,
                    {
                        "HOME": str(self.test_directory / "home"),
                        "PYTHONPATH": os.pathsep.join(
                            str(Path(import_path).resolve()) for import_path in sys.path
                        ),
                        "XDG_CACHE_HOME": str(Path(test_directory) / "cache"),
                        "XDG_CONFIG_HOME": str(Path(test_directory) / "config"),
                        "ENABLE_SHAZAM": "false",
                    },
                    clear=True,
                )
            )
            resources.enter_context(
                patch("sonora.core.config.load_dotenv", return_value=False)
            )
            for boundary in (
                "httpx.HTTPTransport.handle_request",
                "requests.sessions.Session.send",
                "urllib.request.OpenerDirector.open",
                "aiohttp.ClientSession._request",
                "shazamio.Shazam.recognize",
            ):
                resources.enter_context(patch(boundary, side_effect=reject_network))
            reset_library_state()
            close_cache()
            clear_config_cache()
            clear_normalization_cache()
            clear_metadata_cache()
            set_ignore_cache(False)
            try:
                return super().run(result)
            finally:
                reset_library_state()
                close_cache()
                clear_config_cache()
                clear_normalization_cache()
                clear_metadata_cache()
                set_ignore_cache(False)
