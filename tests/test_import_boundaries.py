import json
import subprocess
import sys
import unittest


class TestImportBoundaries(unittest.TestCase):
    def test_cli_import_does_not_load_audio_providers(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; sys.path[:0] = json.loads(sys.argv[1]); "
                    "import sonora.cli.main; print(json.dumps(sorted(sys.modules)))"
                ),
                json.dumps(sys.path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        imported_modules = set(json.loads(result.stdout))
        self.assertFalse(
            imported_modules.intersection({"scipy", "shazamio", "syncedlyrics"})
        )

    def test_small_modules_do_not_load_network_or_dsp_dependencies(self) -> None:
        for module_name in (
            "sonora.core",
            "sonora.audio",
            "sonora.services",
            "sonora.modules",
            "sonora.core.constants",
            "sonora.core.cancellation",
            "sonora.core.retry",
            "sonora.core.audio_files",
            "sonora.core.library_paths",
            "sonora.core.sidecars",
            "sonora.audio.checksum",
            "sonora.core.artists",
            "sonora.modules.normalization",
        ):
            with self.subTest(module=module_name):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import importlib,json,sys; "
                            "sys.path[:0] = json.loads(sys.argv[2]); "
                            "importlib.import_module(sys.argv[1]); "
                            "print(json.dumps(sorted(sys.modules)))"
                        ),
                        module_name,
                        json.dumps(sys.path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                imported_modules = set(json.loads(result.stdout))
                self.assertFalse(
                    imported_modules.intersection(
                        {"scipy", "shazamio", "httpx", "musicbrainzngs", "diskcache"}
                    ),
                    result.stdout,
                )
