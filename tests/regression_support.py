from pathlib import Path

import numpy as np
import soundfile

from tests.support import IsolatedTestCase


class AudioRegressionTestCase(IsolatedTestCase):
    def setUp(self) -> None:
        self.root = self.test_directory / "fixture"
        self.root.mkdir()

    def audio(self, name: str) -> Path:
        audio_path = self.root / name
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(audio_path, np.zeros(44100), 44100)
        return audio_path
