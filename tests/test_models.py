from pathlib import Path

from sonora.core.models import TrackInfo
from tests.support import IsolatedTestCase


class TestCoreModels(IsolatedTestCase):
    def test_track_info_serialization(self) -> None:
        track = TrackInfo(
            file_path=Path("/music/song.flac"),
            artist="Beyoncé",
            title="HALO",
            album="I Am... Sasha Fierce",
            track_number=1,
            bpm=120.0,
        )
        serialized_track = track.to_dict()
        self.assertEqual(serialized_track["artist"], "Beyoncé")
        self.assertEqual(serialized_track["title"], "HALO")
        self.assertEqual(serialized_track["track_number"], 1)
        self.assertEqual(serialized_track["bpm"], 120.0)

        empty_track = TrackInfo(file_path=Path("/music/empty.flac"))
        empty_data = empty_track.to_dict()
        self.assertEqual(empty_data["artist"], "Unknown Artist")
        self.assertIsNone(empty_data["track_number"])
        self.assertIsNone(empty_data["bpm"])
