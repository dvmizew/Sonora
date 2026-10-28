from unittest.mock import patch

from sonora.audio import art
from sonora.core.models import TrackInfo
from sonora.core.playlists import update_local_playlists
from tests.fixtures import image_bytes
from tests.support import IsolatedTestCase


class TestAssetRepairRegressions(IsolatedTestCase):
    def test_playlist_recovers_catalog_album_version_suffix(self) -> None:
        target = self.test_directory / "01 - Careless World.flac"
        target.touch()
        playlist = self.test_directory / "Album.m3u"
        playlist.write_text(
            "01 - Tyga - Careless World (Album Version (Explicit)).flac\n"
        )
        track = TrackInfo(target, artist="Tyga", title="Careless World")
        update_local_playlists(self.test_directory, {}, [track])
        self.assertEqual(playlist.read_text(), "01 - Careless World.flac\n")

    def test_playlist_cannot_substitute_a_different_featured_performer(self) -> None:
        target = self.test_directory / "01 - Song.flac"
        target.touch()
        playlist = self.test_directory / "Album.m3u"
        original = b"01 - Performer - Song (feat. Guest A).flac\n"
        playlist.write_bytes(original)
        track = TrackInfo(
            target, artist="Performer", title="Song", featured_artists="Guest B"
        )
        self.assertEqual(update_local_playlists(self.test_directory, {}, [track]), [])
        self.assertEqual(playlist.read_bytes(), original)
        track.featured_artists = "Guest A"
        update_local_playlists(self.test_directory, {}, [track])
        self.assertEqual(playlist.read_bytes(), b"01 - Song.flac\n")

    def test_invalid_album_artist_copies_repair_from_valid_artist_root(self) -> None:
        artist_dir = self.test_directory / "Performer"
        album = artist_dir / "Performer - Album"
        album.mkdir(parents=True)
        for filename in ("artist.jpg", "banner.jpg"):
            (artist_dir / filename).write_bytes(image_bytes())
            (album / filename).write_bytes(b"fakeimage")
        (album / "folder.jpg").write_bytes(b"invalid album cover")
        with patch.object(art, "fetch_artist_images") as fetch:
            art.process_artist_artwork(album, "Performer", dry_run=True)
            self.assertEqual((album / "artist.jpg").read_bytes(), b"fakeimage")
            art.process_artist_artwork(album, "Performer")
        fetch.assert_not_called()
        for filename in ("artist.jpg", "banner.jpg"):
            self.assertEqual((album / filename).read_bytes(), image_bytes())
        self.assertEqual((album / "folder.jpg").read_bytes(), b"invalid album cover")

    def test_invalid_downloads_cannot_replace_existing_invalid_or_valid_assets(
        self,
    ) -> None:
        (self.test_directory / "artist.jpg").write_bytes(b"fakeimage")
        banner = self.test_directory / "banner.jpg"
        banner.write_bytes(image_bytes())
        with patch.object(
            art, "fetch_artist_images", return_value=(b"fakeimage", b"fakeimage")
        ):
            art.process_artist_artwork(self.test_directory, "Performer")
        self.assertEqual(
            (self.test_directory / "artist.jpg").read_bytes(), b"fakeimage"
        )
        self.assertEqual(banner.read_bytes(), image_bytes())

    def test_failed_cover_fetch_does_not_return_corrupt_existing_image(self) -> None:
        cover = self.test_directory / "cover.jpg"
        cover.write_bytes(b"fakeimage")
        with patch.object(art, "_fetch_candidate_artwork", return_value=None):
            self.assertIsNone(
                art.process_album_cover_art(self.test_directory, "Performer", "Album")
            )
        self.assertEqual(cover.read_bytes(), b"fakeimage")

    def test_playlist_repair_preserves_bom_comments_line_endings_and_remote_entries(
        self,
    ) -> None:
        target = self.test_directory / "01 - Song.flac"
        target.touch()
        track = TrackInfo(target, artist="Performer", title="Song")
        playlist = self.test_directory / "Album.m3u8"
        original = b"\xef\xbb\xbf#EXTM3U\r\n#EXTINF:123,Custom description\r\n01 - Performer - Song.flac\r\nhttps://example.com/live\r\n"
        playlist.write_bytes(original)
        self.assertEqual(
            update_local_playlists(self.test_directory, {}, [track], dry_run=True),
            [playlist],
        )
        self.assertEqual(playlist.read_bytes(), original)
        update_local_playlists(self.test_directory, {}, [track])
        self.assertEqual(
            playlist.read_bytes(),
            original.replace(b"01 - Performer - Song.flac", b"01 - Song.flac"),
        )

    def test_playlist_ambiguous_or_different_version_entries_remain_unchanged(
        self,
    ) -> None:
        tracks = []
        for name in ("01 - Song.flac", "02 - Song.flac"):
            path = self.test_directory / name
            path.touch()
            tracks.append(TrackInfo(path, artist="Performer", title="Song"))
        playlist = self.test_directory / "Album.m3u"
        original = b"00 - Performer - Song.flac\n00 - Performer - Song (live).flac\n../outside.flac\n"
        playlist.write_bytes(original)
        self.assertEqual(update_local_playlists(self.test_directory, {}, tracks), [])
        self.assertEqual(playlist.read_bytes(), original)

    def test_playlist_exact_move_updates_relative_location(self) -> None:
        old = self.test_directory / "old.flac"
        target = self.test_directory / "nested" / "new.flac"
        target.parent.mkdir()
        target.touch()
        playlist = self.test_directory / "Album.m3u"
        playlist.write_bytes(b"old.flac\n")
        update_local_playlists(self.test_directory, {old: target})
        self.assertEqual(playlist.read_bytes(), b"nested/new.flac\n")

    def test_playlist_failed_move_does_not_publish_nonexistent_target(self) -> None:
        old = self.test_directory / "old.flac"
        old.touch()
        missing = self.test_directory / "missing.flac"
        playlist = self.test_directory / "Album.m3u"
        playlist.write_bytes(b"old.flac\n")
        self.assertEqual(
            update_local_playlists(self.test_directory, {old: missing}), []
        )
        self.assertEqual(playlist.read_bytes(), b"old.flac\n")

    def test_playlist_non_utf8_and_atomic_failure_preserve_original(self) -> None:
        target = self.test_directory / "new.flac"
        target.touch()
        playlist = self.test_directory / "Album.m3u"
        original = b"#\xff\nold.flac\n"
        playlist.write_bytes(original)
        self.assertEqual(update_local_playlists(self.test_directory, {}), [])
        self.assertEqual(playlist.read_bytes(), original)
        playlist.write_bytes(b"old.flac\n")
        with (
            patch("sonora.core.files.os.fsync", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            update_local_playlists(
                self.test_directory, {self.test_directory / "old.flac": target}
            )
        self.assertEqual(playlist.read_bytes(), b"old.flac\n")
