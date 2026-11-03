from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.modules.backup import backup_library_tags, restore_library_tags
from tests.fixtures import write_silent_wav
from tests.module_support import ModuleTestCase


class TestBackupWorkflows(ModuleTestCase):
    def test_backup_and_restore(self) -> None:
        wav_path = self.tmp_path / "song.wav"
        write_silent_wav(wav_path)

        initial_info = read_track_metadata(wav_path)
        initial_info.artist = "Test Artist"
        write_track_metadata(initial_info)

        backup_file = self.tmp_path / "backup.json"
        backup_library_tags(self.tmp_path, output_file=backup_file)
        self.assertTrue(backup_file.exists())

        # Modify track info
        track_metadata = read_track_metadata(wav_path)
        track_metadata.artist = "Modified Artist"
        write_track_metadata(track_metadata)

        reloaded = read_track_metadata(wav_path)
        self.assertEqual(reloaded.artist, "Modified Artist")

        # Restore
        restored_count = restore_library_tags(backup_file)
        self.assertEqual(restored_count, 1)

        restored_info = read_track_metadata(wav_path)
        self.assertEqual(restored_info.artist, "Test Artist")

    def test_backup_and_restore_gzipped(self) -> None:
        wav_path = self.tmp_path / "song_gz.wav"
        write_silent_wav(wav_path)

        initial_info = read_track_metadata(wav_path)
        initial_info.artist = "Gzip Artist"
        write_track_metadata(initial_info)

        gz_backup = self.tmp_path / "backup.json.gz"
        backup_library_tags(self.tmp_path, output_file=gz_backup)
        self.assertTrue(gz_backup.exists())

        # Modify
        initial_info.artist = "Changed Artist"
        write_track_metadata(initial_info)

        # Restore from gzip
        restored = restore_library_tags(gz_backup)
        self.assertGreaterEqual(restored, 1)

        reloaded = read_track_metadata(wav_path)
        self.assertEqual(reloaded.artist, "Gzip Artist")

    def test_backup_and_restore_portable_relocation(self) -> None:
        old_dir = self.tmp_path / "old_location"
        old_dir.mkdir()
        wav_path = old_dir / "portable.wav"
        write_silent_wav(wav_path)

        track_metadata = read_track_metadata(wav_path)
        track_metadata.artist = "Original Artist"
        write_track_metadata(track_metadata)

        backup_file = self.tmp_path / "portable_backup.json"
        backup_library_tags(old_dir, output_file=backup_file)

        # Now move the audio file to a brand new folder (simulating disk move/NAS mount change)
        new_dir = self.tmp_path / "new_location"
        new_dir.mkdir()
        new_wav = new_dir / "portable.wav"
        wav_path.rename(new_wav)

        # Modify tags in new location
        mod_info = read_track_metadata(new_wav)
        mod_info.artist = "Altered Artist"
        write_track_metadata(mod_info)

        # Restore pointing target_directory to new_dir
        restored = restore_library_tags(backup_file, target_directory=new_dir)
        self.assertEqual(restored, 1)

        final_info = read_track_metadata(new_wav)
        self.assertEqual(final_info.artist, "Original Artist")
