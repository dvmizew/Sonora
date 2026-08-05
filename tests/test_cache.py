"""
Unit tests for diskcache caching layer.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.core.cache import get_cached_api, set_cached_api


class TestDiskCache(unittest.TestCase):
    @patch("sonora.core.cache.get_cache")
    def test_get_cached_api_hit(self, mock_get_cache):
        mock_cache_inst = MagicMock()
        mock_cache_inst.get.return_value = {"title": "Test Album"}
        mock_get_cache.return_value = mock_cache_inst

        val = get_cached_api("test_key")
        self.assertEqual(val, {"title": "Test Album"})
        mock_cache_inst.get.assert_called_once_with("test_key")

    @patch("sonora.core.cache.get_cache")
    def test_set_cached_api(self, mock_get_cache):
        mock_cache_inst = MagicMock()
        mock_get_cache.return_value = mock_cache_inst

        set_cached_api("test_key", {"title": "Test Album"})
        mock_cache_inst.set.assert_called_once_with("test_key", {"title": "Test Album"}, expire=604800)


if __name__ == "__main__":
    unittest.main()
