from pathlib import Path

from sonora.core.config import (
    SonoraConfig,
    _parse_config_data,
    clear_config_cache,
    get_artist_split_pattern,
    get_balanced_feat_pattern,
    get_bracket_feat_pattern,
    get_config,
    get_disambiguation_pattern,
    get_duplicate_feat_pattern,
    get_feat_tokens_pattern,
)
from sonora.core.models import TrackInfo
from sonora.core.text import (
    clean_unicode_punct,
    normalize_country_name,
    normalize_language_name,
    normalize_legacy_diacritics,
    normalize_script_name,
    preserve_unicode_repertoire,
)
from sonora.core.titles import (
    extract_version_modifier,
    score_artist_title_match,
)
from sonora.core.utils import (
    InterruptedOperationError,
    get_single_release_title,
    is_in_singles_hierarchy,
    is_interruption,
)
from tests.support import IsolatedTestCase


class TestSonoraConfig(IsolatedTestCase):
    def setUp(self) -> None:
        clear_config_cache()

    def tearDown(self) -> None:
        clear_config_cache()

    def test_default_config_properties(self) -> None:
        cfg = get_config()
        self.assertTrue(cfg.is_disc_folder("CD1"))
        self.assertTrue(cfg.is_disc_folder("Disc 2"))
        self.assertTrue(cfg.is_disc_folder("side 1"))
        self.assertFalse(cfg.is_disc_folder("Great Album"))

        self.assertTrue(cfg.is_generic_container("flac"))
        self.assertTrue(cfg.is_generic_container("singles"))
        self.assertTrue(cfg.is_generic_container("downloads"))
        self.assertFalse(cfg.is_generic_container("Pink Floyd"))

    def test_custom_config_instantiation(self) -> None:
        custom_cfg = SonoraConfig(
            disc_folder_patterns=(r"^part\s*\d+$",),
            generic_containers=frozenset({"custom_box"}),
            featuring_conjunctions=("feat", "ft", "avec"),
        )
        self.assertTrue(custom_cfg.is_disc_folder("part 1"))
        self.assertFalse(custom_cfg.is_disc_folder("CD1"))
        self.assertTrue(custom_cfg.is_generic_container("custom_box"))
        self.assertFalse(custom_cfg.is_generic_container("flac"))

    def test_config_parsing_toml_and_json(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            toml_path = tmp_path / "config.toml"
            toml_path.write_text(
                '[sonora]\nartist_match_threshold = 92.5\ndisc_folder_patterns = ["^disc_\\\\d+$"]\n',
                encoding="utf-8",
            )
            parsed_toml = _parse_config_data(toml_path)
            self.assertEqual(parsed_toml.get("artist_match_threshold"), 92.5)
            self.assertEqual(parsed_toml.get("disc_folder_patterns"), ["^disc_\\d+$"])

            json_path = tmp_path / "config.json"
            json_path.write_text(
                '{"artist_match_threshold": 88.0, "featuring_conjunctions": ["feat", "ft"]}',
                encoding="utf-8",
            )
            parsed_json = _parse_config_data(json_path)
            self.assertEqual(parsed_json.get("artist_match_threshold"), 88.0)

            empty_path = tmp_path / "empty.toml"
            empty_path.write_text("", encoding="utf-8")
            self.assertEqual(_parse_config_data(empty_path), {})

            corrupt_path = tmp_path / "corrupt.toml"
            corrupt_path.write_text("invalid [ [ toml", encoding="utf-8")
            self.assertEqual(_parse_config_data(corrupt_path), {})

    def test_dynamic_pattern_generation(self) -> None:
        split_pat = get_artist_split_pattern()
        self.assertEqual(
            split_pat.split("Artist A feat. Artist B"), ["Artist A", "Artist B"]
        )
        self.assertEqual(
            split_pat.split("Artist A & Artist B"), ["Artist A", "Artist B"]
        )

        bracket_pat = get_bracket_feat_pattern()
        self.assertEqual(bracket_pat.sub("", "Title (feat. Artist)").strip(), "Title")
        self.assertEqual(bracket_pat.sub("", "Title [con Artist]").strip(), "Title")

        balanced_pat = get_balanced_feat_pattern()
        self.assertIsNotNone(balanced_pat.match("feat. Someone"))
        self.assertIsNotNone(balanced_pat.match("con Alguien"))
        self.assertIsNone(balanced_pat.match("Random Word"))

        tokens_pat = get_feat_tokens_pattern()
        self.assertEqual(
            [
                t.strip()
                for t in tokens_pat.split("Artist A & Artist B feat. Artist C")
                if t.strip()
            ],
            ["Artist A", "Artist B", "Artist C"],
        )

        disambig_pat = get_disambiguation_pattern()
        self.assertEqual(
            disambig_pat.sub("", "Armin (ROU) feat. ASSAF (ROU)").strip(),
            "Armin feat. ASSAF",
        )

        dup_pat = get_duplicate_feat_pattern()
        self.assertEqual(
            dup_pat.sub("", "Track feat. Artist (feat. Artist)"),
            "Track (feat. Artist)",
        )

    def test_normalize_country_name(self) -> None:
        self.assertEqual(normalize_country_name("US"), "United States")
        self.assertEqual(normalize_country_name("USA"), "United States")
        self.assertEqual(normalize_country_name("RO"), "Romania")
        self.assertEqual(normalize_country_name("ROU"), "Romania")
        self.assertEqual(normalize_country_name("GB"), "United Kingdom")
        self.assertEqual(normalize_country_name("GBR"), "United Kingdom")
        self.assertEqual(normalize_country_name("DE"), "Germany")
        self.assertEqual(normalize_country_name("United States"), "United States")
        self.assertEqual(normalize_country_name("Romania"), "Romania")
        self.assertEqual(normalize_country_name("XW"), "Worldwide")
        self.assertEqual(normalize_country_name("worldwide"), "Worldwide")
        self.assertEqual(normalize_country_name("XE"), "Europe")
        self.assertEqual(
            normalize_country_name("SU"),
            "USSR, Union of Soviet Socialist Republics",
        )
        self.assertEqual(
            normalize_country_name("SUN"),
            "USSR, Union of Soviet Socialist Republics",
        )
        self.assertIsNone(normalize_country_name(None))
        self.assertIsNone(normalize_country_name(""))
        self.assertIsNone(normalize_country_name("   "))
        self.assertEqual(
            normalize_country_name("CustomFictionalCountryXYZ"),
            "CustomFictionalCountryXYZ",
        )

    def test_normalize_language_name(self) -> None:
        self.assertEqual(normalize_language_name("en"), "English")
        self.assertEqual(normalize_language_name("eng"), "English")
        self.assertEqual(normalize_language_name("English"), "English")
        self.assertEqual(normalize_language_name("ro"), "Romanian")
        self.assertEqual(normalize_language_name("ron"), "Romanian")
        self.assertEqual(normalize_language_name("Romanian"), "Romanian")
        self.assertEqual(normalize_language_name("de"), "German")
        self.assertEqual(normalize_language_name("deu"), "German")
        self.assertEqual(normalize_language_name("ger"), "German")
        self.assertEqual(normalize_language_name("ja"), "Japanese")
        self.assertEqual(normalize_language_name("jpn"), "Japanese")
        self.assertIsNone(normalize_language_name(None))
        self.assertIsNone(normalize_language_name(""))
        self.assertIsNone(normalize_language_name("   "))
        self.assertEqual(
            normalize_language_name("KlingonXYZ"),
            "KlingonXYZ",
        )

    def test_normalize_script_name(self) -> None:
        self.assertEqual(normalize_script_name("Latn"), "Latin")
        self.assertEqual(normalize_script_name("latin"), "Latin")
        self.assertEqual(normalize_script_name("Latin"), "Latin")
        self.assertEqual(normalize_script_name("Cyrl"), "Cyrillic")
        self.assertEqual(normalize_script_name("Cyrillic"), "Cyrillic")
        self.assertEqual(normalize_script_name("Arab"), "Arabic")
        self.assertIsNone(normalize_script_name(None))
        self.assertIsNone(normalize_script_name(""))
        self.assertIsNone(normalize_script_name("   "))
        self.assertEqual(normalize_script_name("UnknownScriptXYZ"), "UnknownScriptXYZ")

    def test_fanart_and_shazam_config(self) -> None:
        cfg = SonoraConfig(
            fanart_api_key="test_api_key",
            fanart_client_key="test_client_key",
            enable_shazam=False,
        )
        self.assertEqual(cfg.fanart_api_key, "test_api_key")
        self.assertEqual(cfg.fanart_client_key, "test_client_key")
        self.assertFalse(cfg.enable_shazam)

    def test_clean_unicode_punct_spaces_before_comma(self) -> None:
        self.assertEqual(
            clean_unicode_punct("Stres feat. Anuryh , Boier Bibescu , SHIFT"),
            "Stres feat. Anuryh, Boier Bibescu, SHIFT",
        )
        self.assertEqual(clean_unicode_punct("Artist , Title"), "Artist, Title")
        self.assertEqual(clean_unicode_punct("None"), "None")
        self.assertEqual(clean_unicode_punct(None), "")

    def test_extract_version_modifier(self) -> None:
        self.assertEqual(
            extract_version_modifier("Breaking Me (RetroVision remix)"),
            "retrovision remix",
        )
        self.assertEqual(
            extract_version_modifier("Breaking Me (HUGEL remix)"),
            "hugel remix",
        )
        self.assertEqual(
            extract_version_modifier("Electrified [Extended Version]"),
            "extended version",
        )
        self.assertIsNone(extract_version_modifier("Standard Title"))
        self.assertIsNone(extract_version_modifier("Title (feat. Someone)"))

    def test_preserve_unicode_repertoire(self) -> None:
        self.assertEqual(
            preserve_unicode_repertoire("Bombe în rai", "Bombe in rai"),
            "Bombe în rai",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Bombe in rai", "Bombe în rai"),
            "Bombe în rai",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Andreea Bănică", "Andreea Banica"),
            "Andreea Bănică",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Fără Lacrimi", "Fara Lacrimi"),
            "Fără Lacrimi",
        )
        # International languages (Turkish, Icelandic, German)
        self.assertEqual(
            preserve_unicode_repertoire("Barış Manço", "Baris Manco"),
            "Barış Manço",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Sigur Rós", "Sigur Ros"),
            "Sigur Rós",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Mötley Crüe", "Motley Crue"),
            "Mötley Crüe",
        )
        # Diacritic density: string with more accents preferred over partial
        self.assertEqual(
            preserve_unicode_repertoire("Andreea Bănică", "Andreea Bănica"),
            "Andreea Bănică",
        )
        self.assertEqual(
            preserve_unicode_repertoire("Andreea Bănica", "Andreea Bănică"),
            "Andreea Bănică",
        )
        self.assertEqual(preserve_unicode_repertoire(None, "Candidate"), "Candidate")
        self.assertEqual(preserve_unicode_repertoire("Current", None), "Current")
        self.assertEqual(preserve_unicode_repertoire("Title A", "Title B"), "Title B")

    def test_normalize_legacy_diacritics(self) -> None:
        # T-cedilla is unconditionally normalized to standard comma-below
        self.assertEqual(
            normalize_legacy_diacritics("Ţi-O Dau La Muie"),
            "Ți-O Dau La Muie",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Cât Poţi Tu de Tare"),
            "Cât Poți Tu de Tare",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Mii De Feţe"),
            "Mii De Fețe",
        )
        self.assertEqual(
            normalize_legacy_diacritics("1 TEL. DISTANŢĂ"),
            "1 TEL. DISTANȚĂ",
        )
        self.assertEqual(
            normalize_legacy_diacritics("ROTAŢIE"),
            "ROTAȚIE",
        )

        # Romanian S-cedilla normalized to S-comma below when Romanian markers or words exist
        self.assertEqual(
            normalize_legacy_diacritics("... Şi Cui Îi Pasă?"),
            "... Și Cui Îi Pasă?",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Bag Pula-N Lume Şi V-o Fac Cadou"),
            "Bag Pula-N Lume Și V-o Fac Cadou",
        )
        self.assertEqual(
            normalize_legacy_diacritics("BUCUREŞTI"),
            "BUCUREȘTI",
        )
        self.assertEqual(
            normalize_legacy_diacritics("AM ŞOFER"),
            "AM ȘOFER",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Dani Buşoi"),
            "Dani Bușoi",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Maşina Timpului"),
            "Mașina Timpului",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Leş Elephants Bizarres"),
            "Leș Elephants Bizarres",
        )

        # Turkish S-cedilla is preserved
        self.assertEqual(
            normalize_legacy_diacritics("Barış Manço"),
            "Barış Manço",
        )
        self.assertEqual(
            normalize_legacy_diacritics("Şımarık"),
            "Şımarık",
        )

        # Integration with clean_unicode_punct and preserve_unicode_repertoire
        self.assertEqual(
            clean_unicode_punct("... Şi Cui Îi Pasă?"),
            "... Și Cui Îi Pasă?",
        )
        self.assertEqual(
            preserve_unicode_repertoire("... Si Cui Ii Pasa?", "... Şi Cui Îi Pasă?"),
            "... Și Cui Îi Pasă?",
        )

        # Empty and None handling
        self.assertEqual(normalize_legacy_diacritics(""), "")
        self.assertEqual(normalize_legacy_diacritics(None), "")

    def test_match_score_penalties_and_remixes(self) -> None:
        # Stopword immunity: "The" in band names must not produce high match scores
        self.assertEqual(
            score_artist_title_match(
                "The Motans", "Sangria", "The Re-Tards", "Sangria"
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "The Motans",
                "Sangria",
                "But Then the Self in the Tree Fell Asleep",
                "Sangria",
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "The Motans", "Flagrant", "Audra The Rapper feat. Satasha", "Flagrant"
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "The Motans", "Sativa", "The Body Rampant", "Sativa"
            ),
            0.0,
        )

        # Remix modifier penalty: Different remixers must score 0.0
        self.assertEqual(
            score_artist_title_match(
                "Topic & A7S",
                "Breaking Me (RetroVision remix)",
                "Topic & A7S",
                "Breaking Me (HUGEL remix)",
            ),
            0.0,
        )
        self.assertEqual(
            score_artist_title_match(
                "Topic & A7S",
                "Breaking Me (Riton remix)",
                "Topic & A7S",
                "Breaking Me (HUGEL remix)",
            ),
            0.0,
        )

        # Identical remix modifier should match perfectly
        self.assertGreaterEqual(
            score_artist_title_match(
                "Topic & A7S",
                "Breaking Me (RetroVision remix)",
                "Topic & A7S",
                "Breaking Me (RetroVision remix)",
            ),
            95.0,
        )

    def test_is_interruption(self) -> None:
        self.assertTrue(is_interruption(KeyboardInterrupt()))
        self.assertTrue(is_interruption(InterruptedOperationError()))
        self.assertFalse(is_interruption(RuntimeError("release unlocked lock")))

        nested_exc = RuntimeError("wrapper error")
        nested_exc.__context__ = KeyboardInterrupt()
        self.assertTrue(is_interruption(nested_exc))

        self.assertFalse(is_interruption(RuntimeError("unrelated database error")))
        self.assertFalse(is_interruption(ValueError("invalid value")))
        self.assertFalse(is_interruption(OSError("file not found")))

    def test_is_in_singles_hierarchy(self) -> None:
        self.assertTrue(
            is_in_singles_hierarchy(
                Path("/home/music/Artist/Singles/Artist - Song/01 - Song.flac")
            )
        )
        self.assertTrue(
            is_in_singles_hierarchy(Path("/home/music/Singles/Artist - Song"))
        )
        self.assertTrue(
            is_in_singles_hierarchy(
                Path("/home/music/FLAC/Downloads/song.flac"),
                root_dir=Path("/home/music/FLAC"),
            )
        )
        self.assertFalse(
            is_in_singles_hierarchy(
                Path("/home/music/FLAC/Artist/Album/01 - Song.flac"),
                root_dir=Path("/home/music/FLAC"),
            )
        )
        self.assertFalse(
            is_in_singles_hierarchy(
                Path("/home/music/FLAC/21 Savage/21 Savage - Savage Mode")
            )
        )

    def test_get_single_release_title(self) -> None:
        info1 = TrackInfo(
            file_path=Path("dummy.flac"), title="Minim Doi", album="Track 2"
        )
        self.assertEqual(get_single_release_title(info1), "Minim Doi")

        info2 = TrackInfo(
            file_path=Path("dummy.flac"),
            title="Electrified (extended version)",
            album="O Fortuna (extended version)",
        )
        self.assertEqual(
            get_single_release_title(info2), "Electrified (extended version)"
        )

        info3 = TrackInfo(
            file_path=Path("dummy.flac"),
            title="I Loved You",
            album="I Loved You (Monoir Remix)",
        )
        self.assertEqual(get_single_release_title(info3), "I Loved You (Monoir Remix)")

        info4 = TrackInfo(
            file_path=Path("dummy.flac"),
            title="Tren De Noapte",
            album="Track 2 (feat. Joyce Meyer)",
        )
        self.assertEqual(get_single_release_title(info4), "Tren De Noapte")
