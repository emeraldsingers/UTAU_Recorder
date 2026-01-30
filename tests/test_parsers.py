import unittest
from pathlib import Path

from models.parsers import parse_reclist_text, parse_reclist_line
from models.voicebank import parse_oto_ini


class TestParsers(unittest.TestCase):
    def test_parse_reclist_line(self):
        self.assertEqual(parse_reclist_line("a"), ("a", None))
        self.assertEqual(parse_reclist_line("a\tC4"), ("a", "C4"))
        self.assertEqual(parse_reclist_line("a, C#3"), ("a", "C#3"))
        self.assertIsNone(parse_reclist_line("# comment"))

    def test_parse_reclist_text(self):
        text = "a\n#comment\nb\tD4\n"
        items = parse_reclist_text(text)
        self.assertEqual(items, [("a", None), ("b", "D4")])

    def test_parse_oto_ini(self):
        content = "a.wav=alias,0,0,0,0,0\n"
        path = Path("/tmp/oto.ini")
        path.write_text(content, encoding="utf-8")
        aliases = parse_oto_ini(path)
        self.assertEqual(aliases, [("alias", "a.wav")])


if __name__ == "__main__":
    unittest.main()
