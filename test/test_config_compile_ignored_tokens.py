# Owner(s): ["module: dynamo"]
import tempfile
import tokenize
import unittest
from pathlib import Path
from types import ModuleType

from torch.utils._config_module import get_assignments_with_compile_ignored_comments


class TestCompileIgnoredTokens(unittest.TestCase):
    def parse(self, source):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config_source.py"
            path.write_text(source, encoding="utf-8")
            module = ModuleType("compile_ignored_source")
            module.__file__ = str(path)
            return get_assignments_with_compile_ignored_comments(module)

    def test_assignment_names_and_marker_positions(self):
        for source, expected in [
            ("\n", set()),
            ("foo: int = 1\n", set()),
            ("# @compile_ignored\nfoo: int = 1\n", {"foo"}),
            ("# @compile_ignored\nfoo = bar = 1\n", {"foo"}),
            ("# @compile_ignored\nfoo = 1", {"foo"}),
            ("# coding: utf-8\n# @compile_ignored\nλ = 'é'\n", {"λ"}),
            ("foo = '@compile_ignored'\n", set()),
            ("# @compile_ignored\nfoo=1\n# @compile_ignored\nbar=2\n", {"foo", "bar"}),
        ]:
            with self.subTest(source=source):
                self.assertEqual(self.parse(source), expected)

    def test_unconsumed_markers_still_fail(self):
        for source in [
            "# @compile_ignored\n",
            "# @compile_ignored\n\nfoo=1\n",
            "# @compile_ignored\n# @compile_ignored\nfoo=1\n",
            "# @compile_ignored\nfoo: int\n",
        ]:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    AssertionError, "unconsumed @compile_ignored"
                ):
                    self.parse(source)

    def test_late_tokenization_failure_still_propagates(self):
        with self.assertRaises(tokenize.TokenError) as caught:
            self.parse("# @compile_ignored\nfoo=1\nbar=(\n")
        self.assertEqual(
            caught.exception.args, ("unexpected EOF in multi-line statement", (3, 0))
        )

    def test_source_lookup_failure_still_propagates(self):
        with self.assertRaisesRegex(OSError, "could not get source code"):
            self.parse("")
        with tempfile.TemporaryDirectory() as temporary:
            module = ModuleType("missing_config_source")
            module.__file__ = str(Path(temporary) / "absent.py")
            with self.assertRaisesRegex(OSError, "source code not available"):
                get_assignments_with_compile_ignored_comments(module)


if __name__ == "__main__":
    unittest.main()
