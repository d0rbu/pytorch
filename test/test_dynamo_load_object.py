# Owner(s): ["module: dynamo"]
import types
import unittest
from unittest import mock

from torch._dynamo import trace_rules


class TestLoadObject(unittest.TestCase):
    def test_plain_name_loads_before_unwrapping(self):
        value = object()
        events = []

        def unwrap(result):
            events.append(("unwrap", result))
            return result

        def load(name):
            events.append(("load", name))
            trace_rules.unwrap_if_wrapper = unwrap
            return value

        with (
            mock.patch.object(trace_rules, "_load_obj_from_str", load),
            mock.patch.object(trace_rules, "unwrap_if_wrapper"),
        ):
            self.assertIs(trace_rules.load_object("module.function"), value)
        self.assertEqual(events, [("load", "module.function"), ("unwrap", value)])

    def test_member_lookup_and_missing_attribute(self):
        value = object()
        carrier = types.SimpleNamespace(member=value)
        with mock.patch.object(
            trace_rules, "_load_obj_from_str", return_value=carrier
        ) as load:
            self.assertIs(trace_rules.load_object("module.object#member"), value)
            load.assert_called_once_with("module.object")
            self.assertIsNone(trace_rules.load_object("module.object#absent"))

    def test_multiple_separators_retain_assertion(self):
        with mock.patch.object(trace_rules, "_load_obj_from_str") as load:
            with self.assertRaisesRegex(AssertionError, "Invalid obj name a#b#c"):
                trace_rules.load_object("a#b#c")
            load.assert_not_called()

    def test_string_subclass_keeps_split_dispatch(self):
        value = object()
        events = []

        class Name(str):
            __slots__ = ()

            def split(self, separator):
                events.append(separator)
                return ["forced.module", "member"]

            def __contains__(self, item):
                raise AssertionError("Subclass containment was invoked")

        with mock.patch.object(
            trace_rules,
            "_load_obj_from_str",
            return_value=types.SimpleNamespace(member=value),
        ) as load:
            self.assertIs(trace_rules.load_object(Name("visible.module")), value)
            load.assert_called_once_with("forced.module")
        self.assertEqual(events, ["#"])

    def test_non_string_keeps_split_dispatch(self):
        value = object()

        class Name:
            def split(self, separator):
                return ["forced.module"]

        with mock.patch.object(
            trace_rules, "_load_obj_from_str", return_value=value
        ) as load:
            self.assertIs(trace_rules.load_object(Name()), value)
            load.assert_called_once_with("forced.module")
            self.assertIsNone(trace_rules.load_object(None))

    def test_loader_errors(self):
        for error in [AttributeError, ImportError]:
            with self.subTest(error=error):
                with mock.patch.object(
                    trace_rules, "_load_obj_from_str", side_effect=error("load")
                ):
                    self.assertIsNone(trace_rules.load_object("module.function"))
        with mock.patch.object(
            trace_rules, "_load_obj_from_str", side_effect=ValueError("load")
        ):
            with self.assertRaisesRegex(ValueError, "load"):
                trace_rules.load_object("module.function")

    def test_unwrap_errors(self):
        for error in [AttributeError, ImportError]:
            with self.subTest(error=error):
                with (
                    mock.patch.object(trace_rules, "_load_obj_from_str"),
                    mock.patch.object(
                        trace_rules, "unwrap_if_wrapper", side_effect=error("unwrap")
                    ),
                ):
                    self.assertIsNone(trace_rules.load_object("module.function"))
        with (
            mock.patch.object(trace_rules, "_load_obj_from_str"),
            mock.patch.object(
                trace_rules, "unwrap_if_wrapper", side_effect=ValueError("unwrap")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unwrap"):
                trace_rules.load_object("module.function")


if __name__ == "__main__":
    unittest.main()
