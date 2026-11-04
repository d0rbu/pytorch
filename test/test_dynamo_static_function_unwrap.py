# Owner(s): ["module: dynamo"]
import functools
import inspect
import unittest

from torch._dynamo.utils import (
    unwrap_if_wrapper,
    unwrap_with_attr_name_if_wrapper,
)


class TestStaticFunctionUnwrap(unittest.TestCase):
    def check_result(self, function, expected, attribute):
        actual, actual_attribute = unwrap_with_attr_name_if_wrapper(function)
        self.assertIs(actual, expected)
        self.assertEqual(actual_attribute, attribute)
        self.assertIs(unwrap_if_wrapper(function), expected)

    def test_plain_function_and_false_values(self):
        def function():
            pass

        self.check_result(function, function, None)
        for value in [None, False, 0, "", [], {}]:
            with self.subTest(value=value):
                function._torchdynamo_inline = value
                self.check_result(function, function, None)

    def test_truthy_values_and_static_descriptors(self):
        def function():
            pass

        for value in [object(), True, "wrapped", len, property(lambda _: None)]:
            with self.subTest(value=type(value).__name__):
                function._torchdynamo_inline = value
                self.check_result(function, value, "_torchdynamo_inline")

    def test_dict_subclass_does_not_dispatch_overridden_get(self):
        class Dictionary(dict):
            def __contains__(self, key):
                raise AssertionError("Overridden dictionary contains was invoked")

            def get(self, *args):
                raise AssertionError("Overridden dictionary get was invoked")

        def function():
            pass

        function.__dict__ = Dictionary()
        self.check_result(function, function, None)
        expected = object()
        function.__dict__ = Dictionary(_torchdynamo_inline=expected)
        self.check_result(function, expected, "_torchdynamo_inline")

    def test_truth_test_mutation_repeats_lookup(self):
        def function():
            pass

        for action in ["delete", "replace_dictionary", "replace_value"]:
            with self.subTest(action=action):
                replacement = object()

                class Change:
                    def __bool__(self):
                        if action == "delete":
                            del function._torchdynamo_inline
                        elif action == "replace_dictionary":
                            function.__dict__ = {"_torchdynamo_inline": replacement}
                        else:
                            function._torchdynamo_inline = replacement
                        return True

                function._torchdynamo_inline = Change()
                actual, attribute = unwrap_with_attr_name_if_wrapper(function)
                self.assertIs(actual, function if action == "delete" else replacement)
                self.assertEqual(attribute, "_torchdynamo_inline")

    def test_truth_test_exception_propagates(self):
        def function():
            pass

        class TruthError:
            def __bool__(self):
                raise RuntimeError("truth test failed")

        function._torchdynamo_inline = TruthError()
        with self.assertRaisesRegex(RuntimeError, "truth test failed"):
            unwrap_with_attr_name_if_wrapper(function)

    def test_inspect_sentinel_retains_static_lookup_default(self):
        def function():
            pass

        function._torchdynamo_inline = inspect._sentinel
        self.check_result(function, function, None)

    def test_truth_test_can_replace_value_with_inspect_sentinel(self):
        def function():
            pass

        class Change:
            def __bool__(self):
                function._torchdynamo_inline = inspect._sentinel
                return True

        function._torchdynamo_inline = Change()
        actual, attribute = unwrap_with_attr_name_if_wrapper(function)
        self.assertIs(actual, function)
        self.assertEqual(attribute, "_torchdynamo_inline")

    def test_other_object_kinds_retain_original_behavior(self):
        class Callable:
            @property
            def _torchdynamo_inline(self):
                raise AssertionError("A dynamic descriptor was invoked")

            def __call__(self):
                pass

        for value in [
            len,
            str.upper,
            object.__repr__,
            object(),
            type,
            None,
            Callable(),
        ]:
            with self.subTest(kind=type(value).__name__):
                self.check_result(value, value, None)

        @functools.lru_cache
        def cached():
            pass

        self.check_result(cached, cached, None)


if __name__ == "__main__":
    unittest.main()
