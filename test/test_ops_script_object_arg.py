import unittest

import torch
from torch._ops import _has_script_object_arg


class Argument:
    def __init__(self, kind):
        self.type = kind


class Schema:
    def __init__(self, arguments):
        self.arguments = arguments


class BrokenArgument:
    @property
    def type(self):
        raise RuntimeError("argument type failure")


class ScriptObjectArgTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ordinary = (
            torch._C.parse_schema("rmcp_example(Tensor value) -> Tensor")
            .arguments[0]
            .type
        )
        cls.script_object = next(
            argument.type
            for schema in torch._C._jit_get_all_schemas()
            for argument in schema.arguments
            if isinstance(argument.type, torch.ClassType)
        )

    def test_empty_schema(self):
        self.assertFalse(_has_script_object_arg(Schema([])))

    def test_ordinary_native_schema(self):
        schema = torch._C.parse_schema(
            "rmcp_example(Tensor value, int index, bool enabled) -> Tensor"
        )
        self.assertFalse(_has_script_object_arg(schema))

    def test_class_type_positions_stop_before_later_failure(self):
        for prefix in [[], [Argument(self.ordinary)], [Argument(self.ordinary)] * 3]:
            with self.subTest(prefix_length=len(prefix)):
                schema = Schema(
                    [*prefix, Argument(self.script_object), BrokenArgument()]
                )
                self.assertTrue(_has_script_object_arg(schema))

    def test_type_failure_before_class_type_propagates(self):
        schema = Schema(
            [Argument(self.ordinary), BrokenArgument(), Argument(self.script_object)]
        )
        with self.assertRaisesRegex(RuntimeError, "argument type failure"):
            _has_script_object_arg(schema)

    def test_iteration_failure_propagates(self):
        class BrokenIterable:
            def __iter__(self):
                raise LookupError("argument iteration failure")

        with self.assertRaisesRegex(LookupError, "argument iteration failure"):
            _has_script_object_arg(Schema(BrokenIterable()))

    def test_argument_type_is_read_once_in_order(self):
        visits = []
        ordinary = self.ordinary
        script_object = self.script_object

        class ObservedArgument:
            def __init__(self, name, kind):
                self.name = name
                self.kind = kind

            @property
            def type(self):
                visits.append(self.name)
                return self.kind

        schema = Schema(
            [
                ObservedArgument("ordinary", ordinary),
                ObservedArgument("class", script_object),
                ObservedArgument("unvisited", ordinary),
            ]
        )
        self.assertTrue(_has_script_object_arg(schema))
        self.assertEqual(visits, ["ordinary", "class"])


if __name__ == "__main__":
    unittest.main()
