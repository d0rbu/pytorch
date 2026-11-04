# Owner(s): ["module: custom-operators"]
import inspect
import sys
import unittest
from unittest import mock

import torch


class TestLibraryDeclaringModule(unittest.TestCase):
    def native_definition(self, namespace):
        definition = torch._C._dispatch_library("FRAGMENT", namespace, "", __file__, 1)
        self.addCleanup(definition.reset)
        definition.define("f(Tensor x) -> Tensor", "", ())

    def library(self, namespace):
        library = torch.library.Library(namespace, "FRAGMENT")
        self.addCleanup(library._destroy)
        return library

    def test_explicit_module_keeps_source_wrapper_and_meta_result(self):
        namespace = "_rmcp_declaring_module"
        self.native_definition(namespace)
        library = self.library(namespace)
        with (
            mock.patch.object(
                inspect, "getmodule", side_effect=AssertionError("lookup")
            ),
            mock.patch.object(
                torch.library,
                "_check_pystubs_once",
                wraps=torch.library._check_pystubs_once,
            ) as check,
        ):
            line = sys._getframe().f_lineno + 1
            torch.library.register_fake(
                namespace + "::f",
                lambda x: torch.empty_like(x),
                lib=library,
                _caller_module_name=__name__,
            )
        self.assertEqual(check.call_args.args[2], __name__)
        entry = torch._library.simple_registry.singleton.find(namespace + "::f")
        self.assertEqual(entry.fake_impl.kernel.source, f"{__file__}:{line}")
        op = torch.ops._rmcp_declaring_module.f
        for shape in [(0,), (3, 5), (2, 0, 4)]:
            value = torch.empty(shape, device="meta", dtype=torch.float64)
            result = op(value)
            self.assertEqual(
                (result.shape, result.dtype, result.stride()),
                (value.shape, value.dtype, value.stride()),
            )

    def test_default_native_registration_still_resolves_caller_frame(self):
        namespace = "_rmcp_declaring_default"
        self.native_definition(namespace)
        library = self.library(namespace)
        with mock.patch.object(inspect, "getmodule", wraps=inspect.getmodule) as lookup:
            torch.library.register_fake(
                namespace + "::f", lambda x: torch.empty_like(x), lib=library
            )
        frames = [
            call.args[0]
            for call in lookup.call_args_list
            if inspect.isframe(call.args[0])
        ]
        self.assertEqual(frames, [sys._getframe()])

    def test_default_legacy_library_override_keeps_original_keywords(self):
        calls = []

        class LegacyLibrary(torch.library.Library):
            def _register_fake(
                self, op_name, fn, _stacklevel=1, *, allow_override=False
            ):
                calls.append((op_name, fn, _stacklevel, allow_override))

        library = LegacyLibrary("_rmcp_declaring_legacy", "FRAGMENT")
        self.addCleanup(library._destroy)

        def fn(x):
            return x

        self.assertIs(
            torch.library.register_fake("_rmcp_declaring_legacy::f", fn, lib=library),
            fn,
        )
        self.assertEqual(calls, [("f", fn, 3, False)])

    def test_native_pystub_mismatch_and_missing_errors_match_default(self):
        namespace = "_rmcp_declaring_errors"
        self.native_definition(namespace)
        op = torch.ops._rmcp_declaring_errors.f
        for pystub, required in [(None, True), (("other.module", ""), False)]:
            errors = []
            for explicit in (False, True):
                library = self.library(namespace)
                kwargs = dict(_caller_module_name=__name__) if explicit else {}
                torch.library.register_fake(
                    namespace + "::f",
                    lambda x: torch.empty_like(x),
                    lib=library,
                    **kwargs,
                )
                with (
                    mock.patch.object(
                        torch._C, "_dispatch_pystub", return_value=pystub
                    ),
                    mock.patch.object(
                        torch._library.utils,
                        "requires_set_python_module",
                        return_value=required,
                    ),
                    self.assertRaises(RuntimeError) as error,
                ):
                    op(torch.empty(3, device="meta"))
                errors.append(str(error.exception))
                library._destroy()
            self.assertEqual(errors[0], errors[1])

    def test_explicit_owned_python_definition_keeps_owner_exemption(self):
        namespace = "_rmcp_declaring_owned"
        library = self.library(namespace)
        library.define("f(Tensor x) -> Tensor")
        with (
            mock.patch.object(
                inspect, "getmodule", side_effect=AssertionError("lookup")
            ),
            mock.patch.object(
                torch.library,
                "_check_pystubs_once",
                side_effect=AssertionError("pystub"),
            ),
        ):
            torch.library.register_fake(
                namespace + "::f",
                lambda x: torch.empty_like(x),
                lib=library,
                _caller_module_name=__name__,
            )
        self.assertEqual(
            torch.ops._rmcp_declaring_owned.f(torch.empty(3, device="meta")).shape, (3,)
        )

    def test_explicit_torchvision_module_keeps_exemption(self):
        namespace = "_rmcp_declaring_torchvision"
        self.native_definition(namespace)
        library = self.library(namespace)
        with mock.patch.object(
            torch.library, "_check_pystubs_once", side_effect=AssertionError("pystub")
        ):
            torch.library.register_fake(
                namespace + "::f",
                lambda x: torch.empty_like(x),
                lib=library,
                _caller_module_name="torchvision.example",
            )
        self.assertEqual(
            torch.ops._rmcp_declaring_torchvision.f(
                torch.empty(3, device="meta")
            ).shape,
            (3,),
        )

    def test_explicit_native_cpu_compiled_forward_and_gradient(self):
        namespace = "_rmcp_declaring_square"
        self.native_definition(namespace)
        library = self.library(namespace)
        library.impl("f", lambda x: x * x, "CPU")
        torch.library.register_fake(
            namespace + "::f",
            lambda x: torch.empty_like(x),
            lib=library,
            _caller_module_name=__name__,
        )

        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(inputs[0])

        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * 2 * x

        torch.library.register_autograd(
            namespace + "::f", backward, setup_context=setup_context, lib=library
        )

        def function(x):
            return torch.ops._rmcp_declaring_square.f(x)

        x = torch.linspace(-2, 2, 17, dtype=torch.float64, requires_grad=True)
        compiled = torch.compile(function, backend="eager", fullgraph=True)
        actual = compiled(x)
        torch.testing.assert_close(actual, x * x, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), x)[0], 2 * x, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
