# Owner(s): ["module: custom-operators"]
import inspect
import sys
import unittest
from unittest import mock

import torch


class TestLibraryOwnedFake(unittest.TestCase):
    def library(self, namespace):
        library = torch.library.Library(namespace, "FRAGMENT")
        self.addCleanup(library._destroy)
        return library

    def native_definition(self, namespace):
        library = torch._C._dispatch_library("FRAGMENT", namespace, "", __file__, 1)
        self.addCleanup(library.reset)
        library.define("f(Tensor x) -> Tensor", "", ())

    def test_owned_definition_skips_inapplicable_module_lookup(self):
        namespace = "_rmcp_owned_fake_registration"
        library = self.library(namespace)
        library.define("f(Tensor x) -> Tensor")
        with (
            mock.patch.object(inspect, "getmodule", wraps=inspect.getmodule) as lookup,
            mock.patch.object(
                torch.library,
                "_check_pystubs_once",
                wraps=torch.library._check_pystubs_once,
            ) as pystubs,
        ):
            line = sys._getframe().f_lineno + 1
            torch.library.register_fake(
                namespace + "::f", lambda x: torch.empty_like(x), lib=library
            )
        lookup.assert_not_called()
        pystubs.assert_not_called()
        kernel = torch._library.simple_registry.singleton.find(
            namespace + "::f"
        ).fake_impl.kernel
        self.assertEqual(kernel.source, f"{__file__}:{line}")
        self.assertEqual(
            torch.ops._rmcp_owned_fake_registration.f(
                torch.empty(3, device="meta")
            ).shape,
            (3,),
        )
        with self.assertRaisesRegex(RuntimeError, "already has.*fake impl"):
            torch.library.register_fake(namespace + "::f", lambda x: x, lib=library)

    def test_separate_owner_preserves_original_lookup_and_wrapper(self):
        namespace = "_rmcp_separate_fake_registration"
        definition = self.library(namespace)
        definition.define("f(Tensor x) -> Tensor")
        fake = self.library(namespace)
        with (
            mock.patch.object(inspect, "getmodule", wraps=inspect.getmodule) as lookup,
            mock.patch.object(
                torch.library,
                "_check_pystubs_once",
                wraps=torch.library._check_pystubs_once,
            ) as pystubs,
        ):
            torch.library.register_fake(
                namespace + "::f", lambda x: torch.empty_like(x), lib=fake
            )
        self.assertEqual(
            [
                call.args[0]
                for call in lookup.call_args_list
                if inspect.isframe(call.args[0])
            ],
            [sys._getframe()],
        )
        pystubs.assert_called_once()
        self.assertEqual(
            torch.ops._rmcp_separate_fake_registration.f(
                torch.empty(4, device="meta")
            ).shape,
            (4,),
        )

    def test_native_definition_missing_pystub_and_recovery(self):
        namespace = "_rmcp_native_fake_registration"
        self.native_definition(namespace)
        fake = self.library(namespace)
        self.assertFalse(
            torch.ops._rmcp_native_fake_registration.f.default._defined_in_python
        )
        with mock.patch.object(inspect, "getmodule", wraps=inspect.getmodule) as lookup:
            torch.library.register_fake(
                namespace + "::f", lambda x: torch.empty_like(x), lib=fake
            )
        self.assertEqual(
            [
                call.args[0]
                for call in lookup.call_args_list
                if inspect.isframe(call.args[0])
            ],
            [sys._getframe()],
        )
        op = torch.ops._rmcp_native_fake_registration.f
        with (
            mock.patch.object(
                torch._library.utils, "requires_set_python_module", return_value=True
            ),
            self.assertRaisesRegex(RuntimeError, "companion C\\+\\+"),
        ):
            op(torch.empty(4, device="meta"))
        with mock.patch.object(
            torch._library.utils, "requires_set_python_module", return_value=False
        ):
            self.assertEqual(op(torch.empty(4, device="meta")).shape, (4,))

    def test_destroyed_owner_rejects_native_redefinition(self):
        namespace = "_rmcp_closed_fake_registration"
        library = self.library(namespace)
        library.define("f(Tensor x) -> Tensor")
        library._destroy()
        self.native_definition(namespace)
        # Explicit destruction can leave _defs until finalization. A stale global
        # membership or per-library name does not establish a live owner.
        with (
            mock.patch.object(inspect, "getmodule", wraps=inspect.getmodule) as lookup,
            self.assertRaises(AssertionError),
        ):
            torch.library.register_fake(
                namespace + "::f", lambda x: torch.empty_like(x), lib=library
            )
        self.assertEqual(
            [
                call.args[0]
                for call in lookup.call_args_list
                if inspect.isframe(call.args[0])
            ],
            [sys._getframe()],
        )

    def test_native_module_mismatch_remains_an_error(self):
        wrapped = torch.library._check_pystubs_once(
            lambda x: x, "aten::sin", "actual.module"
        )
        with (
            mock.patch.object(
                torch._C, "_dispatch_pystub", return_value=("expected.module", "")
            ),
            self.assertRaisesRegex(RuntimeError, "expected.module.*actual.module"),
        ):
            wrapped(torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
