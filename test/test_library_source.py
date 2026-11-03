# Owner(s): ["module: custom-operators"]
import gc
import inspect
import linecache
import sys
import types
import unittest
from unittest import mock

import torch
from torch._library.utils import get_source


class TestLibrarySource(unittest.TestCase):
    def test_ordinary_frame_filename_needs_no_source_resolution(self):
        with mock.patch.object(
            inspect, "getsourcefile", wraps=inspect.getsourcefile
        ) as resolver:
            line = sys._getframe().f_lineno + 1
            actual = get_source(1)
        self.assertEqual(actual, f"{__file__}:{line}")
        resolver.assert_not_called()

    def test_no_unused_frame_info(self):
        with mock.patch.object(
            inspect, "getframeinfo", wraps=inspect.getframeinfo
        ) as info:
            line = sys._getframe().f_lineno + 1
            actual = get_source(1)
        self.assertEqual(actual, f"{__file__}:{line}")
        info.assert_not_called()

    def test_releases_frames_and_preserves_resolution_errors(self):
        def retained_helper_frames():
            return {
                id(value)
                for value in gc.get_objects()
                if type(value) is types.FrameType
                and value.f_code is get_source.__code__
            }

        enabled = gc.isenabled()
        before = retained_helper_frames()
        gc.disable()
        try:
            for level in (0, -1):
                for _ in range(16):
                    self.assertTrue(
                        get_source(level).startswith(
                            inspect.getsourcefile(get_source) + ":"
                        )
                    )
            self.assertEqual(retained_helper_frames(), before)
        finally:
            if enabled:
                gc.enable()
        with (
            mock.patch.object(
                inspect, "getsourcefile", side_effect=RuntimeError("source resolution")
            ),
            self.assertRaisesRegex(RuntimeError, "source resolution"),
        ):
            namespace = {"get_source": get_source}
            exec(  # noqa: S102 - Exercise actual bytecode filename resolution.
                compile(
                    "def caller():\n    return get_source(1)\n",
                    "/missing/resolution-error.pyc",
                    "exec",
                ),
                namespace,
            )
            namespace["caller"]()

    def test_locations_and_source_resolution(self):
        line = sys._getframe().f_lineno + 1
        actual = get_source(1)
        self.assertEqual(actual, f"{__file__}:{line}")
        for filename in (
            "<synthetic>",
            "/missing/library-source.py",
            "/missing/library-source.pyc",
            "relative-source.py",
            "/missing/native.so",
            "/missing/library-µ.py",
            "",
        ):
            namespace = {"get_source": get_source}
            exec(  # noqa: S102 - Fixed test code needs a synthetic co_filename.
                compile("def caller():\n    return get_source(1)\n", filename, "exec"),
                namespace,
            )
            self.assertEqual(namespace["caller"](), f"{filename}:2")
        filename = "/cached/library-source.py"
        linecache.cache[filename] = (1, None, ["# cached source\n"], filename)
        try:
            namespace = {"get_source": get_source}
            exec(  # noqa: S102 - Fixed test code needs a .pyc co_filename.
                compile(
                    "def caller():\n    return get_source(1)\n", filename + "c", "exec"
                ),
                namespace,
            )
            self.assertEqual(namespace["caller"](), f"{filename}:2")
        finally:
            linecache.cache.pop(filename, None)

    def test_invalid_stacklevel(self):
        with self.assertRaisesRegex(ValueError, "call stack is not deep enough"):
            get_source(100000)
        for level in (None, 1.5, "1"):
            with self.subTest(level=level), self.assertRaises(TypeError):
                get_source(level)

    def test_no_unused_source_context(self):
        line = sys._getframe().f_lineno + 2
        with mock.patch.object(inspect, "findsource", wraps=inspect.findsource) as find:
            actual = get_source(1)
        self.assertEqual(actual, f"{__file__}:{line}")
        find.assert_not_called()

    def test_fake_registration_source_and_compiled_cpu_gradient(self):
        lib = torch.library.Library("_rmcp_library_source", "FRAGMENT")
        self.addCleanup(lib._destroy)
        lib.define("square(Tensor x) -> Tensor")
        lib.impl("square", lambda x: x * x, "CPU")
        line = sys._getframe().f_lineno + 1
        torch.library.register_fake(
            "_rmcp_library_source::square", lambda x: torch.empty_like(x), lib=lib
        )
        entry = torch._library.simple_registry.singleton.find(
            "_rmcp_library_source::square"
        )
        self.assertEqual(entry.fake_impl.kernel.source, f"{__file__}:{line}")

        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(inputs[0])

        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * 2 * x

        torch.library.register_autograd(
            "_rmcp_library_source::square",
            backward,
            setup_context=setup_context,
            lib=lib,
        )

        def function(x):
            return torch.ops._rmcp_library_source.square(x)

        x = torch.linspace(-2, 2, 17, dtype=torch.float64, requires_grad=True)
        expected = x * x
        compiled = torch.compile(function, backend="eager", fullgraph=True)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.autograd.grad(actual.sum(), x)[0]
        torch.testing.assert_close(gradient, 2 * x, rtol=0, atol=0)
        meta = function(torch.empty(17, device="meta", dtype=torch.float64))
        self.assertEqual(meta.shape, actual.shape)
        self.assertEqual(meta.dtype, actual.dtype)


if __name__ == "__main__":
    unittest.main()
