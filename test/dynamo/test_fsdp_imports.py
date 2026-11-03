# Owner(s): ["module: dynamo"]
import builtins
import subprocess
import sys
import unittest
from unittest import mock

import torch


class FsdpImportTests(unittest.TestCase):
    def test_dynamo_import_does_not_load_fsdp(self):
        code = """\
import sys
import torch._dynamo
from torch._dynamo.variables import functions
from torch._dynamo.variables import torch as torch_variables
assert 'torch.distributed.fsdp' not in sys.modules
assert functions._get_fsdp_param_group.cache_info().misses == 0
assert torch_variables._get_fsdp_param_group.cache_info().misses == 0
"""
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            timeout=60,
        )

    def _modules(self):
        from torch._dynamo.variables import functions
        from torch._dynamo.variables import torch as torch_variables

        return functions, torch_variables

    def _fresh_getter(self, module):
        getter = module._get_fsdp_param_group
        getter.cache_clear()
        self.addCleanup(getter.cache_clear)
        return getter

    def test_optional_module_is_cached_independently(self):
        getters = [self._fresh_getter(module) for module in self._modules()]
        original_import = builtins.__import__
        attempts = []

        def missing(name, *args, **kwargs):
            if name == "torch.distributed.fsdp._fully_shard":
                attempts.append(name)
                raise ModuleNotFoundError("optional FSDP unavailable")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=missing):
            for getter in getters:
                self.assertIsNone(getter())
                self.assertIsNone(getter())
        self.assertEqual(len(attempts), 2)
        for getter in getters:
            self.assertIsNone(getter())
            self.assertEqual(getter.cache_info().misses, 1)

    def test_other_import_errors_keep_identity_and_are_not_cached(self):
        original_import = builtins.__import__
        for module in self._modules():
            for error in (
                ImportError("broken FSDP"),
                RuntimeError("broken dependency"),
            ):
                getter = self._fresh_getter(module)

                def broken(name, *args, import_error=error, **kwargs):
                    if name == "torch.distributed.fsdp._fully_shard":
                        raise import_error
                    return original_import(name, *args, **kwargs)

                with mock.patch("builtins.__import__", side_effect=broken):
                    for _ in range(2):
                        with self.assertRaises(type(error)) as caught:
                            getter()
                        self.assertIs(caught.exception, error)
                self.assertEqual(getter.cache_info().currsize, 0)

    def test_module_cache_keeps_function_identity_and_live_attributes(self):
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group

        for module in self._modules():
            getter = self._fresh_getter(module)
            self.assertIs(getter(), _fsdp_param_group)
            original = _fsdp_param_group.FSDPParamGroup.use_training_state
            replacement = lambda: None
            with mock.patch.object(
                _fsdp_param_group.FSDPParamGroup,
                "use_training_state",
                replacement,
            ):
                self.assertIs(getter().FSDPParamGroup.use_training_state, replacement)
            self.assertIs(getter().FSDPParamGroup.use_training_state, original)

    def test_compiled_bound_and_unbound_training_state(self):
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

        group = FSDPParamGroup(
            [],
            (torch.nn.Linear(1, 1),),
            None,
            None,
            torch.device("cpu"),
            None,
            None,
            None,
        )

        def bound(x):
            with group.use_training_state(TrainingState.FORWARD):
                if group._training_state == TrainingState.FORWARD:
                    return x.sin() * 2
                return x.cos() * 3

        def unbound(x):
            with FSDPParamGroup.use_training_state(group, TrainingState.PRE_BACKWARD):
                if group._training_state == TrainingState.PRE_BACKWARD:
                    return x.cos() * 2
                return x.sin() * 3

        for function in (bound, unbound):
            torch._dynamo.reset()
            x = torch.linspace(-2, 2, 17, dtype=torch.float64, requires_grad=True)
            expected = function(x)
            expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
            compiled = torch.compile(function, backend="eager", fullgraph=True)
            actual = compiled(x)
            actual_gradient = torch.autograd.grad(actual.sum(), x)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=0, atol=0
            )
            self.assertEqual(group._training_state, TrainingState.IDLE)

    def test_reload_resets_each_consumer_cache(self):
        code = """\
import importlib
import torch._dynamo
from torch._dynamo.variables import functions
from torch._dynamo.variables import torch as torch_variables
from torch.distributed.fsdp._fully_shard import _fsdp_param_group
for module in (functions, torch_variables):
    getter = module._get_fsdp_param_group
    assert getter() is _fsdp_param_group
    importlib.reload(module)
    assert module._get_fsdp_param_group is not getter
    assert module._get_fsdp_param_group.cache_info().misses == 0
    assert module._get_fsdp_param_group() is _fsdp_param_group
"""
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            timeout=60,
        )


if __name__ == "__main__":
    unittest.main()
