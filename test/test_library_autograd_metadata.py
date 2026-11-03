# Owner(s): ["module: custom-operators"]
import unittest
from typing import Optional

import torch


class TestAutogradMetadata(unittest.TestCase):
    def test_independent_operator_and_keyword_contexts(self):
        @torch.library.custom_op("rmcp_metadata_test::linear", mutates_args=())
        def linear(x: torch.Tensor, *, scale: float = 2.0) -> torch.Tensor:
            return x * scale

        @torch.library.custom_op("rmcp_metadata_test::square", mutates_args=())
        def square(x: torch.Tensor, *, scale: float = 3.0) -> torch.Tensor:
            return x * x * scale

        def linear_setup(ctx, inputs, keyword_only_inputs, output):
            ctx.scale = keyword_only_inputs["scale"]

        def square_setup(ctx, inputs, keyword_only_inputs, output):
            ctx.scale = keyword_only_inputs["scale"]
            ctx.save_for_backward(inputs[0])

        def square_backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * 2 * x * ctx.scale

        linear.register_autograd(
            lambda ctx, grad: grad * ctx.scale, setup_context=linear_setup
        )
        square.register_autograd(square_backward, setup_context=square_setup)
        x = torch.tensor([-0.5, 0.25, 1.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0.25, -0.5, 2.0], dtype=torch.float64, requires_grad=True)
        first = linear(x)
        second = square(y, scale=4.0)
        third = linear(x, scale=8.0)
        torch.testing.assert_close(first, x * 2, rtol=0, atol=0)
        torch.testing.assert_close(second, y * y * 4, rtol=0, atol=0)
        torch.testing.assert_close(third, x * 8, rtol=0, atol=0)
        third.sum().backward()
        second.sum().backward()
        first.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 10), rtol=0, atol=0)
        torch.testing.assert_close(y.grad, y * 8, rtol=0, atol=0)
        with torch.no_grad():
            result = square(y)
            self.assertFalse(result.requires_grad)
            torch.testing.assert_close(result, y * y * 3, rtol=0, atol=0)

    def test_tensorlist_and_optional_gradients(self):
        @torch.library.custom_op("rmcp_metadata_test::list_scale", mutates_args=())
        def list_scale(
            xs: list[torch.Tensor], anchor: torch.Tensor
        ) -> list[torch.Tensor]:
            return [x * 2 + anchor * 0 for x in xs]

        def list_setup(ctx, inputs, output):
            ctx.save_for_backward(inputs[1])

        def list_backward(ctx, grads):
            self.assertEqual(ctx.needs_input_grad, ([True, False], True))
            (anchor,) = ctx.saved_tensors
            return [grad * 2 for grad in grads], torch.zeros_like(anchor)

        list_scale.register_autograd(list_backward, setup_context=list_setup)
        x = torch.tensor([0.5, 1.0], requires_grad=True)
        frozen = torch.tensor([1.0, 2.0])
        anchor = torch.ones_like(x, requires_grad=True)
        outputs = list_scale([x, frozen], anchor)
        sum(output.sum() for output in outputs).backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2), rtol=0, atol=0)
        torch.testing.assert_close(
            anchor.grad, torch.zeros_like(anchor), rtol=0, atol=0
        )
        self.assertIsNone(frozen.grad)
        self.assertEqual(list_scale([], anchor), [])

        @torch.library.custom_op("rmcp_metadata_test::optional", mutates_args=())
        def optional(x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
            return x.clone() if y is None else x + y

        def optional_setup(ctx, inputs, output):
            ctx.has_y = inputs[1] is not None

        def optional_backward(ctx, grad):
            return grad, grad if ctx.has_y else None

        optional.register_autograd(optional_backward, setup_context=optional_setup)
        for has_y in [False, True]:
            x = torch.ones(2, requires_grad=True)
            y = torch.ones(2, requires_grad=True) if has_y else None
            optional(x, y).sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
            if y is not None:
                torch.testing.assert_close(y.grad, torch.ones_like(y), rtol=0, atol=0)

    def test_registration_failure_and_recovery(self):
        @torch.library.custom_op("rmcp_metadata_test::recovery", mutates_args=())
        def recovery(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        x = torch.ones(2, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "no autograd formula was registered"):
            recovery(x).sum().backward()
        state = {"fail": True}

        def setup_context(ctx, inputs, output):
            if state["fail"]:
                raise RuntimeError("original setup-context failure")

        recovery.register_autograd(
            lambda ctx, grad: grad * 2, setup_context=setup_context
        )
        with self.assertRaisesRegex(RuntimeError, "original setup-context failure"):
            recovery(x)
        state["fail"] = False
        recovery(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
