# -*- coding: utf-8 -*-
"""
Adapted from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L124
"""

import torch
from typing import Callable, Iterable, Sequence, Union


def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
    use_deepspeed: bool = False
):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    :param use_deepspeed: if True, use deepspeed
    """
    if flag:
        if use_deepspeed:
            import deepspeed
            return deepspeed.checkpointing.checkpoint(func, *inputs)

        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, *output_grads):
        detached_inputs = []
        shallow_copies = []
        grad_input_tensors = []
        for x in ctx.input_tensors:
            if x is None:
                detached_inputs.append(None)
                shallow_copies.append(None)
            else:
                detached = x.detach().requires_grad_(True)
                detached_inputs.append(detached)
                shallow_copies.append(detached.view_as(detached))
                grad_input_tensors.append(detached)
        ctx.input_tensors = detached_inputs
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            output_tensors = ctx.run_function(*shallow_copies)
        grads = torch.autograd.grad(
            output_tensors,
            grad_input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        tensor_input_grads = grads[:len(grad_input_tensors)]
        param_grads = grads[len(grad_input_tensors):]
        tensor_iter = iter(tensor_input_grads)
        expanded_input_grads = []
        for x in ctx.input_tensors:
            expanded_input_grads.append(None if x is None else next(tensor_iter))
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + tuple(expanded_input_grads) + tuple(param_grads)
