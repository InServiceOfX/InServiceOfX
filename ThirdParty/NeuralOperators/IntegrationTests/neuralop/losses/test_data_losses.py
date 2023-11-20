from neuralop.losses.data_losses import (
	central_diff_1d,
	central_diff_2d)

import pytest
import torch
import torch.nn as nn

def test_torch_roll_works():
	"""
	https://pytorch.org/docs/stable/generated/torch.roll.html
	torch.roll(input, shifts, dims=None) -> Tensor
	Roll tensor input along given dimension(s). Elements shifted beyond last
	position are reintroduced at first position. If dims is None, tensor will be
	flattened before rolling and then restored to original shape.
	"""
	x = torch.tensor([i for i in range(1, 9)]).view(4, 2)
	for i in range(4):
		for j in range(2):
			assert x[i, j] == i * 2 + j + 1

	y = torch.roll(x, 1)
	for i in range(4):
		for j in range(2):
			if (i != 0 and j != 0):
				assert y[i, j] == i * 2 + j
			else:
				assert y[0, 0] == 8



def test_torch_roll_itself_does_not_mutate():
	x = torch.tensor([i for i in range(1, 9)]).view(4, 2)
	y = torch.roll(x, 1, 0)

	for i in range(4):
		for j in range(2):
			assert x[i, j] == i * 2 + j + 1
			if (i > 0):
				assert y[i, j] == (i - 1) * 2 + j + 1
			else:
				assert y[i, j] == j + 7

	y = torch.roll(x, -1, 0)
	for i in range(4):
		for j in range(2):
			assert x[i, j] == i * 2 + j + 1
			if (i < 3):
				assert y[i, j] == i * 2 + j + 3
			else:
				assert y[i, j] == j + 1

	y = torch.roll(x, shifts=(2, 1), dims=(0, 1))
	for i in range(4):
		for j in range(2):
			assert x[i, j] == i * 2 + j + 1
			if (i < 2):
				assert y[i, j] == i * 2 - j + 6
			else:
				assert y[i, j] == (i - 2) * 2 - j + 2

def test_torch_l1loss_constructs():
	"""
	https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss

	torch.nn.L1Loss(size_average=None,reduce=None,reduction='mean')
	Creates a criterion that measures mean absolute error (MAE) between each
	element in input x and target y.

	Output: scalar. If reduction is 'none', then (*), same shape as input.
	"""
	loss = nn.L1Loss()
	# requires_grad=True tells PyTorch to keep track of operations performed on
	# that tensor so gradients can be calculated with respect to it.
	input = torch.randn(3, 5, requires_grad=True)
	target = torch.randn(3, 5)
	output = loss(input, target)
	output.backward()

	input = torch.tensor([
		[float(i + 1 + j * 5) for i in range(5)] for j in range(3)],
		requires_grad=True)

	target = torch.tensor(
		[[float(5 * (j + 1) - i) for i in range(5)] for j in range(3)])

	output = loss(input, target)
	assert output.item() == pytest.approx(2.4)