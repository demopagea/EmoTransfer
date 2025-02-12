import torch
from torch.nn import functional as F

import numpy as np


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # print('inputs',inputs)
    # print('unnormalized_widths',unnormalized_widths)
    # print('unnormalized_heights',unnormalized_heights)
    # print('unnormalized_derivatives',unnormalized_derivatives)
    # print('min_bin_width',min_bin_width)
    # print('min_bin_height',min_bin_height)
    # print('min_derivative',min_derivative)



    if tails is None:

        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        # print('inputs',inputs)
        # print('inputs.shape',inputs.shape)
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}
    # print('spline_fn inputs')
    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))
    # print('inputs',inputs)
    #print('inside_interval_mask',inside_interval_mask)
    #print('inputs[inside_interval_mask]',inputs[inside_interval_mask])
    #print('inputs[inside_interval_mask].shape',inputs[inside_interval_mask].shape)
    # print('unconstrained_rational_quadratic_spline85')
    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    #print('outside of inputs[inside_interval_mask].shape')
    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    #print('rational_quadratic_spline122')
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")
    #print('rational_quadratic_spline125')

    num_bins = unnormalized_widths.shape[-1]
    #print('rational_quadratic_spline128')

    if min_bin_width * num_bins > 1.0:
        #print('rational_quadratic_spline131')

        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        #print('rational_quadratic_spline135')

        raise ValueError("Minimal bin height too large for the number of bins")
    #print('rational_quadratic_spline138')

    widths = F.softmax(unnormalized_widths, dim=-1)
    #print('rational_quadratic_spline141')

    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    #print('rational_quadratic_spline144')

    cumwidths = torch.cumsum(widths, dim=-1)
    #print('rational_quadratic_spline147')

    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    #print('rational_quadratic_spline150')

    cumwidths = (right - left) * cumwidths + left
    #print('rational_quadratic_spline153')

    cumwidths[..., 0] = left
    #print('rational_quadratic_spline156')

    cumwidths[..., -1] = right
    #print('rational_quadratic_spline159')

    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    #print('rational_quadratic_spline162')

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    #print('rational_quadratic_spline165')

    heights = F.softmax(unnormalized_heights, dim=-1)
    #print('rational_quadratic_spline168')

    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    #print('rational_quadratic_spline171')

    cumheights = torch.cumsum(heights, dim=-1)
    #print('rational_quadratic_spline174')

    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    #print('rational_quadratic_spline177')

    cumheights = (top - bottom) * cumheights + bottom
    #print('rational_quadratic_spline180')

    cumheights[..., 0] = bottom
    #print('rational_quadratic_spline183')

    cumheights[..., -1] = top
    #print('rational_quadratic_spline186')

    heights = cumheights[..., 1:] - cumheights[..., :-1]
    #print('rational_quadratic_spline189')


    if inverse:
        #print('rational_quadratic_spline193')

        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        #print('rational_quadratic_spline196')

        bin_idx = searchsorted(cumwidths, inputs)[..., None]
    #print('rational_quadratic_spline200')

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    #print('rational_quadratic_spline203')

    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    #print('rational_quadratic_spline206')

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    #print('rational_quadratic_spline209')

    delta = heights / widths
    #print('rational_quadratic_spline212')

    input_delta = delta.gather(-1, bin_idx)[..., 0]
    #print('rational_quadratic_spline215')


    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    #print('rational_quadratic_spline219')

    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    #print('rational_quadratic_spline222')

    input_heights = heights.gather(-1, bin_idx)[..., 0]
    #print('rational_quadratic_spline225')
    #print('bin_idx',bin_idx)
    #print('derivatives',derivatives)
    #print('delta',delta)
    #print('heights',heights)


    if inverse:
        #print('rational_quadratic_spline228')
        #print('inputs',inputs)
        #print('input_cumheights',input_cumheights)
        #print('input_derivatives',input_derivatives)
        #print('input_derivatives_plus_one',input_derivatives_plus_one)
        #print('input_delta',input_delta)
        #print('input_heights',input_heights)
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        #print('rational_quadratic_spline233')

        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        #print('rational_quadratic_spline238')

        c = -input_delta * (inputs - input_cumheights)
        #print('rational_quadratic_spline241')
        # #print('b',b)
        # #print('b',b.pow(2))
        # #print('a',a)
        # #print('c',c)

        discriminant = b.pow(2) - 4 * a * c
        #print('rational_quadratic_spline244')
        # #print('discriminant',discriminant)


        assert (discriminant >= 0).all()
        #print('rational_quadratic_spline247')

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        #print('rational_quadratic_spline250')

        outputs = root * input_bin_widths + input_cumwidths
        #print('rational_quadratic_spline253')

        theta_one_minus_theta = root * (1 - root)
        #print('rational_quadratic_spline256')

        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        #print('rational_quadratic_spline262')

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        #print('rational_quadratic_spline269')

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        #print('rational_quadratic_spline272')

        return outputs, -logabsdet
    else:
        #print('rational_quadratic_spline258')

        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        #print('rational_quadratic_spline278')

        return outputs, logabsdet
