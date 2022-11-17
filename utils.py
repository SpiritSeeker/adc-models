"""
Helper module providing utility functions related to ADCs.

Functions
---------
combine_bits(
        bits: np.ndarray, weights: np.ndarray = None
    ) -> np.ndarray
    Combines digital bits with weights to generate waveforms

sine_transition_points(
        data: np.ndarray, n_bits: int, **kwargs
    ) -> np.ndarray
"""

from typing import Tuple

import numpy as np


def combine_bits(bits: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Combines digital bits with weights to generate waveforms.

    Parameters
    ----------
    bits : np.ndarray
        Input digital bits

    weights : np.ndarray, optional
        Weights to be used for combining the digital bits
        Binary weights are used if no weights are provided.
        Bit-resolution of the binary weights is inferred from input data.

    Returns
    -------
    np.ndarray
        Combined waveform
    """

    bits = np.squeeze(np.array(bits))
    bit_precision = bits.shape[-1]

    if weights is None:
        weights = 2.0 ** np.arange(-1, -bit_precision-1, -1)

    return np.sum(bits * weights, axis=-1)

def sine_transition_points(data: np.ndarray, n_bits: int, **kwargs) -> np.ndarray:
    """
    Compute the transition points an ADC from a digitized sine wave.

    Parameters
    ----------
    data : np.ndarray
        Input sine wave

    n_bits : int
        Bit-resolution of the ADC to be assumed

    Returns
    -------
    np.ndarray
        Estimated transition points of the ADC
    """

    data = np.squeeze(np.array(data))

    # Infer range from data if not preovided
    min_value = kwargs.get("min_value", np.min(data))
    max_value = kwargs.get("max_value", np.max(data))

    # Infer amplitude and offset from range
    bin_edges = np.linspace(min_value, max_value, 2**n_bits + 1)
    offset = (min_value + max_value) / 2
    amplitude = (max_value - min_value) / 2

    count, _ = np.histogram(data.flatten(), bin_edges)

    cum_sum = np.cumsum(count)
    transitions = offset - amplitude * np.cos(np.pi * cum_sum / count.sum())

    return transitions

def sine_dnl_inl(
        data: np.ndarray, n_bits: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the DNL and INL of the ADC from a digitzed sine wave.

    Parameters
    ----------
    data : np.ndarray
        Input sine wave

    n_bits : int
        Bit-resolution of the ADC to be assumed

    Returns
    -------
    dnl : np.ndarray
        DNL of valid codes

    inl : np.ndarray
        INL of valid codes
    """

    # Get transition points
    transitions = sine_transition_points(data, n_bits, **kwargs)

    # Find difference in transitions
    linearized_histogram = np.diff(transitions)

    # Remove missing codes in the boundaries
    nonzeros = np.nonzero(linearized_histogram)
    valid_codes = np.arange(nonzeros[0][0], nonzeros[0][-1]+1)
    valid_histogram = linearized_histogram[valid_codes]

    # Find mean LSB value
    lsb = valid_histogram.sum() / valid_histogram.size

    # Calculate DNL and INL
    dnl = np.insert(valid_histogram/lsb - 1, 0, 0)
    inl = np.cumsum(dnl)

    return dnl, inl
