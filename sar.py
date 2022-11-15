"""
Implementation of different SAR ADC topologies.

...

Classes
-------
BinarySingleEnded
    Single-Ended SAR ADC with binary-weighted capacitors.
    Non-idealities include parasitic capacitances and comparator offset.
"""

import warnings

import numpy as np


class BinarySingleEnded:
    """
    Single-Ended SAR ADC with binary-weighted capacitors.

    ...

    Attributes
    ----------
    n_bits : int
        Bit-resolution of the ADC

    v_ref : float
        Reference voltage of the ADC (default is `1.0`)
        Full scale of the ADC is `0` to `v_ref`.

    nonidealities : dict
        Dictionary containing values of non-idealities
        Following non-idealities are modelled:
            * Parasitic capacitance to ground on the comparator input (default is `0`)
            * Comparator offset voltage (default is `0`)

    capacitances : np.ndarray
        Array of capacitances of the ADC from MSB to LSB

    Methods
    -------
    set_reference_voltage(v_ref: float) -> None
        Sets reference voltage of the ADC

    set_parasitic_cap(parasitic_cap: float) -> None
        Sets parasitic capacitance value at the input to the comparator

    set_comparator_offset(comparator_offset: float) -> None
        Sets offset voltage of the comparator

    digitize(data: np.ndarray) -> np.ndarray
        Generate digital bits from the input data using the ADC
    """

    def __init__(self, n_bits: int, mode: str = "ideal", **kwargs) -> None:
        """
        Initializes the ADC and generates capacitance values.

        Parameters
        ----------
        n_bits : int
            Bit-resolution of the ADC

        mode : str, optional
            Two modes, `ideal` and `nonideal` (default is `ideal`)
            `nonideal` assumes capacitor mismatch.
            Ignored if keyword argument `capacitances` is provided.

            Notes:
                * `ideal` mode is only for capacitances to be binary.
                * `ideal` mode can have parasitic caps and comparator offset.

        Raises
        ------
        NotImplementedError
            If `mode` is not `ideal` or `nonideal`
        """

        # ADC Parameters
        self.n_bits = n_bits
        self.v_ref = kwargs.get("v_ref", 1)

        # Non-idealities
        self.nonidealities = {}
        self.nonidealities["parasitic_cap"] = kwargs.get("parasitic_cap", 0)
        self.nonidealities["comparator_offset"] = kwargs.get("comparator_offset", 0)

        # User-provided capacitances
        if "capacitances" in kwargs:
            self.capacitances = np.copy(kwargs["capacitances"])
            return

        # Generate ideal capacitances
        if mode == "ideal":
            powers = np.insert(np.arange(-1, -n_bits-1, -1), n_bits, -n_bits)
            self.capacitances = 2.0 ** powers

        # Generate non-ideal capacitances
        elif mode == "nonideal":
            mismatch_var = kwargs.get("mismatch_var", 0.05)
            random_values = (np.random.randn(2**n_bits) * mismatch_var) + 1

            sub_binary = kwargs.get("sub_binary", True)
            if sub_binary:
                random_values = np.sort(random_values)[::-1]

            self.capacitances = np.zeros(self.n_bits + 1)
            self.capacitances[0] = random_values[0]
            for i in range(n_bits):
                self.capacitances[i+1] = np.sum(random_values[2**i : 2**(i+1)])

            self.capacitances = self.capacitances[::-1] / self.capacitances.sum()

        else:
            raise NotImplementedError(f"Mode '{mode}' not supported.")

    def set_reference_voltage(self, v_ref: float) -> None:
        """Sets reference voltage of the ADC."""
        self.v_ref = v_ref

    def set_parasitic_cap(self, parasitic_cap: float) -> None:
        """Sets parasitic capacitance value at the input to the comparator."""
        self.nonidealities["parasitic_cap"] = parasitic_cap

    def set_comparator_offset(self, comparator_offset: float) -> None:
        """Sets offset voltage of the comparator."""
        self.nonidealities["comparator_offset"] = comparator_offset

    def digitize(self, data: np.ndarray) -> np.ndarray:
        """
        Generate digital bits from the input data using the ADC.

        Parameters
        ----------
        data : np.ndarray
            Input data to be digitized
            Can be `float` as well.

        Returns
        -------
        np.ndarray
            Array containing the digital codes with data-type `np.float32`
            Adds new axis to the data with bits going from MSB to LSB.
        """

        # Convert data to numpy array and clip it to full-scale
        data = np.array(data)
        clipped_data = np.clip(data, 0, self.v_ref)

        # Check if the input saturates the ADC
        if not np.array_equal(data, clipped_data):
            warnings.warn("Input beyond ADC full-scale, clipping.", RuntimeWarning)

        # Input scaling due to parasitic capacitance
        total_cap = np.sum(self.capacitances) + self.nonidealities["parasitic_cap"]
        data = np.squeeze(data) * np.sum(self.capacitances) / total_cap

        # Bit-cycling loop
        bits = np.zeros((*data.shape, self.n_bits))
        for i in range(self.n_bits):
            bits[..., i] = 1

            voltages = self.v_ref * np.sum(
                np.multiply(bits[..., :i+1], self.capacitances[:i+1]),
                axis=-1
            ) / total_cap

            bits[voltages - data > self.nonidealities["comparator_offset"], i] = 0

        return bits
