from enum import Enum


class Order(Enum):
    """Enum for memory order for raw traces in signal block
    """
    C = 1  # C order
    F = 2  # Fortran order
    K = 3  # Original order


class SampleIndex(int):
    """Phantom type for sample index in signal block
    """

    def __add__(self, a: int):
        return SampleIndex(self + a)

    def __sub__(self, a: int):
        return SampleIndex(self - a)


class ChannelIndex(int):
    """Phantom type for channel index in signal block
    """

    def __add__(self, a: int):
        return ChannelIndex(self + a)

    def __sub__(self, a: int):
        return ChannelIndex(self - a)


class SamplingFrequencyHz(float):
    """Phantom type for sampling frequency of signal block
    """

    def __mul__(self, a: float):
        return SamplingFrequencyHz(self * a)

    def __div__(self, a: float):
        return SamplingFrequencyHz(self / a)


class ChannelId(int):
    """Phantom type for channel ID
    """
    pass


class UnitId(int):
    """Phantom type for unit ID
    """
    pass
