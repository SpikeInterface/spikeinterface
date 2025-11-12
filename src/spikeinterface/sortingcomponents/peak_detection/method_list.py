from __future__ import annotations

from .matched_filtering import MatchedFilteringPeakDetector
from .by_channel import ByChannelPeakDetector, ByChannelTorchPeakDetector
from .locally_exclusive import (
    LocallyExclusivePeakDetector,
    LocallyExclusiveTorchPeakDetector,
    LocallyExclusiveOpenCLPeakDetector,
)

detect_peak_methods = {
    "locally_exclusive": LocallyExclusivePeakDetector,
    "locally_exclusive_torch": LocallyExclusiveTorchPeakDetector,
    "locally_exclusive_cl": LocallyExclusiveOpenCLPeakDetector,
    "matched_filtering": MatchedFilteringPeakDetector,
    "by_channel": ByChannelPeakDetector,
    "by_channel_torch": ByChannelTorchPeakDetector,
}
