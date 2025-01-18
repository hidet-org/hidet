# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

from hidet.cuda.device import (
    current_device,
    get_application_freqs,
    set_application_freqs,
    get_supported_graphics_clocks,
)


logger = logging.getLogger(__name__)


class GPUSetFrequencyForBenchmarking:
    _gpu_freq_cache = {}  # class-level cache for freq_wo_throtling {gpu_index: freq}

    def __init__(self):
        """
        Initialize GPU frequency controller for benchmarking.
        """
        self.gpu_index = current_device()
        self.sm_clock = None
        self.original_sm_clock = None
        self.original_mem_clock = None

    def __enter__(self):
        """
        Enter the context, setting the GPU clock frequencies.
        Saves original frequencies and sets new frequencies if specified.
        """
        self.original_sm_clock, self.original_mem_clock = get_application_freqs(self.gpu_index)
        if self.original_sm_clock is not None and self.original_mem_clock is not None:
            self.sm_clock = self.get_freq_wo_throttling()
            set_application_freqs(self.gpu_index, self.sm_clock, self.original_mem_clock)
        else:
            if not hasattr(GPUSetFrequencyForBenchmarking, '_warning_logged'):
                logger.warning("Could not retrieve original GPU frequencies. Skipping frequency setting.")
                GPUSetFrequencyForBenchmarking._warning_logged = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context, restoring the original GPU clock frequencies.
        Ensures GPU returns to initial state regardless of exceptions.
        """
        if self.original_sm_clock is not None and self.original_mem_clock is not None:
            set_application_freqs(self.gpu_index, self.original_sm_clock, self.original_mem_clock)

    def get_freq_wo_throttling(self) -> int:
        """
        Get the GPU frequency without throttling, using cached values when available.

        Returns
        -------
        int
            A safe GPU frequency that avoids thermal throttling, calculated as
            the highest frequency below 85% of maximum supported frequency.
        """
        if self.gpu_index in self._gpu_freq_cache:
            return self._gpu_freq_cache[self.gpu_index]

        NO_THROTTLING_THRESHOLD = 0.85  # 85% of the max frequency. This is an empirical heuristic value.
        supported_freqs = get_supported_graphics_clocks(self.gpu_index)
        max_freq = supported_freqs[0]
        assert max_freq == max(
            supported_freqs
        ), "The maximum frequency should be the first element in supported_freqs"  # make sure max freq is the first one
        for freq in supported_freqs:
            if freq / max_freq <= NO_THROTTLING_THRESHOLD:
                self._gpu_freq_cache[self.gpu_index] = freq
                return freq

        # Fallback to the last supported frequency if no suitable frequency is found below the threshold.
        self._gpu_freq_cache[self.gpu_index] = supported_freqs[-1]
        return supported_freqs[-1]
