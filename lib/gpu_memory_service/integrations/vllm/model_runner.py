# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS model runner subclass for shadow mode.

Plan A: the shadow engine allocates real full-shape KV tensors at init time
through the GMS kv_cache mempool, backed by scratch-aliased physical memory.
At wake, GMSClientMemoryManager.commit_real_backing() swaps the scratch for
unique per-chunk physical at the same VAs. Cudagraphs captured at init survive
the swap because they reference VAs, not physical addresses.

This subclass exists only to track the in_shadow_init flag used by
GMSWorker.wake_up() to decide whether to call exit_shadow_init. All other
overrides the pre-Plan-A version carried (_get_slot_mappings -> (None, None),
_check_and_update_cudagraph_mode PIECEWISE clamp, allocate_kv_cache_on_wake)
are deleted — Plan A makes them unnecessary because the KV tensors are real
from init onward.
"""

from __future__ import annotations

import logging

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = logging.getLogger(__name__)


class GMSShadowModelRunner(GPUModelRunner):
    """GPUModelRunner subclass for shadow mode state tracking.

    Injected via __class__ swap in GMSWorker.init_device(). Carries a boolean
    flag that GMSWorker.wake_up() checks before calling exit_shadow_init().
    """

    @property
    def in_shadow_init(self) -> bool:
        """True until exit_shadow_init() is called (typically at wake)."""
        return getattr(self, "_shadow_init_phase", False)

    def enter_shadow_init(self) -> None:
        self._shadow_init_phase = True
        logger.info("[Shadow] Entered shadow init phase")

    def exit_shadow_init(self) -> None:
        self._shadow_init_phase = False
        logger.info("[Shadow] Exited shadow init phase")
