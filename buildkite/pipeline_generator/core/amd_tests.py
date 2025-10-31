"""Unified AMD test group generation for both CI and Fastcheck modes."""

from typing import Any, Dict, List, Optional

from ..data_models.buildkite_step import get_step_key
from ..data_models.test_step import TestStep
from ..pipeline_config import PipelineGeneratorConfig
from ..utils.amd_command_builder import build_amd_test_command, format_amd_commands
from ..utils.constants import (
    DEFAULT_WORKING_DIR,
    AgentQueue,
    AMDLabelPrefixes,
    AMDQueueLabels,
    BuildStepKeys,
    EnvironmentVariables,
    PipelineMode,
    PriorityValues,
    TestLabels,
)
from .docker_builds import generate_amd_build_step


def get_amd_queue_ci(label: str, num_gpus: Optional[int] = None) -> str:
    """
    Determine AMD queue for CI mode based on test label.

    Maps test labels to AMD GPU counts (8, 4, 2, or 1 GPUs).
    """
    if label in AMDQueueLabels.AMD_MI325_8_LABELS:
        return "amd_mi325_8"
    elif label in AMDQueueLabels.AMD_MI325_4_LABELS:
        return "amd_mi325_4"
    elif label in AMDQueueLabels.AMD_MI325_2_LABELS:
        return "amd_mi325_2"
    else:
        # Default: 1 GPU
        return "amd_mi325_1"


def generate_amd_group(test_steps: List[TestStep], config: PipelineGeneratorConfig) -> Dict[str, Any]:
    """
    Generate the AMD tests group.

    CI mode: All matching tests, no blocks, soft_fail=false, uses AMD MI325 queues
    Fastcheck mode: Only Basic Correctness Test, has block, soft_fail=true, uses MI300_1 queue
    """
    amd_steps = []

    # Add AMD build step (now returns dict directly)
    amd_build_dict = generate_amd_build_step(config)

    # Fastcheck needs depends_on: null explicitly
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        amd_build_dict["depends_on"] = None

    amd_steps.append(amd_build_dict)

    # Add AMD mirror tests
    for test_step in test_steps:
        # Skip tests that don't match mirror hardware
        if not test_step.mirror_hardwares or config.mirror_hw not in test_step.mirror_hardwares:
            continue

        # Fastcheck filter: only Basic Correctness Test
        if config.pipeline_mode == PipelineMode.FASTCHECK:
            if test_step.label != TestLabels.BASIC_CORRECTNESS_TEST:
                continue

            # Fastcheck adds a block for Basic Correctness Test
            block_key = f"block-amd-{get_step_key(test_step.label)}"
            amd_steps.append(
                {
                    "block": f"Run AMD MI300: {test_step.label} with {config.mirror_hw}",
                    "key": block_key,
                    "depends_on": BuildStepKeys.AMD_BUILD,
                }
            )

        # Determine queue and label based on mode
        if config.pipeline_mode == PipelineMode.FASTCHECK:
            queue = AgentQueue.AMD_MI300_1
            label = AMDLabelPrefixes.with_test_and_mirror(test_step.label, config.mirror_hw)
            soft_fail = True
            depends_on = block_key  # Set above in fastcheck mode
        else:  # CI mode
            queue = get_amd_queue_ci(test_step.label, test_step.num_gpus)
            label = AMDLabelPrefixes.with_test(test_step.label)
            soft_fail = False
            depends_on = BuildStepKeys.AMD_BUILD

        # Format commands for AMD test
        commands_str = format_amd_commands(test_step)
        working_dir = test_step.working_dir or DEFAULT_WORKING_DIR
        full_command = build_amd_test_command(working_dir, commands_str)

        amd_step_dict = {
            "label": label,
            "depends_on": depends_on,
            "agents": {"queue": queue},
            "env": {EnvironmentVariables.DOCKER_BUILDKIT: "1"},
            "soft_fail": soft_fail,
            "priority": PriorityValues.AMD_TESTS,
            "command": full_command,
        }
        amd_steps.append(amd_step_dict)

    return {"group": "AMD Tests", "depends_on": None, "steps": amd_steps}
