"""Unified test step conversion logic for both CI and Fastcheck modes."""

from typing import List

from ..data_models.buildkite_step import BuildkiteStep
from ..data_models.test_step import TestStep
from ..pipeline_config import PipelineGeneratorConfig
from ..utils.agent_queues import get_agent_queue
from ..utils.constants import DEFAULT_WORKING_DIR, AgentQueue, GPUType, PipelineMode, PriorityValues, RetryConfig, Scripts, TestLabels
from .docker_plugins import build_plugin_for_test_step


def convert_multi_node_test_step_ci(test_step: TestStep, container_image: str, config: PipelineGeneratorConfig) -> BuildkiteStep:
    """Convert a multi-node test step for CI mode."""
    working_dir = test_step.working_dir or DEFAULT_WORKING_DIR

    # Extract commands for each node
    node_commands: List[List[str]] = []
    if test_step.commands and len(test_step.commands) > 0 and isinstance(test_step.commands[0], list):
        # Multi-node format: list of lists
        node_commands = test_step.commands  # type: ignore
    else:
        # Fallback: use the same commands for all nodes
        simple_commands = test_step.commands if test_step.commands else []
        node_commands = [simple_commands] * (test_step.num_nodes or 2)  # type: ignore

    # Build the multi-node command
    quoted_node_commands = []
    for node_cmds in node_commands:
        # Join commands for this node
        node_cmd_str = " && ".join(node_cmds)
        # Use double quotes to match jinja template
        quoted_node_commands.append(f'"{node_cmd_str}"')

    multi_node_cmd = (
        f"{Scripts.RUN_MULTI_NODE_TEST} {working_dir} {test_step.num_nodes} {test_step.num_gpus or 1} {container_image} {' '.join(quoted_node_commands)}"
    )

    agent_queue = get_agent_queue(test_step.no_gpu, test_step.gpu, test_step.num_gpus, test_step.label)

    # CI mode always uses retry_limit = 1
    retry_limit = 1

    return BuildkiteStep(
        label=test_step.label,
        key=None,  # Don't add keys to regular test steps (matches jinja)
        commands=[multi_node_cmd],
        parallelism=test_step.parallelism,
        soft_fail=test_step.soft_fail or False,
        plugins=None,  # No docker plugin for multi-node
        agents={"queue": agent_queue},
        timeout_in_minutes=None,  # Multi-node steps don't include timeout in plugins
        depends_on="build",  # Test steps depend on build
        retry={
            "automatic": [
                {"exit_status": RetryConfig.EXIT_STATUS_AGENT_LOST, "limit": retry_limit},
                {"exit_status": RetryConfig.EXIT_STATUS_AGENT_TERMINATED, "limit": retry_limit},
            ]
        },
    )


def convert_multi_node_test_step_fastcheck(test_step: TestStep, container_image: str, config: PipelineGeneratorConfig) -> BuildkiteStep:
    """Convert a multi-node test step for fastcheck mode."""
    working_dir = test_step.working_dir or DEFAULT_WORKING_DIR

    # Extract commands for each node
    node_commands: List[List[str]] = []
    if test_step.commands and len(test_step.commands) > 0 and isinstance(test_step.commands[0], list):
        node_commands = test_step.commands  # type: ignore
    else:
        simple_commands = test_step.commands if test_step.commands else []
        node_commands = [simple_commands] * (test_step.num_nodes or 2)  # type: ignore

    # Build the multi-node command
    quoted_node_commands = []
    for node_cmds in node_commands:
        node_cmd_str = " && ".join(node_cmds)
        quoted_node_commands.append(f'"{node_cmd_str}"')

    multi_node_cmd = (
        f"{Scripts.RUN_MULTI_NODE_TEST} {working_dir} {test_step.num_nodes} {test_step.num_gpus or 1} {container_image} {' '.join(quoted_node_commands)}"
    )

    agent_queue = get_agent_queue(test_step.no_gpu, test_step.gpu, test_step.num_gpus, test_step.label)

    # Fastcheck multi-node tests have NO retry, NO soft_fail, NO plugins
    return BuildkiteStep(
        label=test_step.label,
        key=None,
        commands=[multi_node_cmd],
        parallelism=test_step.parallelism,
        soft_fail=None,  # No soft_fail for multi-node in fastcheck
        plugins=None,
        agents={"queue": agent_queue},
        timeout_in_minutes=None,
        depends_on="build",
        retry=None,  # No retry for multi-node in fastcheck
    )


def get_agent_queue_fastcheck(test_step: TestStep) -> str:
    """Get agent queue for fastcheck mode (simpler queue logic)."""
    if test_step.label == TestLabels.DOCUMENTATION_BUILD:
        return AgentQueue.SMALL_CPU_QUEUE_PREMERGE
    elif test_step.no_gpu:
        return AgentQueue.CPU_QUEUE_PREMERGE  # Fastcheck uses simple queue
    elif test_step.num_gpus == 2 or test_step.num_gpus == 4:
        return AgentQueue.GPU_4_QUEUE
    else:
        return AgentQueue.GPU_1_QUEUE


def convert_test_step_to_buildkite_step(test_step: TestStep, container_image: str, config: PipelineGeneratorConfig) -> BuildkiteStep:
    """
    Unified function to convert TestStep into BuildkiteStep.

    Routes to mode-specific logic based on config.pipeline_mode.
    """
    # Check if this is a multi-node test
    if test_step.num_nodes and test_step.num_nodes >= 2:
        if config.pipeline_mode == PipelineMode.FASTCHECK:
            return convert_multi_node_test_step_fastcheck(test_step, container_image, config)
        else:  # CI mode
            return convert_multi_node_test_step_ci(test_step, container_image, config)

    # Get plugin configuration (mode-aware via build_plugin_for_test_step)
    plugin_config = build_plugin_for_test_step(test_step, container_image, config)

    # Determine agents based on mode
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        # Fastcheck has special A100 handling
        if test_step.gpu == GPUType.A100:
            agent_queue = AgentQueue.A100_QUEUE
        else:
            agent_queue = get_agent_queue_fastcheck(test_step)

        # Fastcheck uses retry_limit = 5 (except for A100 which is handled below)
        retry_limit = 5

        # A100 in fastcheck has special handling
        if test_step.gpu == GPUType.A100:
            return BuildkiteStep(
                label=test_step.label,
                key=None,
                commands=[],
                parallelism=test_step.parallelism,
                soft_fail=test_step.soft_fail or False,
                plugins=[plugin_config],
                agents={"queue": agent_queue},
                timeout_in_minutes=None,
                depends_on=None,  # A100 in fastcheck has no depends_on
                retry={
                    "automatic": [
                        {"exit_status": RetryConfig.EXIT_STATUS_AGENT_LOST, "limit": 5},
                        {"exit_status": RetryConfig.EXIT_STATUS_AGENT_TERMINATED, "limit": 5},
                    ]
                },
                priority=PriorityValues.A100_TESTS,
            )
    else:  # CI mode
        agent_queue = get_agent_queue(test_step.no_gpu, test_step.gpu, test_step.num_gpus, test_step.label)
        # CI mode always uses retry_limit = 1
        retry_limit = 1

    # Build the Buildkite step
    buildkite_step = BuildkiteStep(
        label=test_step.label,
        key=None,  # Don't add keys to regular test steps (matches jinja)
        commands=[],  # Empty for docker plugin steps (command is in plugin)
        parallelism=test_step.parallelism,
        soft_fail=test_step.soft_fail or False,
        plugins=[plugin_config],
        agents={"queue": agent_queue},
        timeout_in_minutes=None,  # Don't include timeout unless jinja does
        depends_on="build",  # Will be overridden by caller
        retry={
            "automatic": [
                {"exit_status": RetryConfig.EXIT_STATUS_AGENT_LOST, "limit": retry_limit},
                {"exit_status": RetryConfig.EXIT_STATUS_AGENT_TERMINATED, "limit": retry_limit},
            ]
        },
    )

    return buildkite_step
