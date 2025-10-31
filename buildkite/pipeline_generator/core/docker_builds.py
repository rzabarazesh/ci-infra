"""Unified Docker build step generation for both CI and Fastcheck modes."""

from typing import Dict, List, Optional, Union

from ..data_models.buildkite_step import BuildkiteBlockStep, BuildkiteStep
from ..docker_build_configs import (
    AMDBuildConfig,
    create_cpu_build,
    create_cu118_build,
    create_fastcheck_build,
    create_main_cuda_build,
    create_torch_nightly_build,
)
from ..pipeline_config import PipelineGeneratorConfig
from ..utils.constants import AgentQueue, PipelineMode


def generate_main_build_step(config: PipelineGeneratorConfig) -> Union[BuildkiteStep, Dict]:
    """
    Build the main Docker CUDA image.

    CI: Uses postmerge/test repo based on branch
    Fastcheck: Always uses test-repo with premerge queue

    Returns dict directly to preserve all fields (especially 'key')
    """
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        # Fastcheck: single build with premerge queue
        build_config = create_fastcheck_build(
            commit=config.commit,
            image_tag=config.container_image,
            queue=AgentQueue.CPU_QUEUE_PREMERGE,
            vllm_use_precompiled=config.vllm_use_precompiled,
        )
    else:  # CI mode
        # CI: use postmerge/premerge queue based on branch
        queue = AgentQueue.CPU_QUEUE_POSTMERGE_US_EAST_1 if config.branch == "main" else AgentQueue.CPU_QUEUE_PREMERGE_US_EAST_1
        build_config = create_main_cuda_build(
            commit=config.commit,
            image_tag=config.container_image,
            queue=queue,
            branch=config.branch,
            vllm_use_precompiled=config.vllm_use_precompiled,
        )

    # Return dict directly to preserve key field
    return build_config.to_buildkite_step()


def generate_cu118_build_steps(
    config: PipelineGeneratorConfig,
) -> List[Union[BuildkiteStep, BuildkiteBlockStep]]:
    """
    Build the CUDA 11.8 Docker image.

    CI only - returns empty list for Fastcheck mode.
    """
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        return []  # Fastcheck doesn't build cu118

    # CI mode
    queue = AgentQueue.CPU_QUEUE_POSTMERGE_US_EAST_1 if config.branch == "main" else AgentQueue.CPU_QUEUE_PREMERGE_US_EAST_1

    block_step = BuildkiteBlockStep(block="Build CUDA 11.8 image", key="block-build-cu118", depends_on=None)

    build_config = create_cu118_build(
        commit=config.commit,
        image_tag=config.container_image_cu118,
        queue=queue,
        branch=config.branch,
        vllm_use_precompiled=config.vllm_use_precompiled,
    )

    build_dict = build_config.to_buildkite_step()
    build_step = BuildkiteStep(**build_dict)

    return [block_step, build_step]


def generate_cpu_build_step(config: PipelineGeneratorConfig) -> Optional[Dict]:
    """
    Build the CPU Docker image.

    CI only - returns None for Fastcheck mode.
    Returns dict directly to preserve all fields.
    """
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        return None  # Fastcheck doesn't build CPU image

    # CI mode
    queue = AgentQueue.CPU_QUEUE_POSTMERGE_US_EAST_1 if config.branch == "main" else AgentQueue.CPU_QUEUE_PREMERGE_US_EAST_1

    build_config = create_cpu_build(commit=config.commit, image_tag=config.container_image_cpu, queue=queue)

    # Return dict directly
    return build_config.to_buildkite_step()


def generate_torch_nightly_build_step(config: PipelineGeneratorConfig, depends_on: Optional[str]) -> Dict:
    """
    Build the torch nightly Docker image.

    CI only - should not be called in Fastcheck mode.
    Returns dict directly to preserve all fields.
    """
    queue = AgentQueue.CPU_QUEUE_POSTMERGE_US_EAST_1 if config.branch == "main" else AgentQueue.CPU_QUEUE_PREMERGE_US_EAST_1

    build_config = create_torch_nightly_build(
        commit=config.commit,
        image_tag=config.container_image_torch_nightly,
        queue=queue,
        depends_on=depends_on,
    )

    build_dict = build_config.to_buildkite_step()
    # Add torch nightly specific overrides
    build_dict["soft_fail"] = True
    build_dict["timeout_in_minutes"] = 360

    # Return dict directly
    return build_dict


def generate_amd_build_step(config: PipelineGeneratorConfig) -> Dict:
    """
    Build the AMD Docker image.

    Different configurations for CI vs Fastcheck.
    Returns dict directly to preserve all fields.
    """
    amd_config = AMDBuildConfig(
        image_tag=config.container_image_amd,
        commit=config.commit,
        mirror_hw=config.mirror_hw,
        is_fastcheck=(config.pipeline_mode == PipelineMode.FASTCHECK),
    )

    # Return dict directly
    return amd_config.to_buildkite_step()
