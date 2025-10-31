"""Unified Docker plugin builder for test steps - supports both CI and Fastcheck modes."""

from typing import Dict

from ..command_builders.coverage_injection import CoverageTransformer
from ..command_builders.intelligent_test_selection import TestTargetingTransformer
from ..command_builders.normalizer import flatten_commands, normalize_commands
from ..data_models.docker_config import (
    HF_HOME,
    HF_HOME_FSX,
    DockerEnvironment,
    DockerVolumes,
    SpecialGPUDockerConfig,
    StandardDockerConfig,
    get_a100_kubernetes_config,
    get_h100_kubernetes_config,
)
from ..data_models.test_step import TestStep
from ..pipeline_config import PipelineGeneratorConfig
from ..utils.constants import (
    DEFAULT_WORKING_DIR,
    EnvironmentValues,
    EnvironmentVariables,
    GPUType,
    KubernetesConstants,
    PipelineMode,
    PluginNames,
    ShellCommands,
    TestLabels,
)


def build_docker_command_ci(test_step: TestStep, config: PipelineGeneratorConfig) -> str:
    """
    Build docker command for CI mode with full transformation pipeline.

    Applies transformations in order:
    1. Flatten multi-node commands
    2. Normalize commands (remove backslashes)
    3. Apply intelligent test targeting (if applicable)
    4. Apply coverage injection (if enabled)
    5. Join commands
    """
    # Flatten and normalize commands
    commands = flatten_commands(test_step.commands or [])
    commands = normalize_commands(commands)

    # Try intelligent test targeting first
    targeting_transformer = TestTargetingTransformer()
    targeted_command = targeting_transformer.transform(commands, test_step, config)
    if targeted_command:
        return targeted_command

    # Apply coverage if enabled, otherwise just join commands
    coverage_transformer = CoverageTransformer()
    result = coverage_transformer.transform(commands, test_step, config)
    return result if result else " && ".join(commands)


def build_docker_command_fastcheck(test_step: TestStep, config: PipelineGeneratorConfig) -> str:
    """Build docker command for fastcheck mode (simpler, no coverage)."""
    # Flatten and normalize commands
    commands = flatten_commands(test_step.commands or [])
    commands = normalize_commands(commands)

    # Fastcheck doesn't use coverage or intelligent targeting - just join commands
    return " && ".join(commands)


def build_full_docker_command(test_step: TestStep, config: PipelineGeneratorConfig) -> str:
    """Build the full command that runs inside docker container."""
    # Choose command builder based on mode
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        docker_command = build_docker_command_fastcheck(test_step, config)
    else:  # CI mode
        docker_command = build_docker_command_ci(test_step, config)
    
    working_dir = test_step.working_dir or DEFAULT_WORKING_DIR
    return f"{ShellCommands.CHECK_NVIDIA_GPU} && {ShellCommands.SETUP_DEPRECATED_BEAM_SEARCH} && cd {working_dir} && {docker_command}"


def build_full_docker_command_no_coverage(test_step: TestStep, config: PipelineGeneratorConfig) -> str:
    """Build docker command without coverage injection (for kubernetes)."""
    # Flatten and normalize commands
    commands = flatten_commands(test_step.commands or [])
    commands = normalize_commands(commands)

    # Just join commands without coverage
    docker_command = " && ".join(commands)
    working_dir = test_step.working_dir or DEFAULT_WORKING_DIR
    return f"{ShellCommands.CHECK_NVIDIA_GPU} && {ShellCommands.SETUP_DEPRECATED_BEAM_SEARCH} && cd {working_dir} && {docker_command}"


def build_environment(test_step: TestStep, config: PipelineGeneratorConfig) -> DockerEnvironment:
    """Build environment configuration based on pipeline mode."""
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        # Fastcheck: no CODECOV_TOKEN, no BUILDKITE_ANALYTICS_TOKEN
        return DockerEnvironment(
            hf_home=HF_HOME_FSX,
            fail_fast=config.fail_fast,
            is_main_branch=False,  # Never add BUILDKITE_ANALYTICS_TOKEN in fastcheck
            special_attention_backend=(test_step.label == TestLabels.SPECULATIVE_DECODING_TESTS),
            skip_codecov=True,  # No CODECOV_TOKEN in fastcheck
        )
    else:  # CI mode
        # CI: includes CODECOV_TOKEN and BUILDKITE_ANALYTICS_TOKEN on main branch
        return DockerEnvironment(
            hf_home=HF_HOME_FSX,
            fail_fast=config.fail_fast,
            is_main_branch=(config.branch == "main"),
            special_attention_backend=(test_step.label == TestLabels.SPECULATIVE_DECODING_TESTS),
            skip_codecov=False,  # CI mode always includes coverage token
        )


def build_docker_plugin(test_step: TestStep, container_image: str, config: PipelineGeneratorConfig) -> Dict:
    """Build standard Docker plugin configuration."""
    full_command = build_full_docker_command(test_step, config)
    
    # CI mode adds trailing space; Fastcheck doesn't
    if config.pipeline_mode == PipelineMode.CI:
        if full_command and not full_command.endswith(" "):
            full_command += " "
    
    bash_flags = "-xce" if config.fail_fast else "-xc"

    # Build environment configuration (mode-aware)
    environment = build_environment(test_step, config)

    # Build volumes configuration
    volumes = DockerVolumes(hf_home=HF_HOME_FSX)

    # Determine if mount_buildkite_agent is needed
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        mount_agent = test_step.label == TestLabels.BENCHMARKS or test_step.mount_buildkite_agent
    else:  # CI mode
        mount_agent = test_step.label == TestLabels.BENCHMARKS or test_step.mount_buildkite_agent or config.cov_enabled

    docker_config = StandardDockerConfig(
        image=container_image,
        command=full_command,
        bash_flags=bash_flags,
        has_gpu=not test_step.no_gpu,
        environment=environment,
        volumes=volumes,
        mount_buildkite_agent=mount_agent,
    )

    return docker_config.to_plugin_dict()


def build_special_gpu_plugin(test_step: TestStep, container_image: str, config: PipelineGeneratorConfig) -> Dict:
    """Build Docker plugin for special GPUs (H200, B200)."""
    full_command = build_full_docker_command(test_step, config)
    
    # CI mode adds trailing space
    if config.pipeline_mode == PipelineMode.CI:
        if full_command and not full_command.endswith(" "):
            full_command += " "
    
    bash_flags = "-xce" if config.fail_fast else "-xc"

    gpu_type = test_step.gpu.value if test_step.gpu else "h200"

    # Mode-aware configuration
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        # Fastcheck uses FSX paths, no tokens
        hf_home = HF_HOME_FSX
        skip_codecov = True
        is_main = False
    else:  # CI mode
        # CI uses benchmark paths, has tokens
        hf_home = "/benchmark-hf-cache"
        skip_codecov = False
        is_main = (config.branch == "main")

    docker_config = SpecialGPUDockerConfig(
        image=container_image,
        command=full_command,
        bash_flags=bash_flags,
        gpu_type=gpu_type,
        fail_fast=config.fail_fast,
        is_main_branch=is_main,
        skip_codecov=skip_codecov,
        hf_home=hf_home,
    )

    return docker_config.to_plugin_dict()


def build_fastcheck_a100_kubernetes_plugin(test_step: TestStep, container_image: str) -> Dict:
    """Build Kubernetes plugin for A100 in fastcheck mode (fastcheck-specific)."""
    # Build command without coverage
    commands = flatten_commands(test_step.commands)
    commands = normalize_commands(commands)
    docker_command = " && ".join(commands)
    working_dir = test_step.working_dir or DEFAULT_WORKING_DIR
    full_command = f"{ShellCommands.CHECK_NVIDIA_GPU} && {ShellCommands.SETUP_DEPRECATED_BEAM_SEARCH} && cd {working_dir} && {docker_command}"

    num_gpus = test_step.num_gpus or 1

    # Fastcheck uses command/args format (not command with bash -c)
    pod_spec = {
        "priorityClassName": KubernetesConstants.PRIORITY_CLASS_CI,
        "containers": [
            {
                "image": container_image,
                "command": ["bash"],
                "args": ["-c", f"'{full_command}'"],
                "resources": {"limits": {KubernetesConstants.NVIDIA_GPU_RESOURCE: num_gpus}},
                "volumeMounts": [
                    {
                        "name": KubernetesConstants.DEVSHM_VOLUME,
                        "mountPath": KubernetesConstants.DEV_SHM_PATH,
                    },
                    {"name": KubernetesConstants.HF_CACHE_VOLUME, "mountPath": HF_HOME},
                ],
                "env": [
                    {
                        "name": EnvironmentVariables.VLLM_USAGE_SOURCE,
                        "value": EnvironmentValues.VLLM_USAGE_CI_TEST,
                    },
                    {"name": "NCCL_CUMEM_HOST_ENABLE", "value": 0},  # Integer
                    {"name": "HF_HOME", "value": HF_HOME},
                    {
                        "name": "HF_TOKEN",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": KubernetesConstants.HF_TOKEN_SECRET_NAME,
                                "key": KubernetesConstants.HF_TOKEN_SECRET_KEY,
                            }
                        },
                    },
                ],
            }
        ],
        "nodeSelector": {KubernetesConstants.NVIDIA_GPU_PRODUCT: KubernetesConstants.NVIDIA_A100_PRODUCT},
        "volumes": [
            {
                "name": KubernetesConstants.DEVSHM_VOLUME,
                "emptyDir": {"medium": KubernetesConstants.EMPTY_DIR_MEDIUM},
            },
            {
                "name": KubernetesConstants.HF_CACHE_VOLUME,
                "hostPath": {"path": HF_HOME, "type": KubernetesConstants.HOST_PATH_TYPE},
            },
        ],
    }

    return {PluginNames.KUBERNETES: {"podSpec": pod_spec}}


def build_kubernetes_plugin(test_step: TestStep, container_image: str, config: PipelineGeneratorConfig) -> Dict:
    """Build Kubernetes plugin for A100/H100."""
    # For kubernetes, build command WITHOUT coverage (jinja doesn't inject coverage for kubernetes)
    docker_command_no_cov = build_full_docker_command_no_coverage(test_step, config)
    num_gpus = test_step.num_gpus or 1

    # Route to GPU-specific config
    if test_step.gpu == GPUType.H100:
        return get_h100_kubernetes_config(container_image, docker_command_no_cov, num_gpus)

    if test_step.gpu == GPUType.A100:
        return get_a100_kubernetes_config(container_image, docker_command_no_cov, num_gpus)

    # Fallback to standard Docker plugin
    return build_docker_plugin(test_step, container_image, config)


def build_plugin_for_test_step(test_step: TestStep, container_image: str, config: PipelineGeneratorConfig) -> Dict:
    """
    Build the appropriate plugin configuration for a test step.

    Routes to the correct plugin builder based on GPU type and mode:
    - Fastcheck: Only A100 uses special kubernetes, all others use standard Docker
    - CI: H100/A100 use kubernetes, H200/B200 use special GPU plugin, others use standard Docker
    """
    # Fastcheck mode: only A100 uses special handling
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        # Only A100 uses kubernetes in fastcheck
        if test_step.gpu == GPUType.A100:
            return build_fastcheck_a100_kubernetes_plugin(test_step, container_image)
        # All other GPUs (H100, H200, B200, etc.) use standard Docker in fastcheck
        return build_docker_plugin(test_step, container_image, config)
    
    # CI mode: special handling for multiple GPU types
    # CI mode uses Kubernetes for H100 and A100
    if test_step.gpu in [GPUType.H100, GPUType.A100]:
        return build_kubernetes_plugin(test_step, container_image, config)

    # Special GPUs (H200, B200) - only in CI mode
    if test_step.gpu in [GPUType.H200, GPUType.B200]:
        return build_special_gpu_plugin(test_step, container_image, config)

    # Standard Docker for all others
    return build_docker_plugin(test_step, container_image, config)

