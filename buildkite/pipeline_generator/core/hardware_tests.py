"""Unified hardware-specific test generation for both CI and Fastcheck modes."""

from typing import Any, Dict, List

from ..hardware_test_configs import (
    ASCEND_CONFIG,
    GH200_CONFIG,
    INTEL_GPU_CONFIG,
    INTEL_HPU_CONFIG,
    NEURON_CONFIG,
    TPU_V1_BENCHMARK,
    TPU_V1_TEST,
    TPU_V1_TEST_PART2,
    get_ibm_power_config,
    get_ibm_power_notification_config,
    get_ibm_s390x_config,
    get_intel_cpu_config,
    get_tpu_notification_config,
    get_tpu_v0_notification_config,
)
from ..utils.constants import AgentQueue, BlockLabels, HardwareLabels, Scripts


def generate_all_hardware_tests(branch: str, nightly: bool) -> List[Dict[str, Any]]:
    """
    Generate all hardware-specific test steps using data-driven configuration.

    Used by CI mode. Returns CI-specific hardware tests without blocking steps.
    """
    steps = []

    # Simple hardware tests
    steps.append(NEURON_CONFIG.to_buildkite_step())

    # Intel tests
    intel_cpu = get_intel_cpu_config(branch)
    steps.extend(intel_cpu.to_buildkite_steps())

    steps.append(INTEL_HPU_CONFIG.to_buildkite_step())
    steps.append(INTEL_GPU_CONFIG.to_buildkite_step())

    # Ascend NPU
    steps.append(ASCEND_CONFIG.to_buildkite_step())

    # IBM Power
    ibm_power = get_ibm_power_config(branch)
    steps.extend(ibm_power.to_buildkite_steps())

    # Add IBM Power notification for main branch
    if branch == "main":
        ibm_power_notif = get_ibm_power_notification_config()
        steps.append(ibm_power_notif.to_buildkite_step())

    # IBM Z (s390x)
    ibm_s390x = get_ibm_s390x_config(nightly)
    steps.extend(ibm_s390x.to_buildkite_steps())

    # GH200 (nightly only)
    if nightly:
        steps.append(GH200_CONFIG.to_buildkite_step())

    # TPU tests (V1 only in CI mode, V0 is fastcheck only)
    steps.append(TPU_V1_TEST.to_buildkite_step())
    steps.append(TPU_V1_TEST_PART2.to_buildkite_step())
    steps.append(TPU_V1_BENCHMARK.to_buildkite_step())

    # Add TPU notification for main branch
    if branch == "main":
        tpu_depends_on = get_tpu_notification_config()
        # Build notification command with proper indentation
        tpu_notif_command = (
            """if [[ $$(buildkite-agent step get "outcome" --step "run-tpu-v1-test") != "passed" || """
            """$$(buildkite-agent step get "outcome" --step "run-tpu-v1-test-part2") != "passed" ]]; then
   cat <<- YAML | buildkite-agent pipeline upload
   steps:
     - label: "Notify owners about failing test"
       agents:
         queue: tpu_v6e_queue
       command: echo "TPU V1 Test failed"
       notify:
         - slack:
             channels:
               - "vllm#tpu-ci-notifications"
YAML
fi"""
        )
        steps.append(
            {
                "label": "TPU V1 Test Notification",
                "depends_on": tpu_depends_on,
                "soft_fail": True,
                "agents": {"queue": AgentQueue.TPU_V6E_QUEUE},
                "commands": tpu_notif_command,
            }
        )

    return steps


# ============================================================================
# Fastcheck-specific helper functions
# ============================================================================


def get_tpu_v0_tests() -> List[Dict[str, Any]]:
    """Get TPU V0 tests (fastcheck-only)."""
    return [
        {"block": BlockLabels.RUN_TPU_V0_TEST, "key": "block-tpu-v0", "depends_on": None},
        {
            "label": HardwareLabels.TPU_V0_TEST,
            "key": "run-tpu-v0-test",
            "depends_on": "block-tpu-v0",
            "soft_fail": True,
            "agents": {"queue": AgentQueue.TPU_V5_QUEUE},
            "commands": [
                f'if [[ -f "{Scripts.RUN_TPU_TEST}" ]]; then bash {Scripts.RUN_TPU_TEST}; fi',
                "yes | docker system prune -a",
            ],
        },
        get_tpu_v0_notification_config(),
    ]


def get_tpu_v1_tests(all_hw_tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get TPU V1 tests for fastcheck (with blocks)."""
    steps = []

    for test in all_hw_tests:
        if isinstance(test, dict) and test.get("label", "") == "TPU V1 Test":
            steps.append({"block": "Run TPU V1 Test", "key": "block-tpu-v1", "depends_on": None})
            test_copy = test.copy()
            test_copy["depends_on"] = "block-tpu-v1"
            if "timeout_in_minutes" in test_copy:
                del test_copy["timeout_in_minutes"]
            if "soft_fail" in test_copy:
                del test_copy["soft_fail"]
            steps.append(test_copy)

    # Add notification
    tpu_v1_notif_cmd = """if [ $$(buildkite-agent step get "outcome" --step "run-tpu-v1-test") != "passed" ]; then
   cat <<- YAML | buildkite-agent pipeline upload
   steps:
     - label: "Notify owners about failing test"
       agents:
         queue: tpu_v5_queue
       command: echo "TPU V1 Test failed"
       notify:
         - slack:
             channels:
               - "#tpu-ci-notifications"
YAML
fi
"""
    notification_step: Dict[str, Any] = {
        "label": "TPU V1 Test Notification",
        "depends_on": "run-tpu-v1-test",
        "soft_fail": True,
        "agents": {"queue": "tpu_v5_queue"},
        "commands": tpu_v1_notif_cmd,
    }
    steps.append(notification_step)

    return steps


def get_gh200_test() -> List[Dict[str, Any]]:
    """Get GH200 test for fastcheck (with block)."""
    gh200_step = GH200_CONFIG.to_buildkite_step()
    gh200_step["depends_on"] = "block-gh200"

    return [{"block": "Run GH200 Test", "depends_on": None, "key": "block-gh200"}, gh200_step]


def get_intel_tests(all_hw_tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get Intel tests for fastcheck (with blocks)."""
    steps = []

    for test in all_hw_tests:
        if isinstance(test, dict):
            label = test.get("label", "")
            if "Intel CPU Test" in label:
                steps.append({"block": "Run Intel CPU test", "key": "block-intel-cpu", "depends_on": None})
                test_copy = test.copy()
                test_copy["depends_on"] = "block-intel-cpu"
                steps.append(test_copy)
            elif "Intel GPU Test" in label:
                steps.append({"block": "Run Intel GPU test", "key": "block-intel-gpu", "depends_on": None})
                test_copy = test.copy()
                test_copy["depends_on"] = "block-intel-gpu"
                steps.append(test_copy)

    return steps


def add_neuron_test_fastcheck(steps: List) -> None:
    """Add Neuron test at the beginning for fastcheck mode."""
    neuron_block: Dict[str, Any] = {
        "block": BlockLabels.RUN_NEURON_TEST,
        "depends_on": None,
        "key": "run-neuron-test",
    }
    neuron_test: Dict[str, Any] = {
        "label": HardwareLabels.NEURON_TEST,
        "depends_on": "run-neuron-test",
        "agents": {"queue": AgentQueue.NEURON},
        "command": f"bash {Scripts.RUN_NEURON_TEST}",
        "soft_fail": False,
    }
    steps.append(neuron_block)
    steps.append(neuron_test)
