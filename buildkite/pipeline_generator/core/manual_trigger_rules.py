"""Unified manual trigger (blocking) rules for both CI and Fastcheck modes."""

from ..data_models.test_step import TestStep
from ..pipeline_config import PipelineGeneratorConfig
from ..utils.constants import PipelineMode


def should_run_ci_test(test_step: TestStep, config: PipelineGeneratorConfig) -> bool:
    """
    Determine if a CI test should run based on configuration and file changes.

    This is the CI-specific version of should_run_step.
    """
    # Always run if run_all or nightly is enabled
    if config.run_all or config.nightly:
        return True

    # Check source file dependencies
    if test_step.source_file_dependencies:
        for source_file in test_step.source_file_dependencies:
            for changed_file in config.list_file_diff:
                if source_file in changed_file:
                    return True
        return False

    # If no dependencies specified, always run
    return True


def should_block_ci_test(test_step: TestStep, config: PipelineGeneratorConfig) -> bool:
    """
    Determine if a CI test needs a manual trigger block.

    Tests are blocked if:
    - They have source_file_dependencies that don't match changed files
    - They're marked optional (unless nightly mode)
    """
    # Check if blocked due to file dependencies
    if not should_run_ci_test(test_step, config):
        return True

    # Check if blocked due to being optional (independent of run_all!)
    if test_step.optional and not config.nightly:
        return True

    return False


def should_block_fastcheck_test(test_step: TestStep, config: PipelineGeneratorConfig) -> bool:
    """
    Determine if a fastcheck test needs a manual trigger block.

    In fastcheck mode:
    - Tests with fast_check=True NEVER blocked (run immediately)
    - All non-fast-check tests ARE blocked
    """
    # Fast-check tests never blocked
    if test_step.fast_check:
        return False

    # All other tests are blocked
    return True


def should_block_torch_nightly_test(test_step: TestStep, config: PipelineGeneratorConfig) -> bool:
    """
    Blocking logic for tests in the torch nightly group.

    Note: Torch nightly has DIFFERENT blocking rules than regular tests.
    """
    # Block if optional and not nightly (takes precedence)
    if test_step.optional and not config.nightly:
        return True

    # Unblock if nightly mode
    if config.nightly:
        return False

    # Unblock if no dependencies (always run)
    if not test_step.source_file_dependencies:
        return False

    # Unblock if dependencies match changed files
    for source_file in test_step.source_file_dependencies:
        for changed_file in config.list_file_diff:
            if source_file in changed_file:
                return False

    # Otherwise, block (dependencies don't match)
    return True


def should_block_test(test_step: TestStep, config: PipelineGeneratorConfig) -> bool:
    """
    Unified function to determine if a test needs a manual trigger block.
    
    Routes to mode-specific logic based on config.pipeline_mode.
    """
    if config.pipeline_mode == PipelineMode.FASTCHECK:
        return should_block_fastcheck_test(test_step, config)
    else:  # CI mode
        return should_block_ci_test(test_step, config)

