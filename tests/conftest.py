import os
import sys

import pytest

# Make the packages under src/, the agent modules, and the analysis scripts
# importable regardless of the pytest invocation directory.
_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(_TESTS)
_RUNNER = os.path.join(_PILOT, "src", "prime_agents")
_ANALYSIS = os.path.join(_PILOT, "experiments", "analysis")
for _p in (_PILOT, os.path.join(_PILOT, "src"), _RUNNER, _ANALYSIS, os.path.join(_ANALYSIS, "pilot")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The toy fixture repo has its own pytest suite; it is exercised via the
# run_tests tool in a subprocess, not collected here.
collect_ignore = ["fixtures"]

FIXTURE_INSTANCE = os.path.join(_TESTS, "fixtures", "instance_toy.json")


@pytest.fixture
def toy_instance():
    from prime_runtime.schema import load_instance

    return load_instance(FIXTURE_INSTANCE)
