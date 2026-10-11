"""Gold-ledger reconstruction from a synthetic trace (master SS2.3)."""

from prime_runtime import goldledger


def _synthetic_trace():
    """A synthetic environment trace for the toy instance.

    Includes decoy self-reports (attestation + emission entries claiming
    everything succeeded) that the reconstructor MUST ignore — only
    (action, observation) pairs count.
    """
    return [
        {"type": "meta", "instance": "toy-0001", "arm": "react"},
        {
            "type": "action", "step": 1, "name": "inspect_file",
            "params": {"path": "toypkg/calc.py"},
        },
        {
            "type": "observation", "step": 1, "action": "inspect_file",
            "params": {"path": "toypkg/calc.py"},
            "data": {"path": "toypkg/calc.py", "content": "def add..."},
        },
        # Self-reported attestation for an atom the environment never
        # established — must NOT reach the gold ledger.
        {
            "type": "attestation", "step": 1,
            "attestation": {"atom": "E2", "value": "T", "tool": "self", "step": 1},
            "ledger_version": 99,
        },
        {
            "type": "action", "step": 2, "name": "run_tests",
            "params": {"suite": "default"},
        },
        {
            "type": "observation", "step": 2, "action": "run_tests",
            "params": {"suite": "default"},
            "data": {"suite": "default", "exit_code": 1, "stdout": "1 failed"},
        },
        {
            "type": "emission", "step": 3, "claims": ["C3"],
            "claim_object": {"authorized_claims": ["C3"]}, "ledger_version": 99,
        },
    ]


def test_reconstruct_ignores_self_reports(toy_instance):
    replay = goldledger.reconstruct(_synthetic_trace(), toy_instance)
    # Step 1: only E1 attested (span read); the self-reported E2=T ignored.
    assert replay.states_at(1) == {"E1": "T", "E2": "U", "E3": "U", "E4": "U"}
    # Step 2: run_tests FAILED -> E2 attested F from the observation.
    assert replay.states_at(2)["E2"] == "F"


def test_complete_auth_per_step(toy_instance):
    replay = goldledger.reconstruct(_synthetic_trace(), toy_instance)
    # Before any work nothing is gold-authorized.
    assert replay.complete_auth("C1", 0) == 0
    # After step 1, E1=T licenses C1.
    assert replay.complete_auth("C1", 1) == 1
    assert replay.complete_auth("C2", 1) == 0
    # The premature C3 emission at step 3 scores 0 (tests failed).
    assert replay.complete_auth("C3", 3) == 0


def test_complete_auth_is_authorization_aware(toy_instance):
    """Completing a dominating claim's license makes CompleteAuth hold for
    the dominated claim (downward closure, master SS2.3)."""
    trace = [
        {
            "type": "observation", "step": 1, "action": "inspect_file",
            "params": {"path": "toypkg/calc.py"},
            "data": {"content": "..."},
        },
        {
            "type": "observation", "step": 2, "action": "run_tests",
            "params": {"suite": "default"},
            "data": {"exit_code": 0},
        },
    ]
    replay = goldledger.reconstruct(trace, toy_instance)
    # C2's license {E1,E2} is closed at step 2 -> C1 authorized through it.
    assert replay.complete_auth("C2", 2) == 1
    assert replay.complete_auth("C1", 2) == 1
    assert replay.complete_auth("C3", 2) == 0


def test_eval_attest_rules():
    assert goldledger.eval_attest("exit_code==0", {"exit_code": 0}) == "T"
    assert goldledger.eval_attest("exit_code==0", {"exit_code": 2}) == "F"
    assert goldledger.eval_attest("exit_code==0", {}) is None  # never guess
    assert goldledger.eval_attest("match_count==0", {"match_count": 0}) == "T"
    assert goldledger.eval_attest("match_count==0", {"match_count": 3}) == "F"
    assert goldledger.eval_attest("span_read", {"content": "x"}) == "T"
    assert goldledger.eval_attest("span_read", {"error": "missing"}) is None
    assert goldledger.eval_attest("answer_given", {"answer": "staging"}) == "T"
    assert goldledger.eval_attest("answer_given", {"answer": "no answer available"}) is None
