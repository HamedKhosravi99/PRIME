"""Schema loading and instance-constraint validation (SPEC.md)."""

import dataclasses
import json
import os

import pytest

from prime_runtime.schema import (
    SchemaError,
    load_instance,
    make_program,
    validate_instance,
    validate_program,
)

from conftest import FIXTURE_INSTANCE


def test_load_and_validate_fixture(toy_instance):
    assert toy_instance.id == "toy-0001"
    assert toy_instance.claim_ids == ("C1", "C2", "C3")
    assert toy_instance.atom_ids == ("E1", "E2", "E3", "E4")
    assert os.path.isdir(toy_instance.artifact_dir)
    prog = toy_instance.gold_program()
    assert prog.routes("C3") == (("E1", "E2", "E3", "E4"),)
    validate_instance(toy_instance)  # idempotent, no error


def _mutated_instance(tmp_path, mutate):
    with open(FIXTURE_INSTANCE, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    mutate(raw)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw))
    return str(p)


def test_reject_unlisted_tool(tmp_path):
    def mutate(raw):
        raw["atoms"][0]["resolver"]["tool"] = "telepathy"

    with pytest.raises(SchemaError, match="unlisted tool"):
        load_instance(_mutated_instance(tmp_path, mutate))


def test_reject_order_incoherence(tmp_path):
    def mutate(raw):
        # Stronger claim C2's route no longer contains any route of C1.
        raw["gold_licenses"]["C2"] = [["E2"]]

    with pytest.raises(SchemaError, match="order-incoherent"):
        load_instance(_mutated_instance(tmp_path, mutate))


def test_reject_unrealizable_route(tmp_path):
    def mutate(raw):
        # E2=T with E4=F is a forbidden pattern.
        raw["gold_licenses"]["C1"] = [["E1"]]
        raw["gold_licenses"]["C2"] = [["E1", "E2", "!E4"]]

    with pytest.raises(SchemaError, match="unrealizable"):
        load_instance(_mutated_instance(tmp_path, mutate))


def test_reject_dominated_route_representative(tmp_path):
    def mutate(raw):
        raw["gold_licenses"]["C1"] = [["E1"], ["E1", "E3"]]

    with pytest.raises(SchemaError, match="dominated representative"):
        load_instance(_mutated_instance(tmp_path, mutate))


def test_reject_unsat_phi(tmp_path):
    def mutate(raw):
        raw["phi"] = [{"forbid": {"E1": "T"}}, {"forbid": {"E1": "F"}}]

    with pytest.raises(SchemaError, match="unsatisfiable"):
        load_instance(_mutated_instance(tmp_path, mutate))


def test_reject_too_many_atoms(tmp_path):
    def mutate(raw):
        base = raw["atoms"][0]
        raw["atoms"] = [
            dict(base, id="X%02d" % i) for i in range(21)
        ]
        raw["phi"] = []
        raw["gold_licenses"] = {"C1": [["X00"]], "C2": [["X00", "X01"]],
                                "C3": [["X00", "X01", "X02"]]}

    with pytest.raises(SchemaError, match="closure enumeration"):
        load_instance(_mutated_instance(tmp_path, mutate))


def test_program_is_immutable(toy_instance):
    prog = toy_instance.gold_program()
    with pytest.raises(dataclasses.FrozenInstanceError):
        prog.scope = "other"
    with pytest.raises(TypeError):
        prog.licenses["C1"] = ()
