"""Unit tests for Lic / Auth downward closure / MaxAuth antichain."""

from prime_runtime import auth
from prime_runtime.ledger import Ledger
from prime_runtime.schema import make_program


def _att(atom, value, step=1):
    return {"atom": atom, "value": value, "tool": "t", "params": {}, "output_digest": "", "step": step}


def test_auth_downward_closure():
    """Licensing a stronger claim authorizes every weaker comparable claim,
    even when the weaker claim's own license is not directly satisfied."""
    atoms = ("a", "b")
    prog = make_program(
        ("C1", "C2"),
        (("C1", "C2"),),
        {"C1": [["b"]], "C2": [["a", "b"]]},
        "s",
        atoms,
    )
    led = Ledger(atoms, [])
    led.apply(_att("a", "T", 1))
    led.apply(_att("b", "T", 2))
    assert auth.lic("C2", prog, led)
    assert auth.auth_ids(prog, led) == ["C1", "C2"]
    assert auth.max_auth(prog, led) == ["C2"]

    # Weaker-only evidence licenses only the weaker claim.
    led2 = Ledger(atoms, [])
    led2.apply(_att("b", "T"))
    assert auth.auth_ids(prog, led2) == ["C1"]
    assert auth.max_auth(prog, led2) == ["C1"]


def test_max_auth_is_an_antichain():
    """Two incomparable licensed claims both appear; dominated ones do not."""
    atoms = ("a", "b", "c")
    prog = make_program(
        ("W", "X", "Y"),
        (("W", "X"), ("W", "Y")),  # W below both; X, Y incomparable
        {"W": [["a"]], "X": [["a", "b"]], "Y": [["a", "c"]]},
        "s",
        atoms,
    )
    led = Ledger(atoms, [])
    for i, atom in enumerate(atoms):
        led.apply(_att(atom, "T", i))
    assert auth.auth_ids(prog, led) == ["W", "X", "Y"]
    ma = auth.max_auth(prog, led)
    assert ma == ["X", "Y"]  # antichain: W dominated, X/Y incomparable
    assert not any(a != b and auth.preceq(a, b, prog) for a in ma for b in ma)


def test_auth_empty_on_inconsistent_ledger():
    phi = [{"a": "T", "b": "F"}]
    prog = make_program(("C",), (), {"C": [[]]}, "s", ("a", "b"))
    led = Ledger(("a", "b"), phi)
    led.apply(_att("a", "T", 1))
    led.apply(_att("b", "F", 2))
    # Even a trivially-satisfiable (empty) license authorizes nothing.
    assert auth.auth_ids(prog, led) == []
    assert auth.max_auth(prog, led) == []


def test_preceq_reflexive_transitive():
    prog = make_program(("A", "B", "C"), (("A", "B"), ("B", "C")), {}, "s", ())
    assert auth.preceq("A", "A", prog)
    assert auth.preceq("A", "C", prog)  # transitivity
    assert not auth.preceq("C", "A", prog)
