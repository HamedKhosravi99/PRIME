from toypkg import add, mul


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_mul():
    assert mul(2, 3) == 6
    assert mul(0, 7) == 0
