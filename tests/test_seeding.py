import random

import numpy as np

from cricket_commentary.utils.seeding import rng_for, set_seed


def test_set_seed_makes_streams_repeat():
    set_seed(13)
    a = (random.random(), np.random.rand())
    set_seed(13)
    b = (random.random(), np.random.rand())
    assert a == b


def test_rng_for_is_keyed_and_deterministic():
    x = rng_for(13, "match1", 4, 3).integers(0, 1_000_000)
    y = rng_for(13, "match1", 4, 3).integers(0, 1_000_000)
    z = rng_for(13, "match1", 4, 4).integers(0, 1_000_000)
    assert x == y
    assert x != z  # different ball -> different stream (overwhelmingly)


def test_rng_for_differs_across_seeds():
    assert (
        rng_for(1, "m", 0, 0).integers(0, 1_000_000)
        != rng_for(2, "m", 0, 0).integers(0, 1_000_000)
    )
