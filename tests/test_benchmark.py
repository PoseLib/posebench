from posebench import run_benchmark
from posebench.utils.misc import has_pycolmap


def test_benchmark():
    run_benchmark(
        subset=True,
        only_poselib=True,
    )
