from posebench import run_benchmark


def test_benchmark():
    run_benchmark(
        subset=False,
        subsample=100,
    )
