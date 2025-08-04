from posebench import run_benchmark


def test_benchmark():
    run_benchmark(
        subset=True,
        subsample=10,
    )

if __name__ == '__main__':
    test_benchmark()