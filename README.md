# symbolicKAN

Julia implementation of B-spline KAN for symbolic regression - recreated pretty much as is from [pykan](https://github.com/KindXiaoming/pykan) on a smaller scale to develop understanding.

WORK IN PROGRESS 

Thank you to KindXiaoming and the rest of the KAN community for putting this awesome network out there for the world to see.

<p align="center">
<img src="figures/symbolic_test.png" alt="KAN Network" width="48%" style="padding-right: 20px;">
<img src="figures/symbolic_test_pruned.png" alt="Pruned KAN Network" width="48%">
</p>


## To run

1. Precompile packages:

```bash
bash setup.sh
```

2. Unit tests:

```bash
bash src/unit_tests/run_tests.sh
```

3. Work in progress

## TODO

1. CUDA?