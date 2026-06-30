# Tests

The tests defined here fall into one of two categories.

1. unit tests : these are relatively fast and can be run very regularly. The goal is to test individual components - i.e. units.
2. functional tests : the goal is to check that the ecosystem does what is intended - i.e. that we can actually run an AFQMC calculation and get the correct numbers. We're testing the full system, and proving that the code does what we claim it does.


## Unit Tests

These are run as usual via pytest: `$ pytest` (from the `utils` directory).
See the `pyproject.toml` file for default configuration.


## Functional Tests

The functional tests are also invoked via `pytest`; however, the default settings
defined in the `pyrpoject.toml` file will not  run the functional tests by default.
You must explicitly request them using pytest marks.

### Sample Functional Test Slurm script

A sample script to run the "weekly" functional tests is as follows.

```bash
#!/bin/bash
#SBATCH -J func_tests_safire
#SBATCH --partition=ccq
#SBATCH --constraint=genoa
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH -o functional_tests.o%j

senf safire
module load cuda
module load openmpi/cuda

echo "Starting functional tests... "
date

OUTPUT_DIR="/path/to/scratch/dir"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

set -x PYTHONPATH (pwd) $PYTHONPATH
python functional/run_functional.py \
       all \
       --output-path=$OUTPUT_DIR \
       --mpiexec="srun -n 90 --cpu-bind=cores" \
       --compute cpu \
       --timeout=120 \

echo "... done!"
date
```

> 💡 Note that the tests use a temporary directory for scratch work. 
> The directory is cleared upon the next test run.
> These are a very useful place to debug problems in the test.

### Local Runs

```bash
python functional/run_functional.py \
       all \
       --output-path=$OUTPUT_DIR \
       --mpiexec="mpirun -n 16" \
       --compute cpu \
       --timeout=120
```

### Adding new cases

See the file `functional/functional_cases.py`.



