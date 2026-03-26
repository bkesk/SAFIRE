---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "m5yYPGDk-NIu"}

# Hello SAFIRE

<b>Goal:</b>
At the end of this tutorial, you will know how to run an AFQMC calculation using SAFIRE and extract the AFQMC energy from the output
given all necessary inputs.


## What you will learn




1. what are the inputs and outputs of SAFIRE.
2. how to invoke the SAFIRE executable from the command line in serial, using MPI, and using GPU(s)
3. how to extract the AFQMC energy from the outputs of SAFIRE using afqmctools
4. how to use the tutorial helper functions to perform 2 and 3 from within an interactive Python notebook





+++ {"id": "7FBZ0COY-NIw"}



<b>
Inputs and outputs of SAFIRE.
</b>


---



The purpose of this tutorial is to run a first AFQMC calculation without going into low-level details.

AFQMC is formulated in terms of a generic second quantized Hamiltonian and trial wavefunction. 
In order to take advantage of this flexibility, SAFIRE accepts a generic second quantized Hamiltonian and wavefunction, 
saved in HDF5 files, as input along with AFQMC algorithmic parameters in a json-based input file.
Below is a schematic of the inputs and outputs of SAFIRE.



<div>
<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/01_hello_auxiliary_fields/inputs_and_outputs_all.png" width="800">
</div>







We will learn more about the details of all of these files in future tutorials. For this tutorial, pregenerated input files are provided in the "/molecules/files" directory relative to the root directory of the tutorials.

+++ {"id": "tMJ2Y4re-NIw"}





<b>The Inputs
</b>



---
To perform a calculation, SAFIRE requires a Hamiltonian and a trial wave function. Both are provided as inputs using the files "hamil.h5" and "wfn.h5", respectively. The file "input.json" references these two files and provides further information about how the simulation should be run.




<b> Hamiltonian file : "hamil.h5"
</b>


---



<!--
<div>
<img src="./figs/inputs_and_outputs_hamil.png" width="800"/>
</div>
-->

<div>
<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/01_hello_auxiliary_fields/inputs_and_outputs_hamil.png" width="800">
</div>





This HDF5 file contains the second-quantized Hamiltonian $\hat{H}$.

For this tutorial, we have supplied an HDF5 file containing the Born-Oppenheimer
Hamiltonian for the STO-3G basis hydrogen dimer at a bondlength of $\delta_{H-H} = 1.4$ $a_0$
in "hamil.h5".

<!--
<div>
<img src="./figs/HyrdogenMolecule_v2.png" width="300"/>
</div>
-->

<div>
<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/01_hello_auxiliary_fields/HyrdogenMolecule_v2.png" width="300">
</div>





For technical details on the Hamiltonian HDF5 format, see [the User Guide](https://users.flatironinstitute.org/~beskridge/auxiliary_fields/afqmc.html#dense-cholesky).






<b> Wavefunction file : "wfn.h5"
</b>


---



<!--
<div>
<img src="./figs/inputs_and_outputs_wfn.png" width="800"/>
</div>
-->

<div>
<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/01_hello_auxiliary_fields/inputs_and_outputs_wfn.png" width="800">
</div>



This HDF5 file contains the trial wavefunction, $| \Psi_T \rangle$.


For this tutorial, we have supplied an HDF5 file containing the RHF
ground state as the trial wavefunction.

$$
| \Psi_T \rangle = | \Phi_\mathrm{RHF} \rangle
$$

For technical details on the wavefunction HDF5 format, see [the User Guide](https://users.flatironinstitute.org/~beskridge/auxiliary_fields/afqmc.html#wavefunction-file-formats).





<b> Input file (json format) : "input.json"
</b>


---



<!--
<div>
<img src="./figs/inputs_and_outputs_json.png" width="800"/>
</div>
-->

<div>
<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/01_hello_auxiliary_fields/inputs_and_outputs_json.png" width="800">
</div>




The JSON input file sets AFQMC run parameters and references the locations of the Hamiltonian and trial wavefunction files. It is organized into a hierarchy of input blocks which each control specific
details of the calculation. Understanding the details of the input file is the goal of the [next tutorial](../02_input_file/input_file.ipynb).

**For now, we focus on the `wavefunction` and `hamiltonian` blocks.**

The `wavefunction` input block is used to tell SAFIRE where to find the HDF5 file containing the trial wavefunction.
Similarly, the `hamiltonian` input block is used to tell SAFIRE where to find the HDF5 file containing the Hamiltonian.  

The sample json file that we have provided is reproduced here.

```json
{
"afqmc": {
  "project": {
    "id": "qmc",
    "series": 0
  },
  "execute": {
    "walker_set": {
      "walker_type": "CLOSED"
    },
    "wavefunction": {
      "filename": "files/wfn.h5"
    },
    "hamiltonian": {
      "filename": "files/hamil.h5"
    },
    "timestep": 0.01,
    "steps": 10000,
    "n_walkers_per_mpi_task": 200,
    "seed": 42
  }
}
}
```
  
Notice that the `wavefunction` and `hamiltonian` input blocks point to the HDF5 file where we have saved the RHF ground
state and second-quantized Hamiltonian, respectively.

+++ {"id": "dWRr3WmB-NIw"}







<b>The Outputs
</b>



---
Now that we have seen the inputs of SAFIRE, we will move on to the outputs.





<b> stdout
</b>


---

SAFIRE prints information about the setup of the AFQMC calculation to stdout along with timing information at the end.

We will see the output from SAFIRE throughout the tutorials. Here is a sample header which is printed at the beginning of
the output.



```log

███████╗ █████╗ ███████╗██╗██████╗ ███████╗
██╔════╝██╔══██╗██╔════╝██║██╔══██╗██╔════╝
███████╗███████║█████╗  ██║██████╔╝█████╗  
╚════██║██╔══██║██╔══╝  ██║██╔══██╗██╔══╝  
███████║██║  ██║██║     ██║██║  ██║███████╗
╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝


AF App version: 1.0.0
app git branch: fix_compiler_warnings
app git commit: 83a5e18c30fe824a77185cdb5df6e4dba2918f5c
 AFQMCFactory Project settings:
    -- mixed_precision: false
    -- ncores (local) : 1
    -- n_groups       : 1
    -- id             : qmc
    -- series         : 0
    -- MPI tasks/node : 1
    -- MPI nodes      : 1
    -- MPI tasks      : 1


```




You can see git information from when SAFIRE was compiled.
You can also see MPI / parallelization information.
We will explore the output further in later tutorials.






<b> scalar data file : [id].s[series].scalar.dat
</b>


---



<!--
<div>
<img src="./figs/inputs_and_outputs_scalar.png" width="800"/>
</div>
-->

<div>
<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/01_hello_auxiliary_fields/inputs_and_outputs_scalar.png" width="800">
</div>





A text-based file containing stochastic samples of scalar data, such as the energy.

This file is not meant to be read directly by the user.
Instead, afqmctools provides tools to extract information from this file.
**We will learn how to extract the AFQMC energy from this file later in this tutorial.**

The name of this file is derived from the `id` (string) and `series` (integer) parameters set in the json input file. Setting these parameters allows running multiple AFQMC calculations in a single directory without output file collisions.






<b> observables output HDF5 file : [id].s[series].stat.h5
</b>


---



<!--
<div>
<img src="./figs/inputs_and_outputs_hdf5.png" width="800"/>
</div>
-->

<div>
<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/01_hello_auxiliary_fields/inputs_and_outputs_hdf5.png" width="800">
</div>






An HDF5 file where stochastic samples of non-scalar data, such as the one-body reduced density matrix, are printed by SAFIRE
when requested.

Similar to the `[id].s[series].scalar.dat` file, this file is not meant to be read directly by a user.
Instead, afqmctools provides tools to extract data from this file for you.
You will learn more about this in the [Estimators and post-processing tutorial](../04_using_estimators/04_using_estimators.ipynb).






+++ {"id": "Yhdjs78S-NIx"}



## Running SAFIRE




Now that we have learned what the inputs and outputs of SAFIRE are,
we are ready to learn how to run the SAFIRE executable given the inputs.



### Running SAFIRE in serial

SAFIRE can be invoked from the command line as


```bash
    $ /path/to/SAFIRE/build/bin/safire input.json
```


where `input.json` is the json input file.
The input file can have an arbitrary name, but must end with `.json`.
When you run the above, an AFQMC calculation will be performed based on the contents of input.json,
using the Hamiltonian and trial wavefunction specified there.



**For brevity, we will abreviate `/path/to/SAFIRE/build/bin/safire` as `safire` in this tutorial**.
There are many standard ways to make `safire` accessible from the command line in this way.



You can see the possible command line options for SAFIRE by using the `-h`/`--help` switch.




```bash
    $ safire --help

SAFIRE
Usage:
  /path/to/SAFIRE/build/bin/safire [OPTION...] [optional args]

  -h, --help           print help message
  -v, --version        print version message
      --verbosity arg  0, 1, 2, ...: higher means more (default: 2)
      --debug arg      0, 1, 2, ...: higher means more (default: 0)
      --filenames arg  input filenames
```





### Exercise: Try Running SAFIRE.




Run SAFIRE with the provided input files.
You may use either the code block below, or run SAFIRE in a terminal
as described previously.
In either case, you can find the json input file here

```bash
    /path/to/SAFIRE/tutorials/molecules/files/input.json
```



where `/path/to/SAFIRE` should be replaced with the path where
you have installed SAFIRE.

**The provided json input file assumes that you are running SAFIRE from
`/path/to/SAFIRE/tutorials/mols/01_hello_safire/`**

After running with the default parameters, try changing the verbosity level to
see what changes.

<div class="alert alert-block alert-info">
<b>Note:</b>
    in a jupyter notebook, the `!` symbol will run commands in
    a shell instead of in Python.
</div>



```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: o-XSiSD_-NIx
outputId: 97e5782f-d892-4426-e6d5-e7bbb118cab2
---
!$AFQMC_EXEC files/input.json
```

+++ {"id": "eShMF9Ab-NIy"}

### Post-excercise


You should see the following a the top of the
output after running SAFIRE.

```log
███████╗ █████╗ ███████╗██╗██████╗ ███████╗
██╔════╝██╔══██╗██╔════╝██║██╔══██╗██╔════╝
███████╗███████║█████╗  ██║██████╔╝█████╗  
╚════██║██╔══██║██╔══╝  ██║██╔══██╗██╔══╝  
███████║██║  ██║██║     ██║██║  ██║███████╗
╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝


AF App version: 1.0.0
app git branch: fix_compiler_warnings
app git commit: 83a5e18c30fe824a77185cdb5df6e4dba2918f5c
 AFQMCFactory Project settings:
    -- mixed_precision: false
    -- ncores (local) : 1
    -- n_groups       : 1
    -- id             : qmc
    -- series         : 0
    -- MPI tasks/node : 1
    -- MPI nodes      : 1
    -- MPI tasks      : 1

```




Congratulations, you have now run an AFQMC calculation using SAFIRE!
In the directory that you ran this notebook from, you should see a file called
`qmc.s000.scalar.dat`.
We will return to how to use the tools in `afqmctools` to extract the AFQMC energy later in this tutorial.





+++ {"id": "L0YADhMN-NIy"}

## Running SAFIRE in parallel




### Using MPI


SAFIRE can be run with MPI in the usual ways.
For example, using mpirun, we could run SAFIRE with 64 MPI tasks as

```bash
    $ mpirun -np 64 safire --filenames input.json
```

It is important to note that the number of walkers is specified in the input file
as `n_walkers_per_mpi_task`.
So, the total number of walkers using the sample input file above and 64 MPI tasks
would be

$$
N_\mathrm{walkers} = n_\mathrm{walkers\_per\_mpi} \times n_\mathrm{mpi\_task} = 200 \times 64 = 12800.
$$

For the sample inputs, this is significantly more walkers than necessary,
and for calculations that run on a CPU only build of SAFIRE,
`n_walkers_per_mpi_task` will typically be on the order of 10-50.



### GPU Builds


To run on GPUs, you must compile SAFIRE for the GPU.
See the [documentation](https://users.flatironinstitute.org/~beskridge/auxiliary_fields/getting_started.html#gpu-accelerated-build-at-ccq) for instructions on compiling for GPU.

On GPU builds of SAFIRE, the calculation is more efficient when
the GPU is saturated.
The specific number of walkers needed to achieve this depends on how large the system is and how much memory is available on the GPUs.
For small systems, this could mean using on the order of 1000 walkers or more per task.
For very large systems, this may mean on the order of 10 walkers per MPI task.



+++ {"id": "ifrwtXzA-NIy"}

## Basic post-processing


Now that we've learned how to invoke the SAFIRE executable from the command line in serial, using MPI, and using GPU(s),
we are ready to learn how to extract the AFQMC energy from the outputs using afqmctools.

We previously ran SAFIRE on the Hamiltonian of the STO-3G basis hydrogen dimer
with bondlength $\delta_{H-H} = 1.4 a_0$.
This generated a scalar data file, `qmc.s000.scalar.dat`, which contains
the scalar data that was accumulated during the AFQMC calculation, including the AFQMC energy.

afqmctools provides the `scalar_stats` CLI tool to extract data from the scalar data file.
If you followed the official installation instructions for `afqmctools`, you
should be able to run `$ scalar_stats` from within whichever Python environment you
installed afqmctools in.

By default, `scalar_stats` will compute the average AFQMC energy and the stochastic uncertainty of the
energy based on the samples in the `*.scalar.dat` file.
Below is an example of using `scalar_stats` where we have set the "x axis" to (AFQMC projection) time,
and the equilibration time to 2.0 inverse energy units.

```bash
$ scalar_stats qmc.s000.scalar.dat -s time -e 2.0

====== [analyze_scalar_data Settings] ======

[+] fname            = qmc.s000.scalar.dat
[+] mark_header      = #               
[+] series_column    = time           
[+] nequil           = 2.0             
[+] estimate_equil   = False           
[+] column           = LocalEnergy     
[+] reblock          = 1               
[+] ndiscard         = None            
[+] list             = False           
[+] trace            = False           
[+] append           = None            
[+] dump             = False           
[+] dump_fname       = trace.dat       
[+] verbose          = True            
[+] autocorr         = None            
[+] savefig          = None            
[+] dump_avail_columns = False           

AFQMC Energy    -1.135762 +/-   0.000290 7.94  2.0/100.0
```


You can also view the AFQMC energy as a function of the projection time (sometimes referred to as the "equilibration curve") by
adding the `-t` option to plot the data trace.

Running

```bash
$ scalar_stats qmc.s000.scalar.dat -s time -e 2.0 -t
```

will generate the plot,

<!--
<div>
<img src="./figs/HelloAuxyFields_equil.png" width="600"/>
</div>
-->

<div>
<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/01_hello_auxiliary_fields/HelloAuxyFields_equil.png" width="800">
</div>


One should check the equlibration curve to ensure that the the equilibration
time, `-s time -e 2.0`, is large enough to discard the samples where AFQMC is not sampling from the target many-body wavefunction yet.
On the other hand, the equilibration length should not be so large that it discards equilibrated samples.


<div class="alert alert-block alert-info">
<b>Note:</b>
    Using the `-t` option to plot the trace will only work when running locally;
    you can use the `--savefig [SAVEFIG]` option to save the figure as an image
    in the file "SAVEFIG". The file type is detected from the file extension in
    the name. *.png is generally recommended.
</div>



### Exercise: Try Running scalar_stats




Use the code block below to run scalar_stats on the data that we generated
in the last exercise.
You should get an AFQMC energy of `-1.135762 +/-   0.000290`
if you did not change any settings.

Now, try to changing the equilbration length such that only un-equilibrated samples
are discarded.



```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: -Ej94zXY-NIz
outputId: 74fb0dab-3f45-4fac-ea34-2931de197dcb
---
!scalar_stats qmc.s000.scalar.dat -s time -e 2.0
```

+++ {"id": "Q7NGQf5v-NIz"}



## Post Exercise


If you play around with the equilbration length, you will see that, for very small equilibration length,
the predicted energy will be incorrect.
This is due to the inclusion of un-equilibrated samples in the average.
On the other hand, the average energy should be stable if you use too large of a value for the equilibration length.



+++ {"id": "HxaTJ4Rd-NIz"}

## Some Tutorial Helper Python Functions


Now that we have learned how to extract the AFQMC energy from the outputs of SAFIRE using afqmctools,
we will introduce several tutorial helper functions to invoke SAFIRE and extract the AFQMC energy
from within an interactive Python notebook.

For convenience, we provide two helper functions in the `tutorials_utils` Python package.

1. The `run_afqmc()` function runs the SAFIRE executable in one of the several ways that
we described previously, depending on the `run_mode` parameter.
2. the `get_scratch_dir()` function, will generate a scratch directory and return a Path pointing to the scratch directory.

Together, these functions make it possible to complete the tutorials without leaving the interactive Python notebooks.
We explain each function below.




### get_scratch_dir()


`get_scratch_dir()` will return a reference to the Path of a scratch directory that you get to set.
By default, it will create the directory if it does not already exits.
You can either specify the full path to the scratch directory as,

```python
    from tutorial_utils import get_scratch_dir
    scratch_dir = get_scratch_dir('/path/to/my/scratch/hello_safire')
```

or, you can pass a scratch root directory and scratch directory name as

```python
    from tutorial_utils import get_scratch_dir
    scratch_dir = get_scratch_dir('hello_safire', root_dir='/path/to/my/scratch/')
```

You can tell `get_scratch_dir()` to not attempt to create the directory with

```python
    from tutorial_utils import get_scratch_dir
    scratch_dir = get_scratch_dir('/path/to/my/scratch/hello_safire',create=False)
```




+++ {"id": "N9bQJxCM-NIz"}



### The run AFQMC convenience function


`run_afqmc(input_file=input_file, np=np)` runs the SAFIRE executable as

```bash
    $ mpirun -np [np] safire --filenames [input_file]
```

<div class="alert alert-block alert-warning">
<b>CAUTION:</b>
    The `AFQMC_EXEC` environment variable must be set to the path to the SAFIRE executable
    in order to use `run_afqmc()`.
</div>

Here is a typical example of using `run_afqmc()`.

```python
    from tutorial_utils import run_afqmc, get_scratch_dir

    scratch_dir = get_scratch_dir('/path/to/my/scratch/hello_safire')
    
    run_afqmc(
        run_dir=scratch_dir,                # directory to run SAFIRE in - output files will be there
        input_file="/path/to/inputfile/input.json",
        np=12,                              # number of MPI tasks
        output_file=None                    # optionally direct output to file; if None,
    )                                       # then output will be captured in the jupyter notebook.
```

Where the `np` parameter is forwarded directly to `-np [np]` and
`input_file` is passed to `safire --filenames [input_file]`.

<div class="alert alert-block alert-info">
<b>Note:</b>
    There is no difference between running SAFIRE via `run_afqmc()` or
    directly in the command line other than convenience.
</div>




### (afqmctools) the analyze scalar data function

The same functionality as `$ scalar_stats`, from `afqmctools`, can
be accessed directly in a Python script via the `analyze_scalar_data()` function.
This is a standard component of `afqmctools`, and not just a part of these tutorials.

For convenience, we will favor the use of `analyze_scalar_data()` within interactive Python
notebooks; however, this invokes exactly the same code as using `$ scalar_stats` from
the CLI.
`scalar_stats` can generate plots of the data, both from the CLI and from within Python.
We will make use of this feature during the tutorials to visualize results.




### Exercise: Run the full sample calculation using the convenience wrappers.


Use the code block below to run the entire calculation again.

Just as before, at the end, you should get an AFQMC energy of `-1.135742 +/-   0.000286`
if you did not change any settings.

<div class="alert alert-block alert-info">
<b>Note:</b>
    The python `Path` class from `pathlib`, which is a part of the standard cPython distribution,
    greatly simplifies the process of specifying, and creating directories. We recommend using it
    in these tutorials. See the code block below.
</div>


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: lFXM3f_c-NIz
outputId: 23c92fb6-6385-46fa-a007-1b019b3bffa5
---
from pathlib import Path
import shutil

import numpy as np

from afqmctools.hamiltonian.mol import write_hamiltonian_generic
from afqmctools.wavefunction.mol import write_wfn
from afqmctools.inputs.from_hdf import write_json
from tutorial_utils import run_afqmc, get_scratch_dir
from stats.scalar_dat import analyze_scalar_data

# For you TODO: set a scratch directory for the files that will be generated
home = Path.home()
scratch_dir = get_scratch_dir("hello_safire",home / ".scratch")

# copy the provided files
files_dir = scratch_dir / "files"
files_dir.mkdir(exist_ok=True)
shutil.copy("files/input.json", scratch_dir / "files" )
shutil.copy("files/input.h5", scratch_dir / "files" )
shutil.copy("files/hamil.h5", scratch_dir / "files")

run_afqmc(
    run_dir=scratch_dir,
    input_file="files/input.json",   # the location of the sample input file.
    np=1,                                         # to run in serial just as before!
    output_file=None
)

# post process!
_ = analyze_scalar_data(dict(
    fname = scratch_dir/"qmc.s000.scalar.dat",
    series_column = "time",
    nequil = 5,
    trace = True
))
```

+++ {"id": "gGT0fdst-NIz"}



## Summary





In this tutorial we have learned,

1. what are the inputs and outputs of SAFIRE.
2. how to invoke the SAFIRE executable from the command line in serial, using MPI, and using GPU(s)
3. how to extract the AFQMC energy from the outputs of SAFIRE using afqmctools
4. how to use the tutorial helper functions to perform 2 and 3 from within an interactive Python notebook


