# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from datetime import datetime

import sys
from pathlib import Path
import argparse
import enum

import subprocess as sp

import afqmctools.hamiltonian.model.director as ham
from afqmctools.wavefunction.model import write_free_electron_wfn
from autohf.solver import lattice_hf
import afqmctools.utils.io as io
from afqmctools.inputs.from_hdf import write_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input',
        metavar='input',
        help="Input file containing benchmark params (toml)",
        type=str,
        default="benchmark.toml"
    )
    parser.add_argument(
        '--plotlat','-pl',
        help="plot the lattice",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--plot1b',
        help="plot the 1-body Hamiltonian",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--plot2b',
        help="plot the 2-body Hamiltonian",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--saveplots','-sp',
        help="save plots instead of displaying (useful for remote connections)",
        action="store_true",
        default=False
    )
    return parser.parse_args()

class WfnType(enum.Enum):
    FreeElectron = enum.auto()
    HatreeFock = enum.auto()
    Unknown = enum.auto()

def get_trial_wfn_type(type_name):

    if isinstance(type_name, WfnType):
        return type_name

    if type_name.lower() in {"free-electron", "free_electron", "fe"}:
        return WfnType.FreeElectron
    elif type_name.lower() in {"hartree-fock", "hartree_fock", "hf"}:
        return WfnType.HatreeFock
    else:
        return WfnType.Unknown


def get_input_params(fname):
    return io.read_input_params(fname)


def system_info():
    """
    Check info about the system:
    - Python version
    - AFQMC code: what branch, commit, etc.
    """
    print("Running benchmark on: ", datetime.now())
    print("Python version info: ", sys.version_info)
    # TODO: get info on AFQMC executable - git commit hash, git branch, md5sum of qmcapp

def main():

    system_info()

    args = parse_args()
    input_file = args.input
    input_params = get_input_params(input_file)

    hamiltonianDir = ham.InputFileDirector(
        source=input_file
    )
    hamiltonian = hamiltonianDir.build()

    if args.plotlat:
        import afqmctools.utils.visualize as vis
        vis.plot_lattice(
            lattice=hamiltonianDir.builder.lattice,
            save=args.saveplots
        )
    
    misc_params = input_params["misc_params"]
    nelec = misc_params["nelec"]
    spin_symm = misc_params["spin_symm"]
    hdf5_hamil_outfile = hdf5_wfn_outfile = misc_params["hdf5_outfile"]
    

    io.write_model_hamiltonian(
        hamiltonian=hamiltonian,
        fname=hdf5_wfn_outfile,
        nelec=nelec,
        spin_symm=spin_symm
    )
    
    trial_wfn_type = get_trial_wfn_type(input_params["trial_wavefunction"]["type"])

    if trial_wfn_type == WfnType.FreeElectron:
        write_free_electron_wfn(
            hamiltonian_fname=hdf5_wfn_outfile,
            nelec=nelec,
            spin_symm=spin_symm
            )

    elif trial_wfn_type == WfnType.HatreeFock:
        hf_settings = input_params["trial_wavefunction"]["hartree_fock"]
        if "hamiltonian" in input_params["trial_wavefunction"]:
            # save the lattice and construct Heff for HF
            new_in = {"lattice":input_params["lattice"],
                     "hamiltonian":input_params["trial_wavefunction"]["hamiltonian"]}
            hamiltonianDir = ham.InputFileDirector(
               source=new_in
            )
            heff = hamiltonianDir.build()
            hf_hamiltonian = heff
        else:
            hf_hamiltonian = hamiltonian
        if "nelec" not in hf_settings:
            hf_settings["nelec"] = nelec
        lattice_hf(
            hamiltonian=hf_hamiltonian,
            lattice=hamiltonianDir.builder.lattice,
            settings=hf_settings
        )
        if "output" in hf_settings:
            hdf5_wfn_outfile = hf_settings["output"]


    afqmc_params = input_params["afqmc"]
    write_json("afqmc.json",fwfn0=hdf5_wfn_outfile,fham0=hdf5_hamil_outfile,exec_opts=afqmc_params)

    #print("Launching AFQMC code...",flush=True)
    #afqmc_run_mode = misc_params["afqmc_run_mode"]
    #if afqmc_run_mode in {'no_run','no run', 'none'}:
    #    return
    #else:
    #    with open(misc_params["stdout"],'w') as f:
    #        if afqmc_run_mode == "slurm_cpu":
    #            sp.run(["srun","--cpu-bind=cores",AFQMC_EXEC,"--filenames","afqmc.json"],stdout=f,stderr=f)
    #        elif afqmc_run_mode == "local":
    #            sp.run([AFQMC_EXEC,"--filenames","afqmc.json"],stdout=f,stderr=f)
    #        elif afqmc_run_mode == "mpi":
    #            nproc = str(int(misc_params["nproc"]))
    #            sp.run(["mpirun","-n",nproc,AFQMC_EXEC,"--filenames","afqmc.json"],stdout=f,stderr=f)
    #        else:
    #            raise ValueError("Invalid value for 'afqmc_run_mode'. Supported choices are: 'no run', 'slrum_cpu', 'local'")


            
if __name__ == '__main__':
    main()
