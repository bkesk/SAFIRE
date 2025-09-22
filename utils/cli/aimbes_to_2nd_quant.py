# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import argparse
from itertools import product

import numpy as np
import h5py as h5
from pyscf import gto,scf

from afqmctools.hamiltonian.converter import write_fcidump
from afqmctools.utils.aimbes_utils import (
    read_aimbes_dm,
    check_aimbes_energy,
    read_aimbes_Hamiltonian
)
from afqmctools.utils.linalg import modified_cholesky_direct as cholesky
from afqmctools.hamiltonian.io import write_dense

from afqmctools.utils.io import to_complex

from afqmctools.wavefunction.common import write_wfn

def parse_args():
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
            command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
            Command line arguments.
    """

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('-i', '--input', dest='aimbes_file', type=str,
                                            default=None, help='Input AIMBES h5 file.',
                                            required=True)
    parser.add_argument('-o', '--output', dest='output_file',
                                            type=str, default='FCIDUMP',
                                            help='Output AFQMC h5 file.')
    parser.add_argument('-cd','--cholesky', dest='use_cholesky',
                            action='store_true', default=True,
                            help='use a Cholesky decomposition on two-body interaction')
    parser.add_argument('-t', '--tol', dest='tol',
                    type=float, default=1e-6,
                    help='Cholesky tolerance for two-body interactions')
    parser.add_argument('-v', '--verbose', dest='verbose',
                                            action='store_true', default=False,
                                            help='Verbose output.')
    parser.add_argument('-na',dest="na",type=int,
                            help='Number of spin-up electrons',
                            required=True)
    parser.add_argument('-nb',dest="nb",type=int,
                            help='Number of spin-down electrons',
                            required=True)
    parser.add_argument('-dt','--type',dest="downfold_type",
                            type=str, default='gw',
                            help='type of 1-body downfolding. options: dft, gw')
    parser.add_argument('-d', '--double-counting', dest='use_double_counting',
                                            action='store_true', default=True,
                                            help='Include double counting correction?')
    parser.add_argument('-r','--real',dest='real_all',
                                            action='store_true', default=False,
                                            help='force all integrals to be real valued'
                                            )
    parser.add_argument('-r1','--real1',dest='real_1body',
                                            action='store_true', default=False,
                                            help='force 1-body integrals to be real valued'
                                            )
    parser.add_argument('-r2','--real2',dest='real_2body',
                                            action='store_true', default=False,
                                            help='force 2-body integrals to be real valued'
                                            )
    parser.add_argument('--bands',dest="bands",type=str,
                            default=None,
                            help="range of input bands from Wannier90 format is `lower:upper` "
                        )
    
    # FCIDUMP settings
    parser.add_argument('-u','--use_spinors',dest='use_spinors',
                            action='store_true', default=False,
                            help='Whether to convert to spinor basis'
                        )
    parser.add_argument('-c', '--complex', dest='cplx',
                            action='store_true', default=False,
                            help='Whether to write integrals as complex numbers.'
                        )
    parser.add_argument('-s', '--symmetry', dest='symm',
                            type=int, default=1,
                            help='Symmetry of integral file (1,4,8).'
                        )
    parser.add_argument('--complex-paren', dest='cplx_paren',
                        action='store_true', default=False,
                        help='Whether to write FORTRAN format complex numbers.')

    # HF basis
    parser.add_argument('--canonical',dest='use_canonical_basis',
                        action='store_true', default=False,
                        help='if true, run Hatree-Fock(HF) and transform to the canonical HF basis')
    parser.add_argument('--sc-iter', dest='sc_iter',type=int, default=1,
                        help="(in combination with the `--canonical` option) which self-consistent "
                        "iteration to get initial guess from. default = 1")
    parser.add_argument('--density-input', dest='aimbes_file_for_density', type=str,
                            default=None, help='AIMBES h5 file to source density matrix from')
    # TODO: the default should be to use AFQMC format, and using FCIDUMP/SHCI should not be the default!
    parser.add_argument('--afqmc',dest="use_afqmc",
                        action="store_true",default=False,
                        help="save Hamiltonian in AFQMC format")
    parser.add_argument('--as-kpoint',dest='as_kpoint',
                        action='store_true',default=True,
                        help="(with --afqmc) caste the Hamiltonian as a k-point factorized Hamiltonian with a single k-point")
    
    parser.add_argument('--ncore',dest="ncore",
                        action="store",default=0,
                        help="Number of orbitals to treat as core")
    parser.add_argument('--nactive',dest="nactive",
                        action="store",default=None,
                        help="Number of orbitals to treat as active (by default, all orbitals are active)")
    
    options = parser.parse_args()

    if options.real_all:
        options.real_1body = True
        options.real_2body = True

    return options

# TODO: move once this script is working
def h5_overwrite(f,name,data):
    if name in f.keys():
        del f[name]
    f.create_dataset(name,data=data)

# TODO: move once this script is working
def write_as_k_point(fname, L, H1, nelec):
    base = "/Hamiltonian"

    gamma= np.array([0.,0.,0.])
    nchol = L.shape[-1]
    nmo = np.sqrt(L.shape[0]).astype(np.int64)

    na,nb = nelec

    if np.iscomplexobj(L):
        complex_ints = 1
    else:
        complex_ints = 0

    dims = np.array([0,0,1,nmo,na,nb,0,nchol])

    L = L.reshape(1,nmo*nmo*nchol)

    with h5.File(fname, 'a') as f:
        h5_overwrite(f,base+"/KPFactorized/L0",data=to_complex(L))
        h5_overwrite(f,base+"/NCholPerKP",data=np.array([nchol]))
        h5_overwrite(f,base+"/MinusK",data=np.array([0]))
        h5_overwrite(f,base+"/NMOPerKP",data=np.array([nmo]))
        h5_overwrite(f,base+"/QKTok2",data=np.array([[0]]))
        h5_overwrite(f,base+"/H1_kp0",data=to_complex(H1))
        h5_overwrite(f,base+"/ComplexIntegrals",data=complex_ints)
        h5_overwrite(f,base+"/KPoints",data=np.array([gamma]))
        h5_overwrite(f,base+"/dims",data=dims)
        h5_overwrite(f,base+"/Energies",data=np.array([0.0,0.0]))

## Start: Copied from Paul #######################
# these were meant for tiny Hamiltonians - WAAAY too slow for reasonably sized basis..
def translate_h1(h1):
  nmo = len(h1)
  dtype = h1.dtype
  rh1 = np.zeros([2*nmo, 2*nmo], dtype=dtype)
  for i, j in product(range(2*nmo), repeat=2):
    i1 = i // 2
    ispin = i % 2
    j1 = j // 2
    jspin = j % 2
    rh1[i, j] = h1[i1, j1] if ispin == jspin else 0
  return rh1

def translate_h2(h2):
  nmo = len(h2)
  dtype = h2.dtype
  rh2 = np.zeros([2*nmo, 2*nmo, 2*nmo, 2*nmo], dtype=dtype)
  for i, k, j, l in product(range(2*nmo), repeat=4):
    i1 = i // 2
    ispin = i % 2
    k1 = k // 2
    kspin = k % 2
    if ispin != kspin: continue
    j1 = j // 2
    jspin = j % 2
    l1 = l // 2
    lspin = l % 2
    if jspin != lspin: continue
    rh2[i, k, j, l] = h2[i1, k1, j1, l1]
  return rh2
## End: Copied from Paul #########################

def wan2can_1body(D,H1):
    print("Transforming 1-body Hamiltonian to new basis",flush=True)
    return (D.conj().T @ H1 ) @ D

def wan2can_2body(D,V):
    print("Transforming 2-body Hamiltonian to new basis... ",flush=True)
    print("... transorming first index i -> mu ",flush=True)
    V = np.einsum('mi,ikjl->mkjl',D.conj().T,V)
    print("... transorming second index k -> delta ",flush=True)
    V = np.einsum('kd,mkjl->mdjl',D,V)
    print("... transorming third index j -> nu ",flush=True)
    V = np.einsum('mdjl,nj->mdnl',V,D.conj().T)
    print("... transorming fourth index l -> gamma ",flush=True)
    V = np.einsum('mdnl,lg->mdng',V,D)
    print("... finished transforming 2-body Hamiltonian.",flush=True)
    return V

def main():
    """Convert AIMBES Hamiltonians saved in an hdf5 checkpoint file
        to a FCIDUMP file.

    Parameters
    ----------
    args : list of strings
            command-line arguments.
    """
    options = parse_args()

    H1,Lvecs,_ = read_aimbes_Hamiltonian(
        fname=options.aimbes_file,
        chol_tol=options.tol,
        sc_iter=options.sc_iter
    )
    nelec = (options.na,options.nb)
    nmo = H1.shape[0]

    Lvecs_dagger = np.transpose(Lvecs,axes=(0,2,1)).conj()
    V_abcd = np.einsum("nab,ncd->abcd",Lvecs,Lvecs_dagger)
    del Lvecs_dagger

    if options.aimbes_file_for_density is None: # TODO: rename to aimbes_file_for_energy
        aimbes_file_for_density = options.aimbes_file
    else:
        aimbes_file_for_density = options.aimbes_file_for_density

    measure_energy = all((options.downfold_type != 'gw',))

    if measure_energy:
        print("Checking initial variational energy based on DFT density")
        check_aimbes_energy(
            aimbes_file=aimbes_file_for_density,
            H1=H1,
            V_abcd=V_abcd,
            scf_iter=0,
            mode='variational'
        )
        if options.sc_iter > 0:
            print("Checking initial HF energy using AIMBES' conventions")
            check_aimbes_energy(
                aimbes_file=aimbes_file_for_density,
                H1=H1,
                V_abcd=V_abcd,
                scf_iter=options.sc_iter
            )

    # (OPTIONAL) Run HF and transform to canonical HF basis
    if options.use_canonical_basis:
        print("Transforming to Canonical HF basis")
            
        fake_mol = gto.M()
        fake_mol.nelectron = sum(nelec)
        fake_mol.verbose = 6

        if options.sc_iter >=0:
            rdm0 = read_aimbes_dm(aimbes_file_for_density,scf_iter=options.sc_iter)
        else:
            rdm0 = np.zeros((nmo,nmo),dtype=np.complex128)
            for a in range(nelec[0]):
                rdm0[a,a] += 1.0
            for b in range(nelec[1]):
                rdm0[b,b] += 1.0

        mf = scf.RHF(fake_mol)
        mf.get_hcore = lambda *args: H1
        mf.get_ovlp = lambda *args: np.eye(nmo)
        mf.chkfile = "basis.chk"
        mf._eri = V_abcd
        mf.kernel(dm0=rdm0)

        D = mf.mo_coeff

        H1 = wan2can_1body(D,H1)
        V_abcd = wan2can_2body(D,V_abcd)

        # save the ROHF det as NOMSD
        wfn = np.zeros((nmo,sum(nelec)),dtype=np.complex128)
        wfn[:nelec[0],:nelec[0]] = np.eye(nelec[0],dtype=np.complex128)
        wfn[:nelec[1],nelec[0]:nelec[0]+nelec[1]] = np.eye(nelec[1],dtype=np.complex128)

        write_wfn(
            "rohf_wfn_afqmc.h5", 
            (np.array([1.0+0j]),np.array([wfn])), 
            'collinear',
            nelec, 
            nmo
        )

        print("\n ====== Generating UHF trial wavefunction ====== ")
        #TODO" Make this optional later!
        mf = mf.to_uhf()
        mf.chkfile = "uhf.h5"
        mf.kernel(dm0=mf.make_rdm1())

        uhf_orbitals = mf.mo_coeff

        wfn = np.zeros((nmo,sum(nelec)),dtype=np.complex128)
        wfn[:,:nelec[0]] = uhf_orbitals[0,:,:nelec[0]]
        wfn[:,nelec[0]:nelec[0]+nelec[1]] = uhf_orbitals[0,:,:nelec[1]]

        write_wfn(
           "uhf_wfn_afqmc.h5", (
            np.array([1.0+0j]),np.array([wfn])), 
            'collinear',
            nelec,
            nmo
        )

    wfn = np.zeros((nmo,sum(nelec)),dtype=np.complex128)
    wfn[:nelec[0],:nelec[0]] = np.eye(nelec[0],dtype=np.complex128)
    wfn[:nelec[1],nelec[0]:nelec[0]+nelec[1]] = np.eye(nelec[1],dtype=np.complex128)

    write_wfn(
        "diagonal_wfn_afqmc.h5", 
        (np.array([1.0+0j]),np.array([wfn])), 
        'collinear',
        nelec,
        nmo
    )

    if options.use_afqmc:
        # Need the Hermitian form for Cholesky -> transpose first two axes!
        #Lvecs = cholesky(
        #    np.transpose(V_abcd,(1,0,2,3)).reshape(nmo**2, nmo**2),
        #    tol=options.tol,
        #    verbose=True
        #).T.conj() # NOTE: the conj is important!!

        if not options.as_kpoint and not np.iscomplexobj(Lvecs):
            write_dense(
                filename=options.output_file,
                hcore=H1,
                chol=Lvecs,
                nelec=nelec,
                nmo=H1.shape[-1],
                real_chol = True,
            )
        else:
            nchol = Lvecs.shape[0]
            write_as_k_point(
                fname=options.output_file,
                L=np.transpose(Lvecs,axes=(1,2,0)).reshape(nmo**2,nchol),
                H1=H1,
                nelec=(options.na,options.nb)
            )
    else:
        if options.cplx:
            V_abcd = translate_h2(V_abcd)
            H1 = translate_h1(H1)
            nmo = nmo*2

        print("Writting to FCIDUMP", flush=True)
        write_fcidump(
            filename=options.output_file,
            hcore=H1,
            chol=V_abcd,
            chol_is_eri=True,
            enuc=0.0, # TODO: add option to specify E0
            nmo=nmo,
            nelec=nelec,
            tol=options.tol,
            sym=options.symm,
            cplx=options.cplx,
            paren=options.cplx_paren,
            use_spinor=options.use_spinors
        )


if __name__ == '__main__':
    main()
