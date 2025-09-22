# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
import numpy
from mpi4py import MPI

#TODO: find this module - it doesn't seem to exist
from pyscf_driver import (
    pyscf_driver_init, 
    pyscf_driver_get_info, 
    pyscf_driver_end,
    pyscf_driver_mp2,
    pyscf_driver_hamil,
    pyscf_driver_mp2no
    )


def qe_driver_init(norb, qe_prefix, qe_outdir, atm_labels,
                   intra_image=MPI.COMM_WORLD, inter_image=None,
                   npools=1, outdir='./qedrv', #remove_dir=True,
                   set_soft_links=True, verbose=True,add_image_tag=True):
    """Initializes the QE driver. Must be called before any routine that calls
       the QE driver is executed. Requires a pre-existing QE successful run.

    Parameters
    ----------
    norb: integer
        Number of orbitals read from QE calculation.
    qe_prefix: string
        prefix parameter from QE run.
    qe_outdir: string
        outdir parameter from QE run. (location of QE files).
    atm_labels: array of strings
        Array containing the species labels.
    intra_image: mpi4py communicator. Default: MPI.COMM_WORLD
        Intra image communicator.
    inter_image: mpi4py communicator. Default: None
        Inter image communicator.
    npools: integer. Default: 1
        Number of QE pools used in the driver.
    outdir: string. Default: ./qedrv
        Output directory of the driver. Does not need to be the same as the QE parameter.
    set_soft_links: Bool. Default: True
        If true, soft links to QE files/foulders from qe_outdir will be placed in outdir.
    verbose: Bool. Default: True
        Sets verbosity in driver.
    add_image_tag: Bool. Default True.
        If True, outdir is modified by adding a tag that identifies the image.
        This is needed if running with multiple images simultaneously,
        otherwise the files from different images might conflict with each other.

    Returns
    -------
    qe_info: Python Dictionary
      Dictonary containing all stored information about the QE driver.
      Contents:
        'species' : string array    # array with species labels.
        'nsp' : integer,            # number of species
        'nat' : integer,            # number of atoms
        'at_id' :  integer array    # array with the ids of atoms in the unit cell.  
        'at_pos' : (nat,3) fp array # array with atom positions
        'nkpts' : integer           # number of kpoints
        'kpts' : (nkpts,3) fp array # k-points
        'latt' : (3,3) fp array     # lattice vectors
        'npwx' : integer            # npwx parameter from QE.
        'mesh' : 3D integer array   # FFT mesh 
        'ngm' : integer             # ngm parameter from QE. 
        'outdir' : string           # Location of folder with driver files. 
    """
#    if remove_dir:
#        assert(set_soft_links)
    assert(intra_image.size%npools==0)
    intra_rank = intra_image.rank
    intra_size = intra_image.size

    if inter_image is not None:
        inter_rank = inter_image.rank
        inter_size = inter_image.size
    else:
        inter_rank = 0
        inter_size = 1

    fname = outdir
    if add_image_tag:
        fname += '.'+str(inter_rank)+'/'
    else:
        fname += '/'
    if intra_rank == 0:
#        if remove_dir:
#            os.system('rm -rf '+fname+'\n')
        if set_soft_links:
            os.system('mkdir '+fname)
            os.system('ln -s ./'+qe_outdir+'/'+qe_prefix+'.xml '+fname+'/'+qe_prefix+'.xml')
            os.system('ln -s ./'+qe_outdir+'/'+qe_prefix+'.save/ '+fname+'/'+qe_prefix+'.save')
    MPI.COMM_WORLD.barrier()

    # initialize driver:
    nkpts, nat, nsp, npwx, ngm, mesh = pyscf_driver_init(inter_size, npools, intra_size/npools,
                                                         norb, qe_prefix, fname, verbose)
    atms = numpy.array(atm_labels)  # don't know how to return an array of strings
    atom_ids,atom_pos,kpts,latt = pyscf_driver_get_info(nat,nsp,nkpts)#,atms)
    atom_pos=atom_pos.T
    kpts=kpts.T
    latt=latt.T
    qe_info = {'species' : atms,
               'nsp' : nsp,
               'nat' : nat,
               'at_id' : atom_ids,
               'at_pos' : atom_pos,
               'nkpts' : nkpts,
               'kpts' : kpts,
               'latt' : latt,
               'npwx' : npwx,
               'mesh' : mesh,
               'ngm' : ngm,
               'outdir' : fname
               }
    if verbose and (MPI.COMM_WORLD.rank==0):
        print("# species = {}".format(qe_info['nsp']))
        print("# atoms = {}".format(qe_info['nat']))
        print("# kpts = {}".format(qe_info['nkpts']))
        print("FFT mesh = {} {} {}".format(qe_info['mesh'][0],qe_info['mesh'][1],qe_info['mesh'][2]))
        print(" Atom species: ")
        print(qe_info['species'])
        print(" Atom positions: ")
        print(qe_info['at_pos'])
        print(" Lattice: ")
        print(qe_info['latt'])
        print(" K-points: ")
        print(qe_info['kpts'])

    return qe_info

def qe_driver_end():
    """Finish and perform clean-up on the QE driver.
    After a call to this routine, further calls to the driver are undefined.
    """
    if(MPI.COMM_WORLD.rank==0):
        print(" Closing QE driver.")
    pyscf_driver_end()

def qe_driver_MP2(qe_info,out_prefix='pyscf_drv',
                        diag_type='keep_occ',
                        nread_from_h5=0,h5_add_orbs='',
                        eigcut=1e-3,nextracut=1e-6,kappa=0.0,regp=0):
    """Call the MP2 routine in the driver.

    Parameters
    ----------
    qe_info: Python Dictionary.
        Dictionary with information from QE calculation, generated by qe_driver_init.
    out_prefix: string. Default: 'pyscf_drv'
        Prefix used in all the files generated by the driver.
    diag_type: string. Default: 'keep_occ'
        Defines the type of HF diagonalization performed before the MP2 calculation.
        Options:
            'keep_occ': Only the virtual orbitals/eigenvalues are calculated.
                        Occupied orbitals/eigenvalues are kept from the QE calculation.
            'full': All orbitals/eigenvalues are recalculated.
            'fullpw': A basis set is generated that contains all the plane waves 
                      below the QE wfn cutoff. The HF eigenvalues/orbitals and MP2NO 
                      are calculated in this basis.
    nread_from_h5: integer. Default: 0
        Number of orbitals to read from h5_add_orbs.  
    h5_add_orbs: string. Default: ''
        Name of hdf5 file with additional orbitals to add to the basis set.
    eigcut: fp number. Default: 1e-3
        Cutoff used during the generation of the spin independent basis in UHF/GHF
        calculations. Only the eigenvalues of the overlap matrix (alpha/beta) 
        above this cutoff are kept in the calculation. In order to reproduce
        the UHF/GHF energy accurately, this number must be set to a small value (e.g. 1e-8). 
    nextracut: fp number. Default: 1e-6
        Cutoff used when adding states from h5_add_orbs to the basis set.
        When a new state from the file is being added to the orbital set, 
        the component along all current orbitals in the set is removed.
        The resulting (orthogonal) state is added only if the norm of the unnormalized
        orbital is larger than nextracut (state is afterwards normalized).
        This is used as a way to remove linear dependencies from the basis set. 
    """
    if diag_type=='fullpw':
        emp2=pyscf_driver_mp2(out_prefix,True,diag_type,
                     0,'',0.0,
                     0.0,kappa,regp)
    else:
        emp2=pyscf_driver_mp2(out_prefix,True,diag_type,
                     nread_from_h5,h5_add_orbs,eigcut,
                     nextracut,kappa,regp)
    return emp2

def qe_driver_MP2NO(qe_info,out_prefix='pyscf_drv',
                        appnos=False,
                        diag_type='keep_occ',
                        nread_from_h5=0,h5_add_orbs='',nskip=0,
                        eigcut=1e-3,nextracut=1e-6,mp2noecut=1e-6,kappa=0.0,regp=0):
    """Call the MP2NO routine in the driver.

    Parameters
    ----------
    qe_info: Python Dictionary.
        Dictionary with information from QE calculation, generated by qe_driver_init.
    out_prefix: string. Default: 'pyscf_drv'
        Prefix used in all the files generated by the driver.
    appnos: Bool. Default: False.
        If True, generates approximate natural orbitals.
    diag_type: string. Default: 'keep_occ'
        Defines the type of HF diagonalization performed before the MP2 calculation.
        Options:
            'keep_occ': Only the virtual orbitals/eigenvalues are calculated.
                        Occupied orbitals/eigenvalues are kept from the QE calculation.
            'full': All orbitals/eigenvalues are recalculated.
            'fullpw': A basis set is generated that contains all the plane waves 
                      below the QE wfn cutoff. The HF eigenvalues/orbitals and MP2NO 
                      are calculated in this basis.
    nread_from_h5: integer. Default: 0
        Number of orbitals to read from h5_add_orbs.  
    h5_add_orbs: string. Default: ''
        Name of hdf5 file with additional orbitals to add to the basis set.
    nskip: integer. Default: 0
        Number of states above the HOMO state of the solid to skip
        during the calculation of MP2 NOs. This can be used to avoid divergencies
        in metals. The assumption being that these states will be included in the 
        orbital set directly.
    eigcut: fp number. Default: 1e-3
        Cutoff used during the generation of the spin independent basis in UHF/GHF
        calculations. Only the eigenvalues of the overlap matrix (alpha/beta) 
        above this cutoff are kept in the calculation. In order to reproduce
        the UHF/GHF energy accurately, this number must be set to a small value (e.g. 1e-8). 
    nextracut: fp number. Default: 1e-6
        Cutoff used when adding states from h5_add_orbs to the basis set.
        When a new state from the file is being added to the orbital set, 
        the component along all current orbitals in the set is removed.
        The resulting (orthogonal) state is added only if the norm of the unnormalized
        orbital is larger than nextracut (state is afterwards normalized).
        This is used as a way to remove linear dependencies from the basis set. 
    mp2noecut: fp number. Default: 1e-6
        Cutoff used when adding natural orbitals from the MP2 RDM,
        only states with eigenvalue > mp2noecut will be kept.
        If this number is < 0.0, then a specific number of states is kept and is 
        given by nint(-mp2noecut). 
 
    """
    if diag_type=='fullpw':
        pyscf_driver_mp2no(out_prefix,True,diag_type,appnos,
                     0,'',nskip,0.0,
                     0.0,mp2noecut,kappa,regp)
    else:
        pyscf_driver_mp2no(out_prefix,True,diag_type,appnos,
                     nread_from_h5,h5_add_orbs,nskip,eigcut,
                     mp2noecut,nextracut,kappa,regp)

def qe_driver_hamil(qe_info,out_prefix='pwscf',
                    nread_from_h5=0,h5_add_orbs='',ndet=1,eigcut=1e-3,
                    nextracut=1e-6,thresh=1e-5,ncholmax=15,get_hf=True,
                    get_mp2=True,update_qe_bands=False):
    """Call the MP2 routine in the driver.

    Parameters
    ----------
    qe_info: Python Dictionary.
        Dictionary with information from QE calculation, generated by qe_driver_init.
    out_prefix: string. Default: 'pyscf_drv'
        Prefix used in all the files generated by the driver.
    nread_from_h5: integer. Default: 0
        Number of orbitals to read from h5_add_orbs.  
    h5_add_orbs: string. Default: ''
        Name of hdf5 file with additional orbitals to add to the basis set.
    ndet: integer. Default: 1
        Maximum number of determinants allowed in the trial wave-function.
    eigcut: fp number. Default: 1e-3
        Cutoff used during the generation of the spin independent basis in UHF/GHF
        calculations. Only the eigenvalues of the overlap matrix (alpha/beta) 
        above this cutoff are kept in the calculation. In order to reproduce
        the UHF/GHF energy accurately, this number must be set to a small value (e.g. 1e-8). 
    nextracut: fp number. Default: 1e-6
        Cutoff used when adding states from h5_add_orbs to the basis set.
        When a new state from the file is being added to the orbital set, 
        the component along all current orbitals in the set is removed.
        The resulting (orthogonal) state is added only if the norm of the unnormalized
        orbital is larger than nextracut (state is afterwards normalized).
        This is used as a way to remove linear dependencies from the basis set. 
    thresh: floating point. Detault: 1e-5
        Value used to stop the iterative calculation of Cholesky vectors. The iterations
        stop when the error on a diagonal element falls below this value. 
    ncholmax: integer. Default: 15
        Maximum number of Cholesky vectors allowed (in units of the number of orbitals).
        If the iterative calculation has not converged when this number of Cholesky vectors
        is found, the calculation stops.
    get_hf: Bool. Default: True
        If True, calculate the HF eigenvalues/eigenvectors.
    get_mp2: Bool. Default: True
        If True, calculate the MP2 energy. (If True, get_hf will be set to true.)
    update_qe_bands: Bool. Default: False
        If True, the orbitals in the QE restart file will overwritten with the
        basis set generated by the driver. Orbitals beyond norb (set in qe_driver_init)
        will be left unmodified. 

    Returns
    -------
    ehf: floting point
        The HF energy on this basis. (return 0.0 if not requested)
    emp2: floting point
        The MP2 energy on this basis. (return 0.0 if not requested)
    """
    if get_mp2:
        get_hf = True
    ehf, emp2 = pyscf_driver_hamil(out_prefix, nread_from_h5, h5_add_orbs,
             ndet, eigcut, nextracut, thresh, ncholmax,
             get_hf, get_mp2, update_qe_bands)
    return ehf, emp2
