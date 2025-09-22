# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
Utilities for Interfacing with AIMBES
"""
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
import enum
from warnings import warn

import h5py as h5
import numpy as np
import numba

from afqmctools.utils.linalg import modified_cholesky_direct as cholesky
from afqmctools.utils.io import to_complex, h5_as_dict
from afqmctools.utils.chemistry import atomic_symbol_to_num

class AIMBESHamiltonian(enum.Enum):
    """
    Enumerated type corresponding to the Hamiltonian types from AIMBES.
    """

    BARE=enum.auto() # i.e. from the 'hamil' path in aimbes
    FROZEN_CORE=enum.auto() 
    CRPA=enum.auto()


def get_aimbes_hamil_enum(hamil_type:str|AIMBESHamiltonian) -> AIMBESHamiltonian:
    """Maps general inputs into the AIMBESHamiltonian enumerated type

    Parameters
    ----------
    hamil_type : AIMBESHamiltonian | str
        A description of the Hamiltonian type. If an AIMBESHamiltonian enum is provided,
        it is immediately returned. If a string is provided, a corresponding AIMBESHamitlonian
        enum is returned. See the table above for the mapping from strings to AIMBESHamiltonian
    
    Returns
    -------
    AIMBESHamiltonian
        a AIMBESHamiltonian enum corresponding to the input hamil_type parameter

    Notes
    -----
    Input strings are not case sensitive and acceptable input strings include:

    +-------------------+------------------------------------------------------------+
    | AIMBESHamiltonian |                          strings                           |
    +===================+============================================================+
    |        BARE       | "bare", "hamil", "hamiltonian", "hamil_bare", "hamil bare" |
    +-------------------+------------------------------------------------------------+
    |    FROZEN_CORE    | "frozen_core", "frozen core", "mf_downfold", "mf downfold" |
    +-------------------+------------------------------------------------------------+
    |       CRPA        |          "downfolded", "crpa", "crpa downfolded"           |
    +-------------------+------------------------------------------------------------+
    """
    if isinstance(hamil_type,AIMBESHamiltonian):
        return hamil_type
    elif not isinstance(hamil_type,str):
        logger.dev(f"hamil_type is {type(hamil_type)}, which is not a string or an AIMBESHamiltonian instance")
        raise ValueError(
            "[for developers] hamil_type must be either a str "
            "or an AIMBESHamiltonian instance"
        )
    
    if hamil_type.lower() in ("bare","hamil","hamiltonian","hamil_bare","hamil bare"):
        return AIMBESHamiltonian.BARE
    elif hamil_type.lower() in ("frozen_core","frozen core","mf_downfold","mf downfold"):
        return AIMBESHamiltonian.FROZEN_CORE
    elif hamil_type.lower() in ("downfolded","crpa","crpa downfolded"):
        return AIMBESHamiltonian.CRPA
    else:
        raise ValueError(f"invalid AIMBES Hamiltonian type {hamil_type}")


@dataclass
class TRIQS_HDF5_File:
    """
    Dataclass to contain orbitals and relevant data
      from the TRIQS internal format.

    Attributes
    ----------
    name : str
        The name of the file that Wannier orbitals were read from.
    kpts : np.array
        Array containing all k-points in fractional coordinates
    proj_mat : np.array
        Has shape Nspin x Nkpt x Ndefect x Norb x Nband ). Defines the 
        projection from the Kohn-Sham (KS) band basis into
        the basis defined in the file called "name"
    band_window : np.array
        An array containing the range of KS bands used as inputs to the 'proj_mat'
    wan_centres : np.array | None
        (if applicable) contains the spatial centers of Wannier orbitals. These are
        used within AIMBES to translate Wannier orbitals into the home cell. Do not 
        set these if you aren't using Wannier orbitals!
    """

    name:str
    kpts:np.ndarray
    proj_mat:np.ndarray
    band_window:np.ndarray
    wan_centres:np.ndarray


def read_triqs_hdf5(fname)->TRIQS_HDF5_File:
    """Read a TRIQS_HDF5_File from fname."""
    f = h5.File(fname,'r')
    dft_g = f["dft_input"]
    kpts = dft_g["kpts"][...]           # all k-points in fraction coords
    C_ksIai = dft_g["proj_mat"][...]    # Projection matrix
    if "wan_centres" in dft_g:
        wan_centres = dft_g["wan_centres"][...]
    else:
        wan_centres = None
    
    C_ksIai = C_ksIai[:,:,:,:,:,0] + 1j*C_ksIai[:,:,:,:,:,1] 
    g = f["dft_misc_input"]
    band_window = g["band_window"][...]    

    return TRIQS_HDF5_File(fname,kpts,C_ksIai,band_window,wan_centres)


def check_aimbes_energy(aimbes_file,H1,V_abcd,scf_iter=1,mode="mixed"):
    """
    Calculates and prints the "Hartree-Fock" energy based on the density
    in the given AIMBES checkpoint file, 'aimbes_file', and provided
    Hamiltonian.

    Can evaluate the energy in two modes:
    
    - (defualt) : 'mixed_mode' mimics the output of AIMBES. The energy 
      at iteration n is defined as:
      
      E^{mixed HF}_{n} = Tr[ D_{n} H1 ] + Tr[ F[ D_{n-1} ] D_{n} ]
    
    - 'variational' : uses the same density for building the Fock matrix
      and for evaluating the energy
      
      E^{mixed HF}_{n} = Tr[ D_{n} H1 ] + Tr[ F[ D_{n} ] D_{n} ]

    Convenience function to calculation the energy that AIMBES will output.
    AIMBES computes the "Hartree-Fock" energy at iteration `m+1` by:
    - computing the Fock matrix (F) with the Dm at iteration `m`
    - diagonalize F -> generate new density matrix
    - evaluate the energy with the same F[D_{m}] but using the new D_{m+1}     
    
    TODO: for cRPA and frozen core, we need to be able to map the density matrix
           into the local Hilbert space.
    """
    if mode == "mixed":
        use_mixed = True
        print("Using 'mixed' energy evaluation as in AIMBES")
    elif mode == "variational":
        use_mixed = False
        print("Using 'variational' energy evaluation")
    else:
        raise ValueError(
            f"Invalid energy evaluation mode {mode}: "
            "choices are 'mixed' (default), or 'variational'"
        )

    if use_mixed and scf_iter < 1:
        raise ValueError("Mixed density Energy evaluation is only available for iteration 1 or later")

    print("Evaluating HF Energy as in AIMBES")
    print(f"  getting density matrices from aimbes file: {aimbes_file}")
    print(f"  energy is evaluated using the density from iteration {scf_iter}")

    if use_mixed:
        rdm0,rdm1  = [
            read_aimbes_dm(fname=aimbes_file,scf_iter=scf_iter-1),
            read_aimbes_dm(fname=aimbes_file,scf_iter=scf_iter)
            ]
    else:
        rdm0 = rdm1 = read_aimbes_dm(fname=aimbes_file,scf_iter=scf_iter)

    vj = np.einsum('ijkl,ji->kl', V_abcd, rdm0)
    vk = -0.5*np.einsum('ijkl,jk->il', V_abcd, rdm0)

    E1 = np.trace(H1@rdm1)
    EJ = 0.5*np.trace(vj@rdm1)
    EK = 0.5*np.trace(vk@rdm1)

    print(f"Aimbes E total = {E1+EJ+EK}")
    print(f"  E 1-body = {E1}")
    print(f"  E 2-body = {EJ+EK}\n   -- EJ ={EJ}\n   --EK = {EK}")


def read_aimbes_dm(fname,scf_iter=0):
    """
    Read the density matrix from an AIMBES output file.

    By default, reads the initial density matrix - i.e. `scf_iter=0`
    """
    dm_name = f"/scf/iter{scf_iter}/Dm_skij"
    with h5.File(fname,'r') as f:
        Dm_skij = f[dm_name][...]
        if Dm_skij.shape[0] == 1 and Dm_skij.shape[1] == 1 :
            spin_factor = 2.0
            return spin_factor*(Dm_skij[0,0,:,:,0] + 1j*Dm_skij[0,0,:,:,1])

        elif Dm_skij.shape[0] == 2 and Dm_skij.shape[1] == 1 :
            return [
                Dm_skij[0,0,:,:,0] + 1j*Dm_skij[0,0,:,:,1],
                Dm_skij[1,0,:,:,0] + 1j*Dm_skij[1,0,:,:,1]
                ]
        else:
            raise NotImplementedError("read_aimbes_dm is currently only implemented for single k-points")
        

def read_coqui_orbitals(forbs):
    """Extract orbitals and metadata from COQUÍ HDF5 file.

    Parameters
    ----------
    forbs : str
        orbital HDF file from Coqui

    Returns
    -------
    meta : dict
        metadata
    orbs : numpy.ndarray
        orbitals
    """
    bas_dict = h5_as_dict(forbs)
    orbd = bas_dict["Orbitals"]
    mesh = orbd["fft_mesh"]

    sysd = bas_dict["System"]
    latvec = sysd["lattice_vectors"]
    recvec = sysd["reciprocal_vectors"]

    specs = sysd["species"]
    elems = [atomic_symbol_to_num(specs[i].decode()) for i in sysd["atomic_id"]]
    pos = sysd["atomic_positions"]

    kpoints = sysd["BZ"]["kpoints"]
    nkpts = kpoints.shape[0]

    meta = {
        "mesh": mesh,
        "elem": elems,
        "pos": pos,
        "cell": latvec,
        "recvec": recvec,
        "kpoints": kpoints,
        "nkpts": nkpts
    }

    orbs = []
    if nkpts == 1:
        orbs.append(orbd["psi_s0_k0"].view(np.complex128).squeeze())
        meta["gvecs_k0"] = orbd["miller_wfc"]
    else:
        for k in range(nkpts):
            orbs.append(orbd[f"psi_s0_k{k}"].view(np.complex128).squeeze())
            meta[f"gvecs_k{k}"] = orbd[f"miller_wfc_k{k}"]
    
    return meta, orbs


def write_downfolding_orbitals(fname,kpts,C_ksIai,band_window,wan_centres=None):
    """
    Save orbitals in Wannier90's h5 format to be read into AIMBES
    """
    with h5.File(fname,'w') as f:
        g = f.create_group("dft_input")
        g.create_dataset("kpts",data=kpts)
        ds = g.create_dataset("proj_mat",data=to_complex(C_ksIai))
        ds.attrs["__complex__"] = "1"
        if wan_centres is not None:
            g.create_dataset("wan_centres",data=wan_centres)

        g = f.create_group("dft_misc_input")
        g.create_dataset("band_window",data=band_window)


def aimb_1body_2_afqmc(fname,type="gw",gw_iteration=1,include_dc=True,input_band_range=None,force_real=False,nactive=None,ncore=0):
    """
    Generate the 1-body Hamiltonian from am AIMBES checkpoint file.
    
    There are a few variants of the 1-body Hamiltonian to treat.
    All variants include the terms: `Heff_base_sIab = H0_sIab - mu * I`.

    The variants are:
    1. for DFT-based downfolding, we have `Heff_sIab = Heff_base_sIab + (Vhf_gw_sIab - Vhf_dc_sIab)`
    2. for GW-based downfolding, we have  `Heff_sIab = Heff_base_sIab + (Vhf_gw_sIab - Vhf_dc_sIab) + (Vcorr_gw_sIab - Vcorr_dc_sIab)`     
    """
    if ncore > 0:
        raise NotImplementedError("Applying frozen core via Python is not yet implemented")

    if type.lower() in ("dft","pbe"):
        return _aimb_1body_2_afqmc_dft(fname,gw_iteration=0,include_dc=include_dc,force_real=force_real)
    elif type.lower() in ("gw","correlated"):
        return _aimb_1body_2_afqmc_corr(
            fname,
            gw_iteration=gw_iteration,
            include_dc=include_dc,
            force_real=force_real,
            nactive=nactive,
            ncore=ncore
        )
    elif type.lower() in ("fc","frozen_core"):
        return _aimb_1body_2_afqmc_fc(
            fname,
            input_band_range=input_band_range,
            gw_iteration=gw_iteration,
            force_real=force_real
        )
    elif type.lower() in ("gw_fc","gw_frozen_core"): #TODO: remove this option after test!
        return _aimb_1body_2_afqmc_corr_fc(
            fname,
            gw_iteration=gw_iteration,
            include_dc=include_dc,
            force_real=force_real,
            nactive=nactive,
            ncore=ncore
        )
    elif type.lower() in ("H0","no_df","H0_only"):
        return _aimb_1body_2_afqmc_H0only(fname,gw_iteration=gw_iteration,force_real=force_real)
    elif type.lower() in ("hamil"):
        return _aimb_1body_2_afqmc_hamil(
            fname,
            force_real=force_real,
            nactive=nactive,
            ncore=ncore
        )
    else:
        raise ValueError("Unknown downfolding type")
    

def _aimb_1body_2_afqmc_corr(fname,gw_iteration=1,include_dc=True,force_real=False,nactive=None,ncore=0):
    """
    Generate the 1-body Hamiltonian from am AIMBES checkpoint file
    for embedding on top of GW.

    with, `Heff_base_sIab = H0_sIab`, we will generate,
    
    `Heff_sIab = Heff_base_sIab + (Vhf_gw_sIab - Vhf_dc_sIab) + (Vcorr_gw_sIab - Vcorr_dc_sIab)`
    """
    print(f"Using downfolding terms from GW at iteration {gw_iteration}")
    hdf5_prefix = f"/downfold_1e/iter{gw_iteration}/"

    if ncore > 0:
        raise NotImplementedError


    with h5.File(fname,'r') as f:
        Heff_loc_1body = f[hdf5_prefix + "H0_sIab"][...] # old: "Heff_loc_sIab"
        Heff_loc_1body = Heff_loc_1body[0,0,:,:,0] + 1j*Heff_loc_1body[0,0,:,:,1]


        if hdf5_prefix + "Vhf_gw_sIab" in f.keys():
            V_hartree = f[hdf5_prefix + "Vhf_gw_sIab"][...]
        elif hdf5_prefix + "Vhf_sIab" in f.keys():
            V_hartree = f[hdf5_prefix + "Vhf_sIab"][...]
        else:
            raise RuntimeError(f"{fname} does not contain a HF potential. Check AIMBES calculation.")
        V_hartree = V_hartree[0,0,:,:,0] + 1j*V_hartree[0,0,:,:,1]
        Heff_loc_1body += V_hartree

        if include_dc:
            V_hartree_dc = f[hdf5_prefix + "Vhf_dc_sIab"][...]
            V_hartree_dc = V_hartree_dc[0,0,:,:,0] + 1j*V_hartree_dc[0,0,:,:,1]
            Heff_loc_1body -= V_hartree_dc
        
        if hdf5_prefix + "Vcorr_gw_sIab" in f.keys():
            V_corr = f[hdf5_prefix + "Vcorr_gw_sIab"][...]
        elif hdf5_prefix + "Vcorr_sIab" in f.keys():
            V_corr = f[hdf5_prefix + "Vcorr_sIab"][...]
        else:
            raise RuntimeError(f"{fname} does not contain a correlated potential. Check AIMBES calculation.")

        V_corr = V_corr[0,0,:,:,0] + 1j*V_corr[0,0,:,:,1]
        Heff_loc_1body += V_corr

        if include_dc:
            V_corr_dc = f[hdf5_prefix + "Vcorr_dc_sIab"][...]
            V_corr_dc = V_corr_dc[0,0,:,:,0] + 1j*V_corr_dc[0,0,:,:,1]
            Heff_loc_1body -= V_corr_dc
        
        # symmetrize
        if np.allclose(Heff_loc_1body,Heff_loc_1body.conj().T):
            print("Heff_loc_1body is Hermitian!")
        else:
            print("[Warning] Heff_loc_1body is not Hermitian! Forcing Hermiticity")
            Heff_loc_1body = 0.5*(Heff_loc_1body + Heff_loc_1body.conj().T)

        print("The largest imag. element in Heff_loc: {}".format(np.max(np.abs(Heff_loc_1body.imag))))
        if force_real and np.isclose(np.max(np.abs(Heff_loc_1body.imag)),0.0):
            print(" Imaginary part of the 1-body Heff is 0.0; treating as real!")
            Heff_loc_1body = Heff_loc_1body.real
        elif force_real:
            print(" [Warning] Forcing real!!")
            Heff_loc_1body = Heff_loc_1body.real
        
        return Heff_loc_1body


# TODO: this temporary for testing!
def _aimb_1body_2_afqmc_corr_fc(fname,gw_iteration=1,include_dc=True,force_real=False,nactive=None,ncore=0):
    """
    Generate the 1-body Hamiltonian from am AIMBES checkpoint file
    for embedding on top of GW.

    with, `Heff_base_sIab = H0_sIab`, we will generate,
    
    `Heff_sIab = Heff_base_sIab + (Vhf_gw_sIab - Vhf_dc_sIab) + (Vcorr_gw_sIab - Vcorr_dc_sIab)`
    """
    if ncore > 0:
        raise NotImplementedError


    print(f"Using downfolding terms from GW at iteration {gw_iteration}")
    hdf5_prefix = f"/downfold_1e/iter{gw_iteration}/"
    with h5.File(fname,'r') as f:
        Heff_loc_1body = f[hdf5_prefix + "H0_sIab"][...] # old: "Heff_loc_sIab"
        Heff_loc_1body = Heff_loc_1body[0,0,:,:,0] + 1j*Heff_loc_1body[0,0,:,:,1]

        V_hartree = f[hdf5_prefix + "Vhf_sIab"][...]
        V_hartree = V_hartree[0,0,:,:,0] + 1j*V_hartree[0,0,:,:,1]
        Heff_loc_1body += V_hartree

        if include_dc:
            V_hartree_dc = f[hdf5_prefix + "Vhf_dc_sIab"][...]
            V_hartree_dc = V_hartree_dc[0,0,:,:,0] + 1j*V_hartree_dc[0,0,:,:,1]
            Heff_loc_1body -= V_hartree_dc
        
        # symmetrize
        if np.allclose(Heff_loc_1body,Heff_loc_1body.conj().T):
            print("Heff_loc_1body is Hermitian!")
        else:
            print("[Warning] Heff_loc_1body is not Hermitian! Forcing Hermiticity")
            Heff_loc_1body = 0.5*(Heff_loc_1body + Heff_loc_1body.conj().T)

        print("The largest imag. element in Heff_loc: {}".format(np.max(np.abs(Heff_loc_1body.imag))))
        if force_real and np.isclose(np.max(np.abs(Heff_loc_1body.imag)),0.0):
            print(" Imaginary part of H0 is 0.0; treating as real!")
            Heff_loc_1body = Heff_loc_1body.real
        elif force_real:
            print(" [Warning] Forcing real!!")
            Heff_loc_1body = Heff_loc_1body.real
    
        # TODO: call a FC function here
        return Heff_loc_1body

def _aimb_1body_2_afqmc_H0only(fname,gw_iteration=1,force_real=False):
    """
    Generate the 1-body Hamiltonian from am AIMBES checkpoint file
    for embedding on top of GW.

    with, `Heff_base_sIab = H0_sIab - mu * I`, we will generate,
    
    `Heff_sIab = Heff_base_sIab + (Vhf_gw_sIab - Vhf_dc_sIab) + (Vcorr_gw_sIab - Vcorr_dc_sIab)`
    """
    print(f"Using downfolding terms from GW at iteration {gw_iteration}")
    hdf5_prefix = f"/downfold_1e/iter{gw_iteration}/"
    with h5.File(fname,'r') as f:
        Heff_loc_1body = f[hdf5_prefix + "H0_sIab"][...] # old: "Heff_loc_sIab"
        Heff_loc_1body = Heff_loc_1body[0,0,:,:,0] + 1j*Heff_loc_1body[0,0,:,:,1]

        # symmetrize
        if np.allclose(Heff_loc_1body,Heff_loc_1body.conj().T):
            print("Heff_loc_1body is Hermitian!")
        else:
            print("[Warning] Heff_loc_1body is not Hermitian! Forcing Hermiticity")
            Heff_loc_1body = 0.5*(Heff_loc_1body + Heff_loc_1body.conj().T)

        print("The largest imag. element in Heff_loc: {}".format(np.max(np.abs(Heff_loc_1body.imag))))
        if force_real and np.isclose(np.max(np.abs(Heff_loc_1body.imag)),0.0):
            print(" Imaginary part of H0 is 0.0; treating as real!")
            Heff_loc_1body = Heff_loc_1body.real
        elif force_real:
            print(" [Warning] Forcing real!!")
            Heff_loc_1body = Heff_loc_1body.real

        return Heff_loc_1body

def _aimb_1body_2_afqmc_hamil(fname,force_real=False,nactive=None,ncore=0):
    """
    TODO: read all k-points
    """
    if ncore > 0:
        raise NotImplementedError

    if nactive is not None:
        active = slice(ncore,ncore+nactive)
    else:
        active = slice(ncore)

    with h5.File(fname, 'r') as f:
        # now has shape [nspin,nband,nband,2(real,imag)]
        H1 = f["/System/H1_kp0"][...] 
        if len(H1.shape) == 3:
            raise ValueError(
                "Using an out-of-date Hamiltonian format from AIMBES. "
            )
        # assuming one spin sector for now
        H1 = H1[0,:,:,0] + 1j*H1[0,:,:,1]
        if np.isclose(np.max(np.abs(H1.imag)),0.0):
            print("   Imaginary part of H1 is 0.0; treating as real!")
            H1 = H1.real
        elif force_real:
            print("  [Warning] forcing H1 to real by discarding imaginary part!!")
            H1 = H1.real
    return H1

def _aimb_2body_2_afqmc_hamil(fname,force_real=False,nactive=None,ncore=0):
    """
    TODO: read all k-points
    """
    if ncore > 0:
        raise NotImplementedError

    if nactive is not None:
        active = slice(ncore,ncore+nactive)
    else:
        active = slice(ncore)
    
    with h5.File(fname, 'r') as f:
        Vq = f["/Interaction/Vq0"][...]
        Vq = Vq[:,0,0,:,:,0] + 1j*Vq[:,0,0,:,:,1]
        if np.isclose(np.max(np.abs(Vq.imag)),0.0):
            print("   Imaginary part of Vq is 0.0; treating as real!")
            Vq = Vq.real
        elif force_real:
            print("  [Warning] forcing Vq to real by discarding imaginary part!!")
            Vq = Vq.real

        # some consistency checks:
        nbnd = f["/Interaction/nbnd"][...]
        assert Vq.shape[1] == Vq.shape[2] == nbnd

        # this is what AFQMC currently reads for num bands!
        system = f["System"]
        nbnd_system = system.attrs["number_of_bands"]
        assert nbnd == nbnd_system

    return Vq


@numba.jit(nopython=True,cache=True)
def _calc_rho_local(C_skai,rho_0):
    """Compute and return the k-point averaged local denisty."""
    #pdb.set_trace()
    ns = C_skai.shape[0]
    nk = C_skai.shape[1]
    nlocal = C_skai.shape[2]
    rho_local = np.zeros((ns,nlocal,nlocal),dtype=np.complex128)
    for k in range(nk):
        for s in range(ns):
            rho_local[s,:,:] = C_skai[s,k,:,:] @ rho_0[s,k,:,:] @ C_skai[s,k,:,:].conj().T
    return rho_local / nk

@numba.jit(nopython=True,cache=True)
def _calc_closed_fc_term(rho_0,V):
    """
    """
    n_local_orbs = rho_0.shape[0]
    Heff = np.zeros((n_local_orbs,n_local_orbs),dtype=np.complex128)
    for alpha in range(n_local_orbs):
        for beta in range(n_local_orbs):
            for gamma in range(n_local_orbs):
                for delta in range(n_local_orbs):
                    Heff[alpha,beta] = rho_0[gamma,delta] * (2*V[alpha,beta,delta,gamma] - V[alpha,gamma,delta,beta])
    return Heff

def _aimb_1body_2_afqmc_fc(fname,gw_iteration=1,use_chol=False,input_band_range=None,force_real=False):
    r"""
    Generate the 1-body frozen core Hamiltonian from am AIMBES checkpoint file.

    Specifically, compute and return:

    $\sum_{\gamma\delta}(\rho_0)_{\gamma\delta}\left[ 2V^{bare}_{\alpha\beta\delta\gamma} - V^{bare}_{\alpha\gamma\delta\beta} \right]$

    Inputs:
    - fname:str = name of HDF5 file containing data from AIMBES
    - gw
    """
    V_bacd = get_aimbes_2body(fname=fname,get_screened=False,freq=0,symmetrize=True)

    with h5.File(fname,'r') as f:
        hdf5_prefix = f"/downfold_1e/iter{gw_iteration}/"
        Heff_loc_1body = f[hdf5_prefix + "H0_sIab"][...]
        Heff_loc_1body = Heff_loc_1body[0,0,:,:,0] + 1j*Heff_loc_1body[0,0,:,:,1]

        # Add Hartree potential (computed wrt the input DFT density)
        V_hartree = f[hdf5_prefix + "Vhf_gw_sIab"][...]
        V_hartree = V_hartree[0,0,:,:,0] + 1j*V_hartree[0,0,:,:,1]
        Heff_loc_1body += V_hartree

        C_skIai = f["/downfold_1e/C_skIai"][...]
        rho_0 = f["/scf/iter0/Dm_skij"][...]

    if input_band_range is not None:
        input_band_range = tuple((int(b) for b in input_band_range.split(":")))
        band_slice = slice(*input_band_range)
        rho_0 = rho_0[:,:,band_slice,band_slice,0] + 1j*rho_0[:,:,band_slice,band_slice,1]
    else:
        rho_0 = rho_0[:,:,:,:,0] + 1j*rho_0[:,:,:,:,1]

    C_skai = C_skIai[:,:,0,:,:,0] + 1j*C_skIai[:,:,0,:,:,1] # doesn't necessarily have all
                                                            #   bands from DFT
    
    rho_0_local = _calc_rho_local(C_skai,rho_0.copy())    

    # get double count correction: "closed" / "spin-unpolarized" case 
    if rho_0_local.shape[0] == 1:
        Heff_loc_1body -= _calc_closed_fc_term(rho_0_local[0],V_bacd)
    else:
        raise RuntimeError(
            "invalid desnity shape for calculating frozen core DC: "
            "must have CLOSED / spin-unpolarized form."
        )

    print("The largest imag. element in Heff_loc: {}".format(np.max(np.abs(Heff_loc_1body.imag))))
    if force_real and np.isclose(np.max(np.abs(Heff_loc_1body.imag)),0.0):
        print(" Imaginary part of the 1-body Heff is 0.0; treating as real!")
        # symmetrize
        Heff_loc_1body = 0.5*(Heff_loc_1body + Heff_loc_1body.conj().T)
        Heff_loc_1body = Heff_loc_1body.real

    return Heff_loc_1body


def _aimb_1body_2_afqmc_dft(fname,gw_iteration=1,include_dc=True,force_real=False):
    """
    Generate the 1-body Hamiltonian from am AIMBES checkpoint file
    for embedding on top of DFT.

    with, `Heff_base_sIab = H0_sIab - mu * I`, we will generate,
    
    `Heff_sIab = Heff_base_sIab + (Vhf_gw_sIab - Vhf_dc_sIab)`
    """
    print("Using downfolding terms from DFT")
    #TODO: fix the names!!
    hdf5_prefix = f"/downfold_1e/iter{gw_iteration}/"
    with h5.File(fname,'r') as f:
        Heff_loc_1body = f[hdf5_prefix + "Heff_loc_sIab"][...]
        Heff_loc_1body = Heff_loc_1body[0,0,:,:,0] + 1j*Heff_loc_1body[0,0,:,:,1]

        # need chemical potential term!
        mu = f[hdf5_prefix + "mu"][...]
        Heff_loc_1body -= mu*np.eye(*Heff_loc_1body.shape)

        if include_dc:
            V_hartree_dc = f[hdf5_prefix + "hf_dc_sIab"][...]
            V_hartree_dc = V_hartree_dc[0,0,:,:,0] + 1j*V_hartree_dc[0,0,:,:,1]
            Heff_loc_1body -= V_hartree_dc

        print("The largest imag. element in Heff_loc: {}".format(np.max(np.abs(Heff_loc_1body.imag))))
        if force_real and np.isclose(np.max(np.abs(Heff_loc_1body.imag)),0.0):
            print(" Imaginary part of the 1-body Heff is 0.0; treating as real!")
            # symmetrize
            Heff_loc_1body = 0.5*(Heff_loc_1body + Heff_loc_1body.conj().T)
            Heff_loc_1body = Heff_loc_1body.real

        return Heff_loc_1body


def get_aimbes_2body(fname,get_screened=True,freq=0,symmetrize=False,reshape=False,force_real=False,nactive=None,ncore=0,sc_iter=1):
    """
    Get 2-body interaction term from the AIMBES HDF5 output.
    """
    with h5.File(fname,'r') as f:
        if get_screened:
            print("Using Vloc + Uloc")
            V_abcd = (f[f"/downfold_2e/iter{sc_iter}/Vloc_abcd"][...] 
                      + f[f"/downfold_2e/iter{sc_iter}/Uloc_wabcd"][...][freq])
        else:
            print("Using Vloc")
            V_abcd = f[f"/downfold_2e/iter{sc_iter}/Vloc_abcd"][...]
        
        if nactive is not None:
            M = nactive
        else:
            M = V_abcd.shape[0] - ncore
        active = slice(ncore,ncore+M)

        V_abcd = V_abcd[active,active,active,active,0] + \
                 + 1j*V_abcd[active,active,active,active,1]
        

        # Symmetrize - Q: should I do this on V_abcd or V_bacd??
        if symmetrize:
            V_abcd = (V_abcd + V_abcd.transpose(1,0,2,3)) * 0.5
            V_abcd = (V_abcd + V_abcd.transpose(0,1,3,2)) * 0.5
            V_abcd = (V_abcd + V_abcd.transpose(2,3,0,1)) * 0.5

        if force_real and np.isclose(np.max(np.abs(V_abcd.imag)),0.0):
            print(" Imaginary part of the ERIs is 0.0; treating as real! ")
            V_abcd = V_abcd.real
        
        print("The largest imag. element in V_abcd: {}".format(np.max(np.abs(V_abcd.imag))))

        if reshape:
            if np.iscomplexobj(V_abcd):
                # Need the Hermitian form for Cholesky
                V_abcd = np.transpose(V_abcd,(1,0,2,3)) # TODO: swap conventions! (0,1,3,2)!
            return V_abcd.reshape((M*M,M*M))
        else:
            return V_abcd



def aimb_2body_2_afqmc(fname,use_crpa=True,freq=0,tol=1.0E-6,use_chol=True,symmetrize=True,verbose=False,force_real=False):

    V_bacd = get_aimbes_2body(fname=fname,get_screened=use_crpa,freq=freq,symmetrize=symmetrize,reshape=True,force_real=force_real)


    if use_chol:
        print(" === Performing Cholesky decomp. on V_loc + u_loc === ")
        Lvecs = cholesky(V_bacd,tol=tol,verbose=True).T
        
        if verbose:
            print("\n === Lvecs === ")
            for gamma in range(Lvecs.shape[-1]):
                print(gamma, Lvecs[:,gamma])
        return Lvecs
    else:
        print(" === Diagonalizing V_mu,nu === ")
        eigenvals,Lvecs = np.linalg.eigh(V_bacd)
        
        hs_ops_inds = eigenvals > tol

        if verbose:
            print("non-zero eigenvalues:")
            for e in eigenvals[hs_ops_inds]:
                print(e)
        
        print("Are imaginary components of eigenvalues close to zero??")
        print(np.isclose(eigenvals.imag,np.zeros_like(eigenvals.imag)))
        real_eigenvals = eigenvals.real[np.isclose(eigenvals.imag,np.zeros_like(eigenvals.imag))]

        if verbose:
            print(" === Lvecs === ")
            for gamma in range(Lvecs.shape[-1]):
                print(gamma, Lvecs[:,gamma])

        Lscaled = (np.sqrt(real_eigenvals[hs_ops_inds])*Lvecs[:,hs_ops_inds])

        if verbose:
            print(" === scaled Lvecs === ")
            for gamma in range(Lscaled.shape[-1]):
                print(gamma, Lscaled[:,gamma])

        return Lscaled



def _is_bare_hamil(aimbes_fname,nkpts=0) -> bool:
    """
    TODO: consisder checking all k-point as well; may not be necessary
          if we're just identifying the H type.
    """
    terms_to_check = (
        "/System/H1_kp0",     # non-interacting Hamiltonian for first k-point
        "/Interaction/Vq0"    # bare Coulomb interaction for first k-point as Cholesky
    )
    with h5.File(aimbes_fname,'r') as f:
        terms = [ term in f for term in terms_to_check ]
        return all(terms)


def _is_froze_core_hamil(aimbes_fname) -> bool:

    terms_to_check = (
        "/downfold_1e/iter1/H0_sIab",      # non-interacting Hamiltonian
        "/downfold_2e/iter1/Vloc_abcd",    # the local Coulomb interaction
        "/downfold_1e/iter1/Vhf_sIab",     # the 'Hartree-Fock' potential
        "/downfold_1e/iter1/Vhf_dc_sIab"   # the double-counting correction to the HF potential
        )
    with h5.File(aimbes_fname,'r') as f:
        terms = [ term in f for term in terms_to_check ]
        return all(terms)
    
def _is_crpa_hamil(aimbes_fname,sc_iter=0) -> bool:
        logger.dev("[WARNING] [for developers] _is_crpa_hamil may fail to detec cRPA H depending on the order "
             "in which downfold_1e and downfold_2e are invoked")
        terms_to_check = (
        f"/downfold_1e/iter{sc_iter}/H0_sIab",         # non-interacting Hamiltonian
        f"/downfold_2e/iter{sc_iter+1}/Vloc_abcd",     # the local Coulomb interaction
        f"/downfold_1e/iter{sc_iter}/Vhf_gw_sIab",        # the 'Hartree-Fock' potential
        f"/downfold_1e/iter{sc_iter}/Vhf_dc_sIab",      # the double-counting correction to the HF potential
        f"/downfold_1e/iter{sc_iter}/Vcorr_gw_sIab",   # the GW correlated potential
        f"/downfold_1e/iter{sc_iter}/Vcorr_dc_sIab",   # the double-counting correction to the GW correlated potential
        f"/downfold_2e/iter{sc_iter+1}/Uloc_wabcd"     # the frequency-dependent partially-screen Coulomb interaction
        )
        with h5.File(aimbes_fname,'r') as f:
            terms = [ term in f for term in terms_to_check ]
            return all(terms)


def _infer_aimbes_hamil_type(aimbes_fname,hamil_type=None,sc_iter=1) -> AIMBESHamiltonian:
    """
    Attempts to infer the Hamiltonian type of the Hamiltonian
    saved in the AIMBES HDF5 called 'aimbes_fname' and raises 
    and error if no type can be inferred.

    When the contents of 'aimbes_fname' are consistent with 
    mutiple Hamiltonian types, the most general hamiltonian
    possible will be infered (unless 'hamil_type' is given.)

    If 'hamil_type` is specified, then 'aimbes_fname' is checked
    for consistency with the 'hamil_type' and an error is raised
    it is not consistent.
    """
    if hamil_type is not None:
        hamil_type = get_aimbes_hamil_enum(hamil_type)

        # NOTE: order is not relevant here since hamil_type is set
        if hamil_type == AIMBESHamiltonian.BARE and _is_bare_hamil(aimbes_fname):
            return hamil_type
        elif hamil_type == AIMBESHamiltonian.FROZEN_CORE and _is_froze_core_hamil(aimbes_fname):
            return hamil_type
        elif hamil_type == AIMBESHamiltonian.CRPA and _is_crpa_hamil(aimbes_fname,sc_iter=sc_iter):
            return hamil_type
        else:
            raise ValueError(f"aimbes file '{aimbes_fname}' is not consistent with Hamiltonian type {hamil_type}")
    else:
        # NOTE: order is important here since an aimbes hdf5 file may be 
        #       consistent with multiple Hamiltonian type
        if _is_crpa_hamil(aimbes_fname,sc_iter=sc_iter):
            return AIMBESHamiltonian.CRPA
        elif _is_froze_core_hamil(aimbes_fname):
            return AIMBESHamiltonian.FROZEN_CORE
        elif _is_bare_hamil(aimbes_fname):
            return AIMBESHamiltonian.BARE
        else:
            raise ValueError(f"Could not infer a valid Hamiltonian type from {aimbes_fname}")


def read_bare_hamiltonian(fname):
    """
    Read Hamiltonian from the output of AIMBES's 'hamil' directive.
    """
    print("\n ====== Reading Bare Hamiltonian from AIMBES ====== ")
    print("\n\n ==== Reading 1-body Hamiltonian ==== ")
    H1 = aimb_1body_2_afqmc(fname,type="hamil")
    print("\n\n ==== Reading 2-body Hamiltonian ==== ")
    H2 = _aimb_2body_2_afqmc_hamil(fname)
    with h5.File(fname,'r') as f:
        delta = f["/Interaction/tol"][...]
    return H1,H2,delta


def read_frozen_core_hamiltonian(fname,chol_tol=1.0e-6):
    """
    Read Hamiltonian from the output of AIMBES's 'hamil' directive.
    """
    print("\n ====== Reading Frozen Core/MF Downfold Hamiltonian from AIMBES ====== ")
    print("\n\n ==== Reading 1-body Hamiltonian ==== ")
    H1 = aimb_1body_2_afqmc(fname,type="gw_fc")
    nmo = H1.shape[0]
    print("\n\n ==== Reading 2-body Hamiltonian ==== ")
    Vabcd = get_aimbes_2body(
        fname,
        get_screened=False
    )
    H2 = cholesky(
        np.transpose(Vabcd,(0,1,3,2)).reshape(nmo**2, nmo**2),
        tol=chol_tol,
        verbose=True
    ).T
    return H1,H2.reshape((nmo,nmo,-1)).T,chol_tol

def read_crpa_hamiltonian(fname,chol_tol=1.0e-6,freq=0,sc_iter=0):
    print("\n ====== Reading GW+cRPA Hamiltonian from AIMBES ====== ")
    print("\n\n ==== Reading 1-body Hamiltonian ==== ")
    H1 = aimb_1body_2_afqmc(fname,type="gw",gw_iteration=sc_iter)
    nmo = H1.shape[0]
    print("\n\n ==== Reading 2-body Hamiltonian ==== ")
    Vabcd = get_aimbes_2body(
        fname,
        get_screened=True,
        freq=freq,
        sc_iter=sc_iter+1 # this assumes GW then downfold_2e!!
    )
    H2 = cholesky(
        np.transpose(Vabcd,(0,1,3,2)).reshape(nmo**2, nmo**2),
        tol=chol_tol,
        verbose=True
    ).T
    return H1,H2.reshape((nmo,nmo,-1)).T,chol_tol



def read_aimbes_Hamiltonian(fname:str,hamil_type=None,omega=0,chol_tol=1.0e-6,sc_iter=1):
    r"""
    Read aimbes Hamiltonian from an aimbes HDF5 file.

    The most general Hamiltonian that can be read is:

    .. math::
        \hat{H} = \hat{H}_0 + (\hat{V}_{HF} - \hat{V}_{HF-DC}) + 
                    + (\hat{V}_{Corr} - \hat{V}_{Corr-DC}) 
                    + \hat{V}_{bare} + \hat{U}(omega)
        
    where,

    - :math:`\hat{H}_0` is the non-interacting Hamiltonian
    - :math:`\hat{V}_{HF}` is the Hartree-Fock potential (i.e. 2*J - K ) evaluated over the full hilbert space
    - :math:`\hat{V}_{HF-DC}` is the doulbe counting correction for :math:`\hat{V}_{HF}` (2*J -K evaluated in the active space. For calculations in the full Hilbert space, this is identical to :math:`\hat{V}_{HF}`)

    The "types" of Hamiltonian that can be read are:

    1. the bare Hamiltonian - consists of :math:`\hat{H}_0 + \hat{V}_{bare}`

    Parameters
    ----------
    fname : str
        The name of an aimbes HDF5 file to read
    hamil_type : str, optional
        The type of Hamiltonian to read from the HDF5 file. 
        If not specified, the hamil_type will be inferred based on
        the contents of the 'fname'.
    omega : int, optional
        The index of the frequency to use for the paritally screened
        Coulomb interaction
    chol_tol : float, optional
        Cholesky tolerance
    sc_iter : int, optional
        Self-consistent iteration number
    """
    hamil_type = _infer_aimbes_hamil_type(fname,hamil_type,sc_iter=sc_iter)


    if hamil_type == AIMBESHamiltonian.BARE:
        return read_bare_hamiltonian(fname=fname)
    elif hamil_type == AIMBESHamiltonian.FROZEN_CORE:
        return read_frozen_core_hamiltonian(fname=fname,chol_tol=chol_tol)
    elif hamil_type == AIMBESHamiltonian.CRPA:
        return read_crpa_hamiltonian(fname=fname,sc_iter=sc_iter)
    else: 
        raise ValueError(f"[for developers] tried to read invalid hamil_type {hamil_type}")

