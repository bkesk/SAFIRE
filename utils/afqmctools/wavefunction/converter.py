# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import ast
import h5py
import numpy
import scipy.sparse

import re

from afqmctools.utils.io import from_complex
from afqmctools.utils.types import get_spin_symm_enum

def convert_string(s):
    try:
        c = complex(s)
    except ValueError:
        c = ast.literal_eval(s)
        c = c[0] + 1j*c[1]
    return c


def read_wavefunction(filename):
    with h5py.File(filename,'r') as fh5:
        if 'Wavefunction/NOMSD' in fh5:
            wgroup = fh5['Wavefunction/NOMSD']
            return read_nomsd_hdf5(wgroup)
        elif 'Wavefunction/PHMSD' in fh5:
            wgroup = fh5['Wavefunction/PHMSD']
            return read_phmsd_hdf5(wgroup)
        else:
            raise ValueError("Wavefunction not found")


def read_nomsd_hdf5(wgroup):
    dims = wgroup['dims']
    nmo = dims[0]
    na = dims[1]
    nb = dims[2]
    walker_type = dims[3]
    if walker_type == 2:
        uhf = True
    else:
        uhf = False

    if walker_type == 3:
        npol = 2
    else:
        npol =1
    
    nci = dims[4]
    coeffs = from_complex(wgroup['ci_coeffs'][:], (nci,))
    psi0a = from_complex(wgroup['Psi0_alpha'][:], (npol*nmo,na))
    
    if uhf:
        psi0b = from_complex(wgroup['Psi0_beta'][:], (npol*nmo,nb))
    psi0 = numpy.zeros((npol*nmo,na+nb),dtype=numpy.complex128)
    psi0[:,:na] = psi0a.copy()
    if uhf:
        psi0[:,na:] = psi0b.copy()
    else:
        psi0[:,na:] = psi0a[:,:nb].copy()
    wfn = numpy.zeros((nci,npol*nmo,na+nb), dtype=numpy.complex128)
    for idet in range(nci):
        ix = 2*idet if uhf else idet
        pa = orbs_from_dset(wgroup['PsiT_{:d}/'.format(ix)])
        wfn[idet,:,:na] = pa
        if uhf:
            ix = 2*idet + 1
            wfn[idet,:,na:] = orbs_from_dset(wgroup['PsiT_{:d}/'.format(ix)])
        else:
            wfn[idet,:,na:] = pa[:,:nb]
    return (coeffs,wfn), psi0, (na, nb), get_spin_symm_enum(walker_type)

def read_phmsd_hdf5(wgroup):
    """Read particle-hole multi-Slater determinant wavefunction from HDF5

    Parameters
    ----------
    wgroup : h5py.Group
        the group containing wavefunction data

    Returns
    -------
    wfn : tuple(coeffs:numpy.ndarray, occa:numpy.ndarray, occb:numpy.ndarray)
        where 'coeffs' are the CI coefficients of the wavefunction, occa are 
        the spin-up occupancies, and occb are the spin-down occupancies.
    psi0 : numpy.ndarray
        a numpy array containing the initial Slater determinant. The first na
        columns are spin-up orbitals, and the last nb columns are spin-down
        orbitals.
    (na:int,nb:int)
        a touple containing the number of up (na) and down (nb) electrons.

    """
    dims = wgroup['dims']
    nmo = dims[0]
    na = dims[1]
    nb = dims[2]
    walker_type = dims[3]
    if walker_type == 2:
        uhf = True
    else:
        uhf = False
    nci = dims[4]
    coeffs = from_complex(wgroup['ci_coeffs'][:], (nci,))
    occs = wgroup['occs'][:].reshape((nci,na+nb))
    occa = occs[:,:na]
    occb = occs[:,na:]-nmo
    wfn = (coeffs, occa, occb)
    psi0a = from_complex(wgroup['Psi0_alpha'][:], (nmo,na))
    if uhf:
        psi0b = from_complex(wgroup['Psi0_beta'][:], (nmo,nb))
    psi0 = numpy.zeros((nmo,na+nb),dtype=numpy.complex128)
    psi0[:,:na] = psi0a.copy()
    if uhf:
        psi0[:,na:] = psi0b.copy()
    else:
        psi0[:,na:] = psi0a.copy()
    return wfn, psi0, (na,nb), get_spin_symm_enum(walker_type)

def orbs_from_dset(dset):
    """Will read actually A^{H} but return A.
    """
    dims = dset['dims'][:]
    wfn_shape = (dims[0],dims[1])
    nnz = dims[2]
    data = from_complex(dset['data_'][:],(nnz,))
    indices = dset['jdata_'][:]
    pbb = dset['pointers_begin_'][:]
    pbe = dset['pointers_end_'][:]
    indptr = numpy.zeros(dims[0]+1)
    indptr[:-1] = pbb
    indptr[-1] = pbe[-1]
    wfn = scipy.sparse.csr_array((data,indices,indptr),shape=wfn_shape)
    return wfn.toarray().conj().T.copy()

def read_dice_h5_wavefunction(input_file, ndets, state):
    nup = 0
    ndn = 0
    nmo = 0
    uhf = True   # enable GHF later on...

    with h5py.File(input_file) as fh5:

        assert( state < fh5["/nroots"][0] )
        assert( ndets <= fh5["/ndets"][0] )
        nmo = fh5["/norbs"][0] 
        
        ci_coeff = fh5["/coeff_r"+str(state)][:]  
        ci_confg = fh5["/confg_r"+str(state)][:,:]  
        assert( ci_confg.shape[0] == fh5["/ndets"][0] )
        assert( ci_confg.shape[1] == nmo )

    # get nup/ndn
    for i,a in enumerate(ci_confg[0,:]):
        if chr(a) == '2':
            nup+=1
            ndn+=1
        elif chr(a) == 'a':
            nup+=1
        elif chr(a) == 'b':
            ndn+=1
        else:
            if not (chr(a) == '0'):
                print("Unknown configuration character: ",chr(a))
                assert(0)

    # fine array of sorted coefficients    
    sorted_index = numpy.argsort(numpy.abs(ci_coeff))

    coeffs = numpy.zeros(ndets,dtype=complex) 
    occa = numpy.zeros((ndets,nup),dtype=int) 
    occb = numpy.zeros((ndets,ndn),dtype=int) 

    # MAM: Dice already prints in alpha-beta parity 
    for p in range(ndets):
        n = sorted_index[-p-1] 
        na=0
        nb=0
        for i,a in enumerate(ci_confg[n,:]):
            if chr(a) == '2':
                occa[p,na] = i
                na+=1
                occb[p,nb] = i
                nb+=1
            elif chr(a) == 'a':
                occa[p,na] = i
                na+=1
            elif chr(a) == 'b':
                occb[p,nb] = i
                nb+=1
            else:
                if not (chr(a) == '0'):
                    print("Unknown configuration character: ",chr(a))
                    assert(0)
        coeffs[p] = convert_string(ci_coeff[n])

    return (coeffs, occa, occb), nmo, nup, ndn, uhf

def read_with_check(f):
    line = f.readline()    
    if line=='':
      print("Problems parsing file: Reached EOF.") 
      assert(0)
    return line

def _real_coeffs(line_data) -> bool :
    """Checks if *split* Dice ASCII output line has real-valued coefficients
    
    Parameters
    ----------
    line_data : list[str]
        contains a .split() line corresponding to a Slater det. entry.

    Notes
    -----
    Dice uses the following format for real-valued coeffs:
    
    ```txt
    0    -0.5032009288     2 2 2 2 0   0 0 0 0 0   0 
    1     0.0883320042     2 2 2 0 0   0 2 0 0 0   0 
    2    -0.0928684363     2 2 0 2 2   0 0 0 0 0   0 
    3     0.0804092922     2 2 0 2 a   b 0 0 0 0   0 
    ```

    and for complex-valued coeffs:

    ```txt
    0    -0.5032009288     0.7487247473 2 2 2 2 0   0 0 0 0 0   0 
    1     0.0883320042     0.0439272635 2 2 2 0 0   0 2 0 0 0   0 
    2    -0.0928684363     0.0232217507 2 2 0 2 2   0 0 0 0 0   0 
    3     0.0804092922    -0.0218992778 2 2 0 2 a   b 0 0 0 0   0 
    ```

    differentiating only requires us to check `line_data[2]`

    """
        
    if re.match(r'^[0,a,b,2]$', line_data[2]):
        return True
    elif re.match(r'^[-]?[0-9]+\.[0-9]+$',line_data[2]):
        return False
    else:
        raise ValueError("[For Developers] invalid Dice Slater determinant format")
    

def read_dice_ascii_wavefunction(input_file:str, ndets:int, state:int):
    """Reads SHCI wavefunction from the text-based output of Dice

    Parameters
    ----------
    input_file : str
        the name of a file containing Dice's text-based output
    ndets : int
        the number of Slater determinants to read
    state : int
        the index of the state to read

    Returns
    -------
    tuple
        has length 5 and contains (wfn, nmo:int, nup:int, ndn:int, 
        walker_type:int) where `nmo` is the number of orbitals, 
        `nup` is the number of up electrons, `ndn` is the number of 
        down electrons, and `walker_type` is an int encoding the 
        type of walker (0 : CLOSED, 1: COLLINEAR, 2: NONCOLLINEAR
        4: FULLYSPINPOLARIZED). `wfn` is a touple containing:
        (coeffs:numpy.array, occa:numpy.array, occb:numpy.array)
        where 'coeffs' are the CI coefficients of the wavefunction,
        occa are the spin-up occupancies, and occb are the 
        spin-down occupancies. All three arrays are the same lenght.
        
    Raises
    ------
    ValueError
        If ndets is not specified
    """

    if ndets is None:
        raise ValueError("read_dice_acsii_wavefunctions requires an integer value for 'ndets'")

    nup = 0
    ndn = 0
    nmo = 0

    with open(input_file) as f:
        line = read_with_check(f) 
        # find "Printing most important determinants"
        while line.find('Printing most important determinants') < 0:
          line = read_with_check(f) 
        # now look for the correct state
        data = read_with_check(f).split() 
        while (len(data)!=3) or (data[0].find("State")<0) or (int(data[2]) != state): 
            data = read_with_check(f).split() 
        # read first configuration and get nmo, nup, ndn
        data = read_with_check(f).split() 
        assert(len(data)>2)
        nmo = len(data)-2
        
        real_coeffs =  _real_coeffs(data)
        if real_coeffs:
            start_occ = 2
        else:
            start_occ = 3

        for i,a in enumerate(data[start_occ:]):
            if (a=='2') or (a=='a'):
                nup+=1
            if (a=='2') or (a=='b'):
                ndn+=1
        print("Number of electrons: up:{}, down:{}".format(nup,ndn))

        coeffs = numpy.zeros(ndets,dtype=complex) 
        occa = numpy.zeros((ndets,nup),dtype=int) 
        occb = numpy.zeros((ndets,ndn),dtype=int) 

        idet=0
        while len(data)==nmo+2:
            if real_coeffs:
                coeffs[idet] = convert_string(data[1])
            else:
                coeffs[idet] = convert_string(data[1]) + 1j*convert_string(data[2])
            na=0
            nb=0
            for i,a in enumerate(data[start_occ:]):
                if (a=='2') or (a=='a'):
                    occa[idet,na] = i
                    na+=1
                if (a=='2') or (a=='b'):
                    occb[idet,nb] = i
                    nb+=1
            assert(na==nup)
            assert(nb==ndn)
            idet+=1
            if idet == ndets:
                break
            data = read_with_check(f).split() 
    coeffs = coeffs[:idet]
    occa=occa[:idet,:]
    occb=occb[:idet,:]
    wfn = (numpy.array(coeffs), numpy.array(occa), numpy.array(occb))

    if ndn == 0:
        walker_type = 4
    else:
        walker_type = 2

    return wfn, nmo, nup, ndn, walker_type
