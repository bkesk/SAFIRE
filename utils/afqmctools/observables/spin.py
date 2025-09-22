# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Defines shared mathematical operations for calculating
      angular momentum

Routines
--------
"""
import numpy as np

from afqmctools.hamiltonian.model.ham_class import get_spin_symm_enum,SpinSymm

from warnings import warn

def _process_rdm(rdm,spin_symm:SpinSymm):
    r"""Extract rmd spin sectors for internal use

    Always rerturns rho_uu, rho_dd, rho_ud, rho_du, even if some are zero.

    .. math::
        \rho_{\alpha\beta} = \sum_{\sigma\sigma'} \rho_{\sigma\sigma',\alpha\beta}


    Parameters
    ----------
    rdm:np.array|list
        the 1-body reduced density matrix (rdm). See notes for correct input shape.
    spin_symm:SpinSymm:str
        the spin symmetry of the rdm. if spin_symm is a SpinSymm instance, it is used as is.
        if spin_symm is a string, it is converted to a SpinSymm instance. Accepted string
        values are {'rhf','uhf','ghf','closed','collinear','noncollinear'}. String values
        are not case-sensitive.
    
    Returns
    -------
    tuple(np.array,np.array,np.array,np.array)
        the rdm slice corresponding to each spin sector: rho_uu, rho_dd, rho_ud, rho_du

    Raises
    ------
    ValueError
        If the length of the rdm list is not 1 or 2.

    Notes
    -----
    Input rdm(s) are either a 2D numpy array or a list of 2D numpy arrays. If a list is provided,
        it must have length of 1 for closed-shell or noncollinear rmds, or 2 for collinear systems.
        For Collinear systems, the first rdm in the list is the spin-up rdm and the second is the spin-down rdm.
    The correct input shape for the rdm depends on the spin symmetry of the system:
        #. rhf / closed: an array of shape (M,M) where M is the number of spatial orbitals
        #. uhf / collinear: a list of 2 arrays of shape (M,M) or an array of shape (2M,M) where M is the number of spatial orbitals
        #. ghf / noncollinear: (2M,2M) where M is the number of spatial orbitals
    """
    if spin_symm is not None:
        spin_symm = get_spin_symm_enum(spin_symm)

    # unpack rdm
    if isinstance(rdm,list) or isinstance(rdm,tuple):
        if len(rdm) not in (1,2):
            raise ValueError("rdm list must have length of 1 for closed-shell or noncollinear systems, or 2 for collinear systems")
        rdm = np.vstack(rdm) if len(rdm) == 2 else rdm[0]

    if spin_symm == SpinSymm.CLOSED:
        M = rdm.shape[-1]
        if rdm.shape[0] != M:
            raise ValueError("rdm shape is not square for rhf/closed-shell rdm")
        rho_uu = rdm
        rho_dd = rho_uu
        rho_ud = np.zeros_like(rho_uu)
        rho_du = rho_ud
    elif spin_symm == SpinSymm.COLLINEAR:
        M = rdm.shape[-1]

        if rdm.shape[0] != 2*M:
            raise ValueError("rdm shape is not 2M x M for uhf/collinear rdm")
        rho_uu = rdm[:M]
        rho_dd = rdm[M:]
        rho_ud = np.zeros_like(rho_uu)
        rho_du = rho_ud
    elif spin_symm == SpinSymm.NONCOLLINEAR:
        M = rdm.shape[-1] // 2
        up = slice(0,M)
        down = slice(M,2*M)
        rho_uu = rdm[up,up]
        rho_dd = rdm[down,down]
        rho_ud = rdm[up,down]
        rho_du = rdm[down,up] 

    return rho_uu, rho_dd, rho_ud, rho_du

def _trMM(A,B):
  """
  Fast version of np.trace(A@B)
  """
  # could replace this with einsum("ij,ji",A,B)
  # but einsum overhead may not be worth it
  return (A*B.T).sum()

def spin_squared(rdm,spin_symm:SpinSymm=SpinSymm.COLLINEAR,verbose:bool=False):
    r"""
    Compute the spin squared operator as:

    :math:`<S^2> = <S_z^2 + \frac{1}{2}(S_+ S_- + S_- S_+)>`

    Parameters
    ----------
    rdm:np.array|list
        the 1-body reduced density matrix (rdm). See notes for correct input shape.
    spin_symm:SpinSymm:str
        the spin symmetry of the rdm. if spin_symm is a SpinSymm instance, it is used as is.
        if spin_symm is a string, it is converted to a SpinSymm instance. Accepted string
        values are {'rhf','uhf','ghf','closed','collinear','noncollinear'}. String values
        are not case-sensitive.
    verbose:bool
        print S^2 contributions if True; print nothing if False
    
    Returns
    -------
    spin_squared:float
        the expectation value of the spin squared operator

    Notes
    -----
    Input rdm(s) are either a 2D numpy array or a list of 2D numpy arrays. If a list is provided,
        it must have length of 1 for closed-shell or noncollinear rmds, or 2 for collinear systems.
        For Collinear systems, the first rdm in the list is the spin-up rdm and the second is the spin-down rdm.
    The correct input shape for the rdm depends on the spin symmetry of the system:
        #. rhf / closed: an array of shape (M,M) where M is the number of spatial orbitals
        #. uhf / collinear: a list of 2 arrays of shape (M,M) or an array of shape (2M,M) where M is the number of spatial orbitals
        #. ghf / noncollinear: (2M,2M) where M is the number of spatial orbitals
    """
    rho_uu,rho_dd,rho_ud,rho_du = _process_rdm(rdm,spin_symm=spin_symm)
    nup,ndn = rho_uu.trace(), rho_dd.trace()

    #Sz2_up = 0.25*( tr_uu**2 + tr_uu - np.trace(rho_uu @ rho_uu))
    #Sz2_down = 0.25*( tr_dd**2 + tr_dd - np.trace(rho_dd @ rho_dd))
    #Sz2_mixed = 0.25*( -2*tr_uu*tr_dd )
    #Sz2_noco = 0.25*(2*np.trace(rho_ud@rho_du))

    #Sz2 = Sz2_up + Sz2_down + Sz2_mixed + Sz2_noco

    ## 0.5*(S_plusS_minus + S_minusS_plus)
    #S2_spin_flip = 0.5*(tr_uu + tr_dd - 2*np.trace(rho_uu @ rho_dd))
    #S2_spin_flip += 0.5*(2*np.trace(rho_ud @ rho_du))

    #S2 = Sz2 + S2_spin_flip
    #
    #if verbose:
    #    print(f"<S^2> = {S2=}")
    #    print(f"    [+] <S_z^2> = {Sz2}")
    #    print(f"        [+] <S_z^2> up-only contributions {Sz2_up=}")
    #    print(f"        [+] <S_z^2> down-only contributions {Sz2_down=}")
    #    print(f"        [+] <S_z^2> up-down contributions {Sz2_mixed=}")
    #    print(f"        [+] <S_z^2> non-collinear contributions {Sz2_noco=}")
    #    print(f"    [+] 1/2(<S_+S+-> + <S_-S_+>) {S2_spin_flip}")
    #    
    #return S2
    S2  = 0.5*(nup+ndn)+rho_ud.trace()*rho_du.trace()
    S2 -= _trMM(rho_uu,rho_dd)

    Sz2  = 0.25*(nup+ndn) + 0.25*(nup-ndn)**2
    Sz2 -= 0.25*(_trMM(rho_uu,rho_uu)+_trMM(rho_dd,rho_dd))
    Sz2 -= -0.25*(_trMM(rho_ud,rho_du)+_trMM(rho_du,rho_ud))

    if verbose:
      print(f"<S^2> = {S2=}")
      print(f"       [+] <Sz^2> {Sz2=}")
    S2 += Sz2
    if S2.imag > 1e-8:
      warn(f"Warning! S^2 imaginary part is large:{S2}")
    return S2.real

def spin_spin(rdm,spin_symm:SpinSymm=SpinSymm.COLLINEAR, resolveXY=False):
    r"""
    Compute the spin squared operator as:
    
    :math:`<S_i S_j> = <S^z_i S^z_j + \frac{1}{2}(S^+_i S^-_j + S^-_i S^+_j)>`

    Parameters
    ----------
    rdm:np.array|list
        the 1-body reduced density matrix (rdm). See notes for correct input shape.
    spin_symm:SpinSymm:str
        the spin symmetry of the rdm. if spin_symm is a SpinSymm instance, it is used as is.
        if spin_symm is a string, it is converted to a SpinSymm instance. Accepted string
        values are {'rhf','uhf','ghf','closed','collinear','noncollinear'}. String values
        are not case-sensitive.
    resolveXY:bool
        return a the XY and Z contributions seperately

    Returns
    -------
    spin_spin:np.array
        the expectation value of the spin-spin operator.
        If resolveXY is true, return (SXXYY,SZZ)

    Notes
    -----
    Input rdm(s) are either a 2D numpy array or a list of 2D numpy arrays. If a list is provided,
        it must have length of 1 for closed-shell or noncollinear rmds, or 2 for collinear systems.
        For Collinear systems, the first rdm in the list is the spin-up rdm and the second is the spin-down rdm.
    The correct input shape for the rdm depends on the spin symmetry of the system:
        #. rhf / closed: an array of shape (M,M) where M is the number of spatial orbitals
        #. uhf / collinear: a list of 2 arrays of shape (M,M) or an array of shape (2M,M) where M is the number of spatial orbitals
        #. ghf / noncollinear: (2M,2M) where M is the number of spatial orbitals
    """
    rho_uu,rho_dd,rho_ud,rho_du = _process_rdm(rdm,spin_symm=spin_symm)
    N = rho_uu.shape[0]
    SzSz = np.zeros((N,N),dtype=rho_uu.dtype)
    SXY = np.zeros((N,N),dtype=rho_uu.dtype)
    for ref in range(N):
      for i in range(ref,N):
        if i==ref:
          SzSz[ref,i] += rho_uu[ref,ref]*(1-rho_dd[i,i])\
                      + rho_dd[ref,ref]*(1-rho_uu[i,i])
          SzSz[ref,i] -= rho_ud[ref,i]*(-rho_du[i,ref])\
                        + rho_du[ref,i]*(-rho_ud[i,ref])

          # 0.5*(XX+YY) = 1/4 n_is(1-n_is')
          SXY[ref,i] += (rho_uu[ref,i]*(1-rho_dd[i,ref])\
                    + rho_dd[ref,i]*(1-rho_uu[i,ref]))
          SXY[ref,i] += (rho_ud[i,i]*(rho_du[ref,ref])\
                    + rho_du[i,i]*(rho_ud[ref,ref]))
        else:
          SzSz[ref,i] += rho_uu[ref,ref]*(rho_uu[i,i]-rho_dd[i,i])\
                       + rho_dd[ref,ref]*(rho_dd[i,i]-rho_uu[i,i])

          # -<Cdag_i C_j> <Cdag_j C_i>
          SzSz[ref,i] -= rho_uu[ref,i]*rho_uu[i,ref]\
                        +rho_dd[ref,i]*rho_dd[i,ref]

          SzSz[ref,i] -= rho_ud[ref,i]*(-rho_du[i,ref])\
                       + rho_du[ref,i]*(-rho_ud[i,ref])
          # 0.5*(XX+YY)
          SXY[ref,i] += (rho_uu[ref,i]*(-rho_dd[i,ref])\
                    + rho_dd[ref,i]*(-rho_uu[i,ref]))
          SXY[ref,i] += (rho_ud[i,i]*(rho_du[ref,ref])\
                    + rho_du[i,i]*(rho_ud[ref,ref]))
    SzSz /= 4.0
    SXY  /= 2.0

    SzSz += np.triu(SzSz,k=1).T.conj()
    SXY += np.triu(SXY,k=1).T.conj()
    if resolveXY: return np.stack([SXY,SzSz])
    return SXY+SzSz

def local_spin(rdm,spin_symm:SpinSymm=SpinSymm.COLLINEAR):
    r"""
    Compute the expectation value of the x,y,z-component of the spin operator
    on each site as:

    :math:`\langle S^\alpha_i \rangle = \frac{1}{2}(n_{i\uparrow} - n_{i\downarrow})`

    Parameters
    ----------
    rdm:np.array|list
        the 1-body reduced density matrix (rdm). See spin_spin notes for correct input shape.
    spin_symm:SpinSymm:str
        the spin symmetry of the rdm. if spin_symm is a SpinSymm instance, it is used as is.
        if spin_symm is a string, it is converted to a SpinSymm instance. Accepted string
        values are {'rhf','uhf','ghf','closed','collinear','noncollinear'}. String values
        are not case-sensitive.
    
    Returns
    -------
    spin_vec:np.array
        (3,M) array; the expectation value of the spin operator at each site

    """
    rho_uu,rho_dd,rho_ud,rho_du = _process_rdm(rdm,spin_symm=spin_symm)
    Z = 0.5*(rho_uu.diagonal() - rho_dd.diagonal())
    X = 0.5*(rho_ud.diagonal() + rho_du.diagonal())
    Y = 0.5*1j*(rho_ud.diagonal() - rho_du.diagonal())
    return np.stack([X,Y,Z])

def spin_z(rdm,spin_symm:SpinSymm=SpinSymm.COLLINEAR):
    r"""
    Compute the expectation value of the z-component of the spin operator as:
    
    :math:`\langle S_z \rangle = \frac{1}{2}(\sum_{i} n_{i\uparrow} - n_{i\downarrow})`

    Parameters
    ----------
    rdm:np.array|list
        the 1-body reduced density matrix (rdm). See notes for correct input shape.
    spin_symm:SpinSymm:str
        the spin symmetry of the rdm. if spin_symm is a SpinSymm instance, it is used as is.
        if spin_symm is a string, it is converted to a SpinSymm instance. Accepted string
        values are {'rhf','uhf','ghf','closed','collinear','noncollinear'}. String values
        are not case-sensitive.
    
    Returns
    -------
    spin_z:float
        the expectation value of the z-component of the spin operator

    Notes
    -----
    Input rdm(s) are either a 2D numpy array or a list of 2D numpy arrays. If a list is provided,
        it must have length of 1 for closed-shell or noncollinear rmds, or 2 for collinear systems.
        For Collinear systems, the first rdm in the list is the spin-up rdm and the second is the spin-down rdm.
    The correct input shape for the rdm depends on the spin symmetry of the system:
        #. rhf / closed: an array of shape (M,M) where M is the number of spatial orbitals
        #. uhf / collinear: a list of 2 arrays of shape (M,M) or an array of shape (2M,M) where M is the number of spatial orbitals
        #. ghf / noncollinear: (2M,2M) where M is the number of spatial orbitals
    """
    rho_uu,rho_dd,_,_ = _process_rdm(rdm,spin_symm=spin_symm)

    Sz = 0.5*(np.trace(rho_uu) - np.trace(rho_dd))

    return Sz
