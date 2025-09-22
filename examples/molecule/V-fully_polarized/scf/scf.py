#!/usr/bin/env python3

# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, mcscf


def main():
    """
    Computing the energy of ferro-magnetically coupled
       Vandium atoms.       
    """

    # 0. use an isolated vanadium atom to build a guess for the ferro-magnetic case
    single_mol = gto.M(
        atom='V 0. 0. 0.',
        basis='ccpvdz',
        spin=3,
        verbose=5
    )

    mf = scf.ROHF(single_mol).newton()
    mf.chkfile = 'rohf.chk'
    mf.kernel()

    mf.analyze()

    #ncas_list = [6+3+5 + i for i in range(12)]
    ncas_list= [32]
    e_cas = []

    
    for ncas in ncas_list:

        #if ncas > 24:
        #    break
        
        mycas = mcscf.CASCI(mf,ncas,3)
        result =  mycas.kernel()
        #print(f"{ncas} {e}")
        e_cas.append(result[0])


    for n,e in zip(ncas_list,e_cas):
        print(f"{n} : {e}")

    #plt.plot(ncas_list[:len(e_cas)],e_cas)
    #plt.show()

if __name__ == '__main__':
    main()
