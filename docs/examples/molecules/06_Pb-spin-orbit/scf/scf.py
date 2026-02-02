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

from pyscf import gto, scf


def main():
    
    ecp_soc = {
        'Pb' : '''
        Pb nelec 60
        Pb ul
        2       1.0000000              0.0000000
        Pb S
        2      12.2963030            281.2854990
        2       8.6326340             62.5202170
        Pb P
        2      10.2417900             72.2768970      -144.553795
        2       8.9241760            144.5910830       144.591083
        2       6.5813420              4.7586930        -9.517385
        2       6.2554030              9.9406210         9.940621
        Pb D
        2       7.7543360             35.8485070       -35.848507
        2       7.7202810             53.7243420        35.816228
        2       4.9702640             10.1152560       -10.115256
        2       4.5637890             14.8337310         9.889154
        Pb F
        2       3.8875120             12.2098920        -8.139928
        2       3.8119630             16.1902910         8.095145
        Pb G
        2       5.6915770             -9.0966650         4.548332
        2       5.7155670            -11.5319960        -4.612798
        '''
    }

    
    with open("aug-cc-pvdz-pp.dat",'r') as f:
        basis = f.read()

    mol = gto.M(
        atom="Pb 0. 0. 0.",
        basis=basis,
        ecp=ecp_soc,
        charge=-1,
        spin=3,
        verbose=4
    )

    # 1. Compute ROHF orbitals to use as a basis.
    mf = scf.ROHF(mol)
    mf.chkfile = 'rohf.chk'
    mf.kernel()

    print(f"ROHF electronic energy: {mf.energy_elec()}")

    # 2.a. compute a "No SOC" trial wavefunction
    mf = scf.GHF(mol=mol)
    mf.chkfile = 'ghf.chk'
    mf.kernel()

    print(f"GHF electronic energy: {mf.energy_elec()}")

    # 2.b. compute a "SOC" trial wavefunction
    mf = scf.GHF(mol=mol)
    mf.chkfile = 'ghf_soc.chk'
    mf.with_soc = True
    mf.kernel()

    print(f"SOC-GHF electronic energy: {mf.energy_elec()}")


if __name__ == '__main__':
    main()
