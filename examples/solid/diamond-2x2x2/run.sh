#!/bin/bash

cwd=`pwd`
if [ ! -f scf/chkfile.h5 ]; then
# step 1: vanilla PySCF
echo "SCF"
cd scf
python3 scf.py > scf.out
# step 2: orthogonalize AOs (prepare basis set)
echo "AO rotation"
python3 orthoAO.py
cd $cwd
fi

# step 3: write hamiltonain w/o using k-point symmtry
if [ ! -d ham ]; then
mkdir ham
fi
if [ ! -f ham/afqmc.h5 ]; then
echo "ham"
cd ham
mpirun -n 8 pyscf_to_afqmc.py -i ../scf/chkfile.h5 -o afqmc.h5 -a -t 1e-5 -v > ham.out
cd $cwd
fi

# step 4: exploit k-point symmtry
if [ ! -d kpham ]; then
mkdir kpham
fi
if [ ! -f kpham/afqmc.h5 ]; then
echo "kpham"
cd kpham
mpirun -n 8 pyscf_to_afqmc.py -i ../scf/chkfile.h5 -o afqmc.h5 -a -kp -t 1e-5 -v > ham.out
cd $cwd
fi

# step 5: run AFQMC
if [ ! -d afqmc ]; then
mkdir afqmc
cd afqmc
write_afqmc_json.py -i ../ham/afqmc.h5 -b 200
cd $cwd
fi

echo "kp input"
if [ ! -d kpafqmc ]; then
mkdir kpafqmc
cd kpafqmc
write_afqmc_json.py -i ../kpham/afqmc.h5 -b 200
cd $cwd
fi
