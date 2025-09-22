#!/bin/bash

cwd=`pwd`

cd scf
python3 scf.py > scf.out
cd $cwd

cd ham
python3 ham.py > ham.out
cd $cwd

if [ ! -d afqmc/soc ]; then
  mkdir -p afqmc/soc
fi
cd afqmc/soc
write_afqmc_json -i ../../ham/afqmc_soc.h5 -b 1200 -s 10 -ss 5
cd $cwd

if [ ! -d afqmc/sf ]; then
  mkdir -p afqmc/sf
fi
cd afqmc/sf
write_afqmc_json -i ../../ham/afqmc_sf.h5 -b 1200 -s 10 -ss 5 
cd $cwd
