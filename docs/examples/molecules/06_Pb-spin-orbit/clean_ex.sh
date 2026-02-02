#!/bin/bash

cwd=`pwd`

cd scf
rm scf.out *.chk
cd $cwd

cd ham
rm ham.out *.h5
cd $cwd

rm -r ./afqmc
