# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from h5py import File
import numpy
from pyscf.pbc import gto, tools
from pyscf.pbc.dft import numint
from pyscf import gto as molgto
import os
import sys
from mpi4py import MPI

def make_cell(latt,sp_label,atid,atpos,basis_='gthdzvp',pseudo_='gthpbe',mesh=None,prec=1e-8):
    """Generates a PySCF gto.Cell object.

    Parameters
    ----------
    latt: (3,3) floating point array.
      Lattice vectors in Bohr. Used to define cell.a
    sp_label: array of strings.
      Array containing species labels/symbols. 
    atid: integer array. 
      Array that contains the mapping between the atoms in the unit cell and their
      species label (as defined by sp_label).
    basis_: string. Default: 'gthdzvp'
      Basis set string. Used to define cell.basis.
    pseudo_: string. Default: 'gthpbe'
      Pseudopotential string. Used to define cell.pseudo.
    mesh: 3-d array. Default: None
      Used to define cell.mesh.
    prec: floating point. Default: 1e-8
      Used to define cell.precision     

    Returns
    -------
    cell: PySCF get.Cell object.
    """
    assert(len(atid) == len(atpos))
    assert(atpos.ndim == 2)
    assert(atpos.shape[1] == 3)
    cell = gto.Cell()
    cell.a = '''
       {} {} {}
       {} {} {}
       {} {} {}'''.format(latt[0,0],latt[0,1],latt[0,2],
                          latt[1,0],latt[1,1],latt[1,2],
                          latt[2,0],latt[2,1],latt[2,2])
    atom = ''
    for i in range(len(atid)):
        atom += '''{} {} {} {} \n'''.format(sp_label[atid[i]-1],
                                            atpos[i,0],atpos[i,1],atpos[i,2])
    cell.atom = atom
    cell.basis = basis_
    cell.pseudo = pseudo_
    if mesh is not None:
        cell.mesh = mesh
    cell.verbose = 5
    cell.unit = 'B'
    cell.precision = prec
    cell.build()
    return cell

def write_esh5_orbitals(cell, name, kpts = numpy.zeros((1,3),dtype=numpy.float64)):
    """Write periodic AO basis to hdf5 file.

    Parameters
    ----------
    cell: PySCF get.Cell object
      PySCF cell object which contains information of the system, including 
      AO basis set, FFT mesh, unit cell information, etc.
    name: string 
      Name of hdf5 file.
    kpts: array. Default: numpy.zeros((1,3)
      K-point array of dimension (nkpts, 3)
    dtype: datatype. Default: numpy.float64
      Datatype of orbitals in file.   

    """

    def to_complex(array):
        shape = array.shape
        return array.view(numpy.float64).reshape(shape+(2,))
    nao = cell.nao_nr()

    fh5 = File(name,'w')
    coords = cell.gen_uniform_grids(cell.mesh)
    
    kpts = numpy.asarray(kpts)
    nkpts = len(kpts)
    norbs = numpy.zeros((nkpts,),dtype=int)
    norbs[:] = nao

    grp = fh5.create_group("OrbsG")
    dset = grp.create_dataset("reciprocal_vectors", data=cell.reciprocal_vectors())
    dset = grp.create_dataset("number_of_kpoints", data=len(kpts))
    dset = grp.create_dataset("kpoints", data=kpts)
    dset = grp.create_dataset("number_of_orbitals", data=norbs)
    dset = grp.create_dataset("fft_grid", data=cell.mesh)
    dset = grp.create_dataset("grid_type", data=int(0))
    nnr = cell.mesh[0]*cell.mesh[1]*cell.mesh[2]
    # loop over kpoints later
    for (ik,k) in enumerate(kpts):
        ao = numint.KNumInt().eval_ao(cell, coords, k)[0]
        fac = numpy.exp(-1j * numpy.dot(coords, k))
        for i in range(norbs[ik]):
            aoi = fac * numpy.asarray(ao[:,i].T, order='C')
            aoi_G = tools.fft(aoi, cell.mesh)
            aoi_G = aoi_G.reshape(cell.mesh).transpose(2,1,0).reshape(nnr)
            dset = grp.create_dataset('kp'+str(ik)+'_b'+str(i), data=to_complex(aoi_G))
    fh5.close()

def make_image_comm(nimage, comm=MPI.COMM_WORLD):
    """Splits a communicator into image communicators, consistent with QE partitioning.
       nimage consecutive ranks in comm belong to the same image communicator.
       The number of distinct image communcators is comm.size/nimage. 

    Parameters
    ----------
    nimage: integer
      Number of image communicators, must divide comm.size.
    comm: mpi4py MPI comunicator. Default: MPI.COMM_WORLD
      A valid mpi4py communicator. 

    Returns
    -------
    intra_image: mpi4py MPI comunicator.
      Communicator between mpi tasks within an image.
    inter_image:mpi4py MPI comunicator.
      Communicator between mpi tasks on different images, but having the same rank in the image.
    """
    parent_nproc = comm.size
    parent_mype = comm.rank
    assert( parent_nproc%nimage == 0 )
    nproc_image = parent_nproc / nimage
    my_image_id = parent_mype / nproc_image
    me_image    = parent_mype%nproc_image
    intra_image = comm.Split(my_image_id,comm.rank)
    inter_image = comm.Split(me_image,comm.rank)
    return intra_image, inter_image

def get_qe_information(qe_prefix, qe_outdir, outdir='./qedrv',verbose=False):
    """Retreats basic information from a QE calculation. Assumes QE was compiled with HDF5 support.
       Requires a pre-existing, successful QE run.

    Parameters
    ----------
    qe_prefix: string
        prefix parameter from QE run.
    qe_outdir: string
        outdir parameter from QE run. (location of QE files).
    outdir: string. Default: ./qedrv
        Output directory of the driver. Does not need to be the same as the QE parameter.
    verbose: bool. Default: False
        Print information. 

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
        'outdir' : string           # location of output files 
    """
    # initialize driver:
    #nkpts, nat, nsp, npwx, ngm, mesh = pyscf_driver_init(inter_size, npools, intra_size/npools,
    #                                                     norb, qe_prefix, fname, verbose)

    # there should be a built-in way to do this
    def find_element_in_list(tree, tag):
      assert(len(tree) > 0)
      for el in tree:
        if el.tag == tag:
          return el
      raise RuntimeError("Could not find tag in XML tree")

    import math
    import xml.etree.ElementTree as xml
    tree = xml.ElementTree(file=qe_outdir+"/"+qe_prefix+".xml")
    base = list(tree.getroot())
    output_xml = find_element_in_list(base, "output")  
    species_xml = find_element_in_list(output_xml, "atomic_species")
    nsp = int(species_xml.attrib['ntyp'])
    atm_labels = []
    for s in species_xml:
      atm_labels.append(s.attrib['name'])
    atms = numpy.array(atm_labels)  

    atom_struct_xml = find_element_in_list(output_xml, "atomic_structure")
    atom_pos_xml = find_element_in_list(atom_struct_xml, "atomic_positions")
    cell_xml = find_element_in_list(atom_struct_xml, "cell")
    nat = int(atom_struct_xml.attrib['nat'])
    alat = float(atom_struct_xml.attrib['alat'])
    tpiba = 2.0 * math.pi / alat
    atom_ids = numpy.zeros((nat),dtype=int)
    atom_pos = numpy.zeros((nat,3),dtype=float)
    for i,at in enumerate(atom_pos_xml):
      # fortran returns 1-based indexing, so doing the same here	
      atom_ids[i] = atm_labels.index(at.attrib['name'])+1
      pos = at.text.split(' ')
      assert(len(pos)==3)
      for k,v in enumerate(pos):
        atom_pos[i,k] = float(v)  
    latt = numpy.zeros((3,3),dtype=float)
    assert(len(list(cell_xml))==3)
    for i,ai in enumerate(cell_xml):
      vi = ai.text.split(' ')
      assert(len(vi)==3)
      for k,vk in enumerate(vi):
        latt[i,k] = float(vk) 
    
    basis_xml = find_element_in_list(output_xml, "basis_set")
    fft_grid_xml = find_element_in_list(basis_xml, "fft_grid")
    mesh = numpy.zeros(3,dtype=int)
    mesh[0] = int(fft_grid_xml.attrib['nr1'])
    mesh[1] = int(fft_grid_xml.attrib['nr2'])
    mesh[2] = int(fft_grid_xml.attrib['nr3'])
    ngm = int(find_element_in_list(basis_xml,"ngm").text)
    npwx = int(find_element_in_list(basis_xml,"npwx").text)

    band_xml = find_element_in_list(output_xml, "band_structure")
    nkpts = int(find_element_in_list(band_xml, "nks").text)
    kpts = numpy.zeros((nkpts,3),dtype=float)
    nk=0
    for blk in band_xml:
      if blk.tag == 'ks_energies':
        kpi = find_element_in_list(blk,"k_point").text.split() 
        assert(len(kpi)==3)
        for i,ki in enumerate(kpi):
          kpts[nk][i] = float(ki)*tpiba
        nk=nk+1

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
               'outdir' : outdir 
               }
    if verbose: 
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

def gen_qe_gto(qe_info,bset,x=[],
               fname='pyscf.orbitals.h5',prec=1e-12):
    """Write periodic AO basis set in real space to hdf5 file.
        This routine constructs a new gaussian basis set from a OptimizableBasisSet
        object and an array of optimizable parameters. 
        With the resulting basis set, a new gto.Cell object is constructed consistent 
        with the QE calculation and used to generate the periodic AO basis set.

    Parameters
    ----------
    qe_info: Python Dictionary.
        Dictionary with information from QE calculation, generated by qe_driver_init.
    bset: Object of type OptimizableBasisSet.
        Contains information about a (possibly dynamic) basis set.
    x: fp array. Default: [] (no variable parameters in bset)
        Array with variational parameters in the bset object.
    fname: string. Default: 'pyscf.orbitals.h5'
        Name of hdf5 file.
    prec: floating point number. Default: 1e-12
        Precision used to generate AO orbitals in real space. Controls sum over periodic images. 

    Returns
    -------
    nao: integer
        Number of atomic orbitals generates.
    """
    assert(len(x) == bset.number_of_params)
    basis = {}
    for I, atm in enumerate(qe_info['species']):
        basis.update({atm: molgto.parse( bset.basis_str(atm,x) )})

    cell = make_cell(qe_info['latt'],
                 qe_info['species'],
                 qe_info['at_id'],
                 qe_info['at_pos'],
                 basis_=basis,
                 mesh=qe_info['mesh'],prec=prec)
    nao = cell.nao_nr()
    write_esh5_orbitals(cell, qe_info['outdir']+"/"+fname, kpts=qe_info['kpts'])
    return nao

