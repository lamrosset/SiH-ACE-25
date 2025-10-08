import ase
import natsort
import numpy as np
from numpy import random
from ase.io import read,write
from ase.neighborlist import *
from numpy.linalg import *
from ase import Atom, Atoms
from ase.visualize import view
import os


def adjust_neigh_pos2(atoms, neighbors):
    params=atoms.cell.lengths()
    pos=np.concatenate([[atoms.positions[i]] for i in neighbors])

    diffs=np.zeros((3))
    for i in range(len(diffs)):
        diffs[i]=np.abs(np.diff(pos[:,i]))

    for i in range(len(diffs)):
        if diffs[i]>params[i]/2:
            if pos[0,i]<params[i]/2:
                pos[0,i]+=params[i]
            else:
                pos[1,i]+=params[i]

    return pos



def get_Nvec(atoms, atom, neighbors):
    if len(neighbors)==2:
        v1=atoms.get_distance(atom, neighbors[0], mic=True, vector=True)
        v2=atoms.get_distance(atom, neighbors[1], mic=True, vector=True)
        Nvec=np.cross(v1,v2)
        Nvec=Nvec/np.linalg.norm(Nvec)

    else:        
        v1=atoms.get_distance(neighbors[0], neighbors[1], mic=True, vector=True)
        v2=atoms.get_distance(neighbors[0], neighbors[2], mic=True, vector=True)
        Nvec=np.cross(v1,v2)
        Nvec=Nvec/np.linalg.norm(Nvec)
    
    return Nvec



def dec_coord(atoms, selec_atom, neighbors, bondlen):
    if type(selec_atom) is list:
        atom=selec_atom[0]
    else:
        atom=selec_atom
    params=atoms.cell.lengths()

    if len(neighbors)==2 or len(neighbors)==3:
        Nvec=get_Nvec(atoms, atom, neighbors)
    else:
        raise ValueError('This is not a dangling bond defect.')
    

    #Choose best position of the atom on either side of the central atom
    temp1=atoms.positions[atom]+Nvec*bondlen
    atoms.append(Atom('H', position=temp1))
    dis1=min(atoms.get_distances(len(atoms)-1,np.delete(np.arange(0,len(atoms)-1), selec_atom), mic=True))

    temp2=atoms.positions[atom]-Nvec*bondlen
    atoms.append(Atom('H', position=temp2))
    dis2=min(atoms.get_distances(len(atoms)-1,np.delete(np.arange(0,len(atoms)-2), selec_atom), mic=True))

    if dis1<dis2:
        pos_newatom=temp2
    else:
        pos_newatom=temp1
    del atoms[[-2, -1]]

    for i in range(len(pos_newatom)):
        if pos_newatom[i]>params[i]:
            pos_newatom[i]-=params[i]
        elif pos_newatom[i]<0:
            pos_newatom[i]+=params[i]
    newatom=Atom('H', position=pos_newatom)
    atoms.append(newatom)

    return atoms
    


def decorate_shared(atoms, pick, neighbors, stretch):
    params=atoms.cell.lengths()
    pos_neighbors=adjust_neigh_pos2(atoms, neighbors)
    
    midpoint_pos=np.zeros(3)
    for pos in pos_neighbors:
        print(pos)
        midpoint_pos[0]+=pos[0]
        midpoint_pos[1]+=pos[1]
        midpoint_pos[2]+=pos[2]
    midpoint_pos/=len(neighbors)

    temp_atom=Atom('H', position=midpoint_pos)
    atoms.append(temp_atom)
    vec=atoms.get_distance(pick, len(atoms)-1, mic=True, vector=True)
    vec=vec/np.linalg.norm(vec)
    del atoms[len(atoms)-1]

    pos_newatom=midpoint_pos+vec*stretch #midpoint and some stretch
    for i in range(len(pos_newatom)):
        if pos_newatom[i]>params[i]:
            pos_newatom[i]-=params[i]
        elif pos_newatom[i]<0:
            pos_newatom[i]+=params[i]
    newatom=Atom('H', position=pos_newatom)
    atoms.append(newatom)

    return atoms



def create_void(atoms, pick, pneighbors, neigh, shared):
    if shared==0: #decorate each Si atom with a H (if it's 3 or 4 fold connected)
        for neighb in pneighbors:
            mneighbors=[]
            for line in neigh[:]:
                if line[0]==neighb:
                    mneighbors.append(line[1])
            mneighbors.remove(pick)

            if len(mneighbors)>4:
                pass

            elif len(mneighbors)==4:
                if random.randint(0,2):
                    mneighbors.pop(random.randint(0,4))
                    bondlen=random.normal(1.4,0.04)
                    atoms=dec_coord(atoms, [neighb, pick], mneighbors, bondlen)

            else:
                bondlen=random.normal(1.4,0.04)
                atoms=dec_coord(atoms, [neighb, pick], mneighbors, bondlen)

    elif shared==1: #decorate 2 Si atoms with 1H each, and 2 Si atoms with 1 shared H    
        for neighb in pneighbors[:2]:
            mneighbors=[]
            for line in neigh[:]:
                if line[0]==neighb:
                    mneighbors.append(line[1])
            mneighbors.remove(pick)

            while len(mneighbors)>=4:
                mneighbors.pop(random.randint(0,len(mneighbors)))
            
            bondlen=random.normal(1.4,0.04)
            atoms=dec_coord(atoms, [neighb, pick], mneighbors, bondlen)
        
        stretch=random.normal(0.1,0.02)
        atoms=decorate_shared(atoms, pick, pneighbors[2:], stretch)

    else: # add a H2 molecule
        print('H2 molecule')
        bondlen1=random.normal(0.37,0.01)
        bondlen2=random.normal(0.37,0.01)
        center=atoms.positions[pick]
        vec=atoms.get_distance(pick, pneighbors[0], mic=True, vector=True)
        vec=vec/np.linalg.norm(vec)
        at1=center+vec*bondlen1
        at2=center-vec*bondlen2

        atoms.append(Atom('H', position=at1))
        atoms.append(Atom('H', position=at2))

    return atoms


def rand_int(atoms, nb_int):
    for i in range(nb_int):
        dist=0
        while dist<1:
            params=atoms.cell.lengths()
            int_pos=np.zeros((3))
            int_pos[0]=random.random()*params[0]
            int_pos[1]=random.random()*params[1]
            int_pos[2]=random.random()*params[2]
            int_atom=Atom('H', position=int_pos)
            atoms.append(int_atom)
            dist=min(atoms.get_distances(len(atoms)-1,np.arange(0,len(atoms)-1)))
            del atoms[len(atoms)-1]
        
        int_atom=Atom('H', position=int_pos)
        atoms.append(int_atom)
    
    return atoms



def decorate_existing_defects(atoms, coord_dict, neigh):
    # Decorate dict of defects
    safe_ind=[]

    for atom in coord_dict[3]:
        safe_ind.append(atom)
        neighbors=[]
        for line in neigh[:]:
            if line[0]==atom:
                neighbors.append(line[1])
        
        bondlen=random.normal(1.55,0.08)
        atoms=dec_coord(atoms, atom, neighbors, bondlen)
        safe_ind.append(len(atoms)-1)

    for atom in coord_dict[2]:
        #print('pos of central atom is')
        #print(atoms.positions[atom])
        safe_ind.append(atom)
        neighbors=[]
        for line in neigh[:]:
            if line[0]==atom:
                neighbors.append(line[1])
        
        bondlen=random.normal(1.55,0.08)
        atoms=dec_coord(atoms, atom, neighbors, bondlen)
        
        neighbors.append(len(atoms)-1)
        bondlen=random.normal(1.55,0.08)
        atoms=dec_coord(atoms, atom, neighbors, bondlen)
    
        safe_ind.append(len(atoms)-2)
        safe_ind.append(len(atoms)-1)
    
    for atom in coord_dict[1]:
        print(atoms.info['label'])
        raise ValueError('Here is a structure with a 1 coordinated atom')

    return atoms, safe_ind



def voids(atoms, nb_voids, safe_ind, len_Si):
    del_atoms=[]
    i,j=neighbor_list('ij', atoms, cutoff={('Si', 'Si'): 2.85, ('Si', 'H'): 1.8,('H', 'H'): 1})
    neigh=np.stack((i.T,j.T), axis=1)
    coord = np.bincount(i)

    for count in range(nb_voids):
        # Choose atom to delete while avoiding protected atoms
        pick=random.randint(0,len_Si)
        while pick in safe_ind and coord[pick]!=4:
            pick=random.randint(0,len_Si)

        del_atoms.append(pick)
        safe_ind.append(pick)

        # Obtain list of neighbors of the to-be deleted atom
        neighbors=[]
        for line in neigh[:]:
            if line[0]==pick:
                neighbors.append(line[1])

        shared=random.choice(np.arange(0, 3), p=[0.5, 0.25, 0.25])
        atoms=create_void(atoms, pick, neighbors, neigh, shared)

    if del_atoms!=[]:
        del_atoms=np.sort(del_atoms)[::-1]
        del atoms[del_atoms]
            
    return atoms



def dec_and_void(atoms, nb_voids, nb_int, coord_dict, neigh):
    len_Si=len(atoms)
    atoms,safe_ind=decorate_existing_defects(atoms, coord_dict, neigh)
    atoms=voids(atoms, nb_voids, safe_ind, len_Si)
    atoms=rand_int(atoms, nb_int)
    
    return atoms


def dec_and_void_partial(atoms, nb_voids, nb_int, coord_dict, neigh):
    len_Si=len(atoms)
    atoms,safe_ind=decorate_existing_defects(atoms, coord_dict, neigh)
    atoms=voids(atoms, nb_voids, safe_ind, len_Si)
    atoms=rand_int(atoms, nb_int)
    
    return atoms


def hydrogenate_amorph(atoms, conc=random.normal(11,3)):
    i,j=neighbor_list('ij', atoms, 2.85)
    neigh=np.stack((i.T,j.T), axis=1)
    coord = np.bincount(i)
    coord_dict={0:[], 1:[], 2:[], 3:[]}
    
    for i in range(0, len(coord)):
        if coord[i]<4:
            coord_dict[coord[i]].append(i)

    count_defH=len(coord_dict[3])+len(coord_dict[2])*2 # Determine nb of H atoms from bond defects
    # conc=random.normal(11,3) #Obtain [H]
    totH=round(len(atoms)*conc*0.01/(1-conc*0.01)) # Determine how many total H atoms 
    print(totH)
    addH=int(totH-count_defH) # Determine how many additional H atoms
    print(addH)

    if addH<0:
        nb_int=0
        nb_voids=0

    elif addH>0 and addH<=2:
        nb_int=addH
        nb_voids=0

    else:
        nb_voids=0
        if (addH-nb_voids*3)>0:
            nb_int=int(addH-nb_voids*3)
        else:
            nb_int=0

    atoms=dec_and_void(atoms, nb_voids, nb_int, coord_dict, neigh)
    print(atoms)

    return atoms


def hydrogenate_other(atoms, conc=random.normal(11,3)):
    #Obtain [H]
    totH=round(len(atoms)*conc*0.01/(1-conc*0.01)) # Determine how many total H atoms 
    nb_int=totH
    nb_voids=0
    coord_dict={0:[], 1:[], 2:[], 3:[]}
    neigh=0

    atoms=dec_and_void(atoms, nb_voids, nb_int, coord_dict, neigh)
    return atoms


# Hconcs=np.arange(50,55,5)/100
# repeats=np.arange(1,6)

# for Hconc in Hconcs:
#     for i in repeats:
#         atoms=read(f'/Users/louiserosset/Documents/SiH/Testing/Paper/tensile_testing_ini_quenches/Hconc-0/{i}/out_str/out_anneal_SiH.data', format='lammps-data', style='atomic')
#         print(atoms)
#         n_H=int(Hconc*len(atoms))
#         print(n_H)
#         n_Si=len(atoms)-n_H
#         wdir=f'/Users/louiserosset/Documents/SiH/Testing/Paper/tensile_testing_ini_quenches/Hconc-{Hconc*100:.0f}/{i:.0f}'
#         os.system('mkdir -p '+wdir)
#         print('made dir')

#         i,j=neighbor_list('ij', atoms, 2.85)
#         neigh=np.stack((i.T,j.T), axis=1)
#         coord = np.bincount(i)
#         coord_dict={0:[], 1:[], 2:[], 3:[]}
        
#         for i in range(0, len(coord)):
#             if coord[i]<4:
#                 coord_dict[coord[i]].append(i)

#         count_defH=len(coord_dict[3])+len(coord_dict[2])*2 # Determine nb of H atoms from bond defects

#         addH=int(n_H-count_defH) # Determine how many additional H atoms

#         if addH<0:
#             nb_int=0
#             nb_voids=0

#         elif addH>0 and addH<=2:
#             nb_int=addH
#             nb_voids=0

#         else:
#             nb_voids=0
#             if (addH-nb_voids*3)>0:
#                 nb_int=int(addH-nb_voids*3)
#             else:
#                 nb_int=0

#         print(count_defH)
#         print(nb_int)
#         print(nb_voids)

#         hydrogenated=dec_and_void(atoms, nb_voids, nb_int, coord_dict, neigh)
#         write(f'{wdir}/SiH.data', hydrogenated, format='lammps-data', atom_style='atomic')