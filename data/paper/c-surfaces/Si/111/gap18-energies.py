from quippy.potential import Potential
from ase.io import read, write
# from pyace import PyACECalculator
from ase.build import cut, bulk, surface, molecule
from ase.build.tools import sort
import matplotlib.pyplot as plt

import numpy as np
from ase.filters import UnitCellFilter
from ase.optimize import BFGS


potfilename='../../../../../models/GAP18/gp_iter6_sparse9k.xml'
calc=Potential(name='IP GAP', param_filename=potfilename)


bulk=read('../../bulk/bulk_ace_relaxed.xyz')
bulk.calc=calc
ucf = UnitCellFilter(bulk, hydrostatic_strain=True)
opt = BFGS(ucf, trajectory='relax.traj')
opt.run(fmax=0.001)
bulk_ace_en=bulk.get_potential_energy()/len(bulk)
print(bulk_ace_en)


unrelaxed=read(f'./unrelaxed.xyz')
unrelaxed.calc=calc
unrelaxed_ace_en=unrelaxed.get_potential_energy()
area=unrelaxed.cell[0,0]*unrelaxed.cell[1,1]
unrelaxed_gamma_ace=(unrelaxed_ace_en-bulk_ace_en*len(unrelaxed))/(2*area)

surfaces=[3,5,7,9]
surface_ens=[]
surface_gamma_ens=[]
surface_delta_gamma=[]

offsets=np.arange(len(surfaces)+1)

# Plot step-style bars
for i in range(len(surfaces)):
    surface=surfaces[i]
    offset=offsets[i]
    structure=read(f'./{surface}x{surface}.xyz')
    structure.calc=calc

    L = structure.get_cell()
    bulkL = bulk.get_cell()
    scaling_factor = bulkL[0,0]*surface/L[0,0]
    structure.set_cell(L*scaling_factor, scale_atoms=True)

    ucf = UnitCellFilter(structure, hydrostatic_strain=True)
    opt = BFGS(ucf, trajectory='relax.traj')
    opt.run(fmax=0.001)
    write(f'./{surface}x{surface}-relaxed-gap18.xyz', structure)


    structure_ace_en=structure.get_potential_energy()
    surface_ens.append(structure_ace_en)

    area=structure.cell[0,0]*structure.cell[1,1]
    gamma_ace=(structure_ace_en-bulk_ace_en*len(structure))/(2*area)
    surface_gamma_ens.append(gamma_ace)
    
    delta=gamma_ace-unrelaxed_gamma_ace
    surface_delta_gamma.append(delta)


# Save
dictionary = {'bulk_ace_en':bulk_ace_en, 'unrelaxed_ace_en': unrelaxed_ace_en, 'unrelaxed_gamma_ace': unrelaxed_gamma_ace, 'surface_ens':surface_ens, 'surface_gamma_ens':surface_gamma_ens, 'surface_delta_gamma': surface_delta_gamma}
np.save('gap18-preds-Si111.npy', dictionary) 
