from quippy.potential import Potential
from ase.io import read, write
from pyace import PyACECalculator
from ase.build import cut, bulk, surface, molecule
from ase.build.tools import sort
import matplotlib.pyplot as plt

import numpy as np
from ase.filters import UnitCellFilter
from ase.optimize import BFGS


potfilename='../../../../../models/SiH-ACE-25.yaml'
calc=PyACECalculator(potfilename)

bulk_ace=read('../../bulk/bulk_ace_relaxed.xyz')
bulk_ace.calc=calc
bulk_ace_en=bulk_ace.get_potential_energy()/len(bulk_ace)


unrelaxed=read(f'./unrelaxed.xyz')
unrelaxed.calc=calc
unrelaxed_ace_en=unrelaxed.get_potential_energy()
area=unrelaxed.cell[0,0]*unrelaxed.cell[1,1]
unrelaxed_gamma_ace=(unrelaxed_ace_en-bulk_ace_en*len(unrelaxed))/(2*area)

surfaces=['3x3', '5x5', '7x7', '9x9']
surface_ens=[]
surface_gamma_ens=[]
surface_delta_gamma=[]

offsets=np.arange(len(surfaces)+1)

# Plot step-style bars
for i in range(len(surfaces)):
    surface=surfaces[i]
    offset=offsets[i]
    structure=read(f'./{surface}.xyz')

    structure.calc=calc

    ucf = UnitCellFilter(structure, hydrostatic_strain=True)
    opt = BFGS(ucf, trajectory='relax.traj')
    opt.run(fmax=0.001)
    write(f'./{surface}-relaxed-ace25.xyz', structure)
    structure_ace_en=structure.get_potential_energy()
    surface_ens.append(structure_ace_en)

    area=structure.cell[0,0]*structure.cell[1,1]
    gamma_ace=(structure_ace_en-bulk_ace_en*len(structure))/(2*area)
    surface_gamma_ens.append(gamma_ace)
    
    delta=-unrelaxed_gamma_ace+gamma_ace
    surface_delta_gamma.append(delta)


# Save
dictionary = {'bulk_ace_en':bulk_ace_en, 'unrelaxed_ace_en': unrelaxed_ace_en, 'unrelaxed_gamma_ace': unrelaxed_gamma_ace, 'surface_ens':surface_ens, 'surface_gamma_ens':surface_gamma_ens, 'surface_delta_gamma': surface_delta_gamma}
np.save('ace-25-preds-Si111.npy', dictionary) 