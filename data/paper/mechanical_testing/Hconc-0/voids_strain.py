import numpy as np
from ase.io import read, write
from ase import Atoms
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import math
import concurrent.futures
import argparse
import matplotlib.pyplot as plt
plt.style.use('~/plot.mplstyle')
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
import glob
import natsort


def compute_void_radius(atoms, spacing, cutoff, min_points):
    cell = atoms.get_cell()
    cell_vol=atoms.get_volume()
    positions = atoms.get_positions()
    shifts = np.array([[i,j,k] for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)])
    all_pos = (positions[:,None,:] + shifts[None,:,:].dot(cell)).reshape(-1,3)
    tree = cKDTree(all_pos)

    lengths = np.linalg.norm(cell, axis=1)
    

    n_pts = np.maximum(np.ceil(lengths / spacing).astype(int), 2)
    grid_axes = [np.linspace(0, 1, n) for n in n_pts]
    fx, fy, fz = np.meshgrid(*grid_axes, indexing='ij')
    grid_frac = np.stack([fx.ravel(), fy.ravel(), fz.ravel()], axis=1)
    grid_cart = grid_frac.dot(cell)

    dists, _ = tree.query(grid_cart, k=1)
    void_mask = dists > cutoff
    void_pts = grid_cart[void_mask]
    if void_pts.size == 0:
        return 0.0, void_pts, cell

    eps = spacing * 1.2
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(void_pts)
    voxel_vol = spacing ** 3
    volumes = []
    keep_pts = []
    for lbl in np.unique(labels):
        pts = void_pts[labels == lbl]
        if pts.shape[0] < min_points:
            continue
        volumes.append(pts.shape[0] * voxel_vol)
        keep_pts.append(pts)
    total_vol = sum(volumes)
    void_all = np.vstack(keep_pts) if keep_pts else np.empty((0, 3))

    # Convert volume to equivalent sphere radius
    r_eq = (3 * total_vol / (4 * math.pi)) ** (1 / 3) if total_vol > 0 else 0.0
    return r_eq, void_all, cell, cell_vol, total_vol


def compute_density(atoms):
    mass_u = atoms.get_masses().sum()         
    volume_A3 = atoms.get_volume()            
    mass_g = mass_u * 1.66054e-24            
    volume_cm3 = volume_A3 * 1e-24       
    return mass_g / volume_cm3 if volume_cm3 > 0 else 0.0


def export_xyz(pts, cell, filename, symbol='Ar'):
    if pts.size == 0:
        return
    atoms = Atoms(symbols=[symbol] * len(pts), positions=pts, cell=cell, pbc=True)
    write(filename, atoms, format='xyz')


def calc_pores(traj, spacing=0.3, cutoff=3, min_points=3, out='voids.xyz', symbol='Ar', threads=1):
    
    r0, pts0, cell0, vol0, pore0 = compute_void_radius(traj[0], spacing, cutoff, min_points)
    density = compute_density(traj[0])
    print(f"Density of first structure: {density:.3f} g/cm³")
    export_xyz(pts0, cell0, out, symbol)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(compute_void_radius, at, spacing, cutoff, min_points)
                   for at in traj]
        results = [f.result() for f in futures]
        radii = [r[0] for r in results]
        cell_v = [r[3] for r in results]
        pore_v = [r[4] for r in results]

    return np.array(radii), np.array(pore_v), np.array(cell_v)

cms=1/2.54
# H concentrations
Hconcs = np.arange(0, 5, 5)

# Define custom color scale
colors_rgb = [
    mcolors.to_rgb('#e97cbb'),  # pink (low H)
    mcolors.to_rgb('#f9e7bb'),  # yellow (mid H)
    mcolors.to_rgb('#4b57cc')   # blue (high H)
]
custom_cmap = LinearSegmentedColormap.from_list('custom_hsla_gradient', colors_rgb)
norm = mcolors.Normalize(vmin=min(Hconcs), vmax=max(Hconcs))  # Normalize Hconc range 

# Use subplots to manage Axes
# fig, axes = plt.subplots(1,2, figsize=(12*cms, 6*cms))

strains=np.arange(0, 0.3, 0.001)
repeats=np.arange(1,6)
cut=100

# Plotting
for Hconc in Hconcs:
    for i in repeats:
        color = custom_cmap(norm(Hconc))

        files = natsort.natsorted(glob.glob(f'/u/vld/spet5633/Silicon/a-Si_potential/DFT_1200eV/Testing/Paper/tensile_testing/TT/Hconc-{Hconc:.0f}/{i:.0f}/dump/*.atom'))
        frames = [read(f) for f in files]

        # try:
        #     pore_v=np.loadtxt(f'/Users/louiserosset/Documents/SiH/Testing/Paper/tensile_testing_ini_quenches/TT/Hconc-{Hconc:.0f}/{i:.0f}/pore_v-{cut}.txt')
        #     radii= np.loadtxt(f'/Users/louiserosset/Documents/SiH/Testing/Paper/tensile_testing_ini_quenches/TT/Hconc-{Hconc:.0f}/{i:.0f}/radii-{cut}.txt')
        #     cell_v=np.loadtxt(f'/Users/louiserosset/Documents/SiH/Testing/Paper/tensile_testing_ini_quenches/TT/Hconc-{Hconc:.0f}/{i:.0f}/cell_v-{cut}.txt')

        # except:
        #     # for atoms in frames:
            #     del[atoms[atoms.numbers==1]]
            #     atoms.numbers=np.ones(len(atoms))*14
            # print(frames[0])

        radii, pore_v, cell_v=calc_pores(frames, spacing=0.25, cutoff=2.5, min_points=3, threads=32)
        np.savetxt(f'/u/vld/spet5633/Silicon/a-Si_potential/DFT_1200eV/Testing/Paper/tensile_testing/TT/Hconc-{Hconc:.0f}/{i:.0f}/cell_v-{cut}.txt', cell_v)
        np.savetxt(f'/u/vld/spet5633/Silicon/a-Si_potential/DFT_1200eV/Testing/Paper/tensile_testing/TT/Hconc-{Hconc:.0f}/{i:.0f}/pore_v-{cut}.txt', pore_v)
        np.savetxt(f'/u/vld/spet5633/Silicon/a-Si_potential/DFT_1200eV/Testing/Paper/tensile_testing/TT/Hconc-{Hconc:.0f}/{i:.0f}/radii-{cut}.txt', radii)

    # axes[0].plot(strains, pore_v/cell_v*100, color=color)
    # axes[1].scatter(Hconc, np.max(radii), color=color)


# fig.tight_layout()
# plt.savefig('pores-Hconc.svg')