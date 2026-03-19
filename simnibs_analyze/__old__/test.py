# Fichier à visualiser
nii_file = "/Users/hippolyte.dreyfus/Desktop/hemianotACS/Data/derivatives/simnibs-config-healthyV2/0001/simulations/simulation_simulation_fef_hemianotacs_bc8ae6ee/mni_volumes/0001_TDCS_1_scalar_MNI_magnE.nii.gz"

# # Affichage de la carte en vue orthogonale
# display = plotting.plot_stat_map(
#     nii_file,
#     display_mode="ortho",   # axial + sagittal + coronal
#     cut_coords=None,  # Auto-détection des coordonnées
#     colorbar=True,
#     cmap="hot",
#     title="Electric Field Magnitude (magnE)"
# )

# fig = plotting.plot_surf_stat_map(
#     nii_file,
#     "map_on_surface.gii",
#     hemi="left",
#     view="lateral",   # lateral, medial, dorsal, ventral
#     colorbar=True
# )


import pyvista as pv
import nibabel as nib
import numpy as np

# Charger nifti
img = nib.load(nii_file)
data = img.get_fdata()

# Récupérer infos spatiales
affine = img.affine
spacing = img.header.get_zooms()[:3]
origin = affine[:3, 3]

# Créer grille
grid = pv.ImageData()

grid.dimensions = np.array(data.shape) + 1
grid.spacing = spacing
grid.origin = origin

# Ajouter données
grid.cell_data["values"] = data.flatten(order="F")

# Plot offscreen
plotter = pv.Plotter(off_screen=True)
plotter.add_volume(grid, cmap="inferno")
plotter.show(screenshot="screenshot.png")
