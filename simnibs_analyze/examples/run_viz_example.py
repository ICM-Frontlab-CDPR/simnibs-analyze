


import simnibs reader as snr
from simnibs_analyze import Visualizer

sim = snr.simulation...
seg = snr.segmentation...
sim.set_segmentation(seg)


efield = results.magnE 


roi = efield.get_roi(coords=[28, -8, 54], radius=5.0)


cleaned_roi = roi.postprocess(
    smooth_fwhm=2.0,         # Gaussian smoothing in mm (None to skip)
    outlier_method='iqr',    # 'iqr' or 'z'
    portion=0.5,            # e.g. 0.95 to trim to central 95% 
)

# VIZ
viz = Visualizer(sim) ## ca prend bien une sim ? pas une Roiresults ?



# colocalisation de labelprep et des efields, (sans le scalp --> ca cest gerer dans le reader pas dans le visualizer)
tissues_efield = viz.coloc(label_prep_brain, cleaned_roi)  #un niftii 4D jimagine ?



tissues_efield.plot_2D_acs(ref = x,y,z )
tissues_efield.plot_2D_parallel_slicing(ref = x,y,z )

tissues_efield.plot_3D_efields_figures( camera_angle= , relative_shade= (0.5,1)) #relative shade must be the size of the 4th dimension of .coloc 


#### FOR COHORT LEVEL 

##viz.set_color_scale( array of efield). ## HOW TO 






# -----------------
# # later
# SETUP des differentes colocalisations dont Toni a besoin
# scalp =

# set_opti
# set_simu
# viz = Visualizer(cleaned_roi)

# analyse = Analyzer(cleaned_roi)