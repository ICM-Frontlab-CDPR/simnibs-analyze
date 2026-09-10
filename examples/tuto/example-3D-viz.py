# t1_path = "/home/hippolyted/Data/stimSD/_tmp_simnibs-outputs/1-simnibs-preps/0001/m2m_0001/T1.nii.gz"
# efield_path = "/home/hippolyted/Data/stimSD/_tmp_simnibs-outputs/2-simnibs-simu/0001/simulations/simulation_AFFT-left_8dbde024/subject_volumes/0001_TDCS_1_scalar_magnE.nii.gz"
# roi_path = "/home/hippolyted/Data/hemianotACS/data/derivatives/synthstroke-masks/0008/3DT1_4_lesion.nii.gz"


# # ====================== EXEMPLES ======================

# ## vols doit etre au format niivue.
# vols = [
#     {"path": t1_path, "colormap": "gray", "opacity": 1.0},
#     {
#         "path": efield_path,
#         "colormap": "hot",
#         "cal_min": 0.1,
#         "cal_max": 1.0,
#         "opacity": 0.7,
#     },
#     {"path": roi_path, "colormap": "red", "opacity": 0.4},
# ]

# # Rendu de base
# render_3d(vols, "efield.png", azimuth=120, elevation=15)

# # Orientations (caméra)
# render_3d(vols, "vue_gauche.png", azimuth=270, elevation=0)  # profil gauche
# # render_3d(vols, "vue_droite.png", azimuth=90,  elevation=0)    # profil droit
# # render_3d(vols, "vue_face.png",   azimuth=180, elevation=0)    # face (antérieur)
# # render_3d(vols, "vue_dessus.png", azimuth=0,   elevation=90)   # vue du dessus
# # render_3d(vols, "vue_3qrt.png",   azimuth=135, elevation=20)   # trois-quarts

# # Balayage multi-angles (pour un GIF / une planche)
# for az in range(0, 360, 45):
#     render_3d(vols, f"rot_{az:03d}.png", azimuth=az, elevation=15)

# # Colormap différente (viridis) — T1 + E-field seulement
# vols_viridis = [
#     {"path": t1_path, "colormap": "gray", "opacity": 1.0},
#     {
#         "path": efield_path,
#         "colormap": "viridis",
#         "cal_min": 0.1,
#         "cal_max": 1.0,
#         "opacity": 0.8,
#     },
# ]
# render_3d(vols_viridis, "vue_viridis.png", azimuth=270, elevation=0)

# # Opacité : T1 translucide pour voir le champ "à travers"
# vols_transp = [
#     {"path": t1_path, "colormap": "gray", "opacity": 0.5},
#     {
#         "path": efield_path,
#         "colormap": "hot",
#         "cal_min": 0.1,
#         "cal_max": 1.0,
#         "opacity": 0.9,
#     },
#     {"path": roi_path, "colormap": "red", "opacity": 0.3},
# ]
# render_3d(vols_transp, "vue_opacite.png", azimuth=135, elevation=20)

# # Seuillage : ne garder que les valeurs fortes du champ
# vols_seuil = [
#     {"path": t1_path, "colormap": "gray", "opacity": 1.0},
#     {
#         "path": efield_path,
#         "colormap": "hot",
#         "cal_min": 0.3,
#         "cal_max": 0.8,
#         "opacity": 0.9,
#     },
# ]
# render_3d(vols_seuil, "vue_seuil.png", azimuth=135, elevation=20)

# # Options de scène : fond blanc + cube d'orientation (figure de publication)
# opts_pub = {
#     "backColor": [1, 1, 1, 1],
#     "isColorbar": True,
#     "isOrientCube": True,
#     "isRuler": False,
# }
# render_3d(vols, "vue_publication.png", nv_opts=opts_pub, azimuth=135, elevation=20)
