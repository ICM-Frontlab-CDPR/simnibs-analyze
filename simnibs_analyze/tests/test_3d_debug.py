# """Diagnostic script for ColocVolume 3D rendering pipeline."""

# import nibabel as nib
# import numpy as np

# # ── 1. Créer des volumes synthétiques ─────────────────────────────────────
# print("=" * 60)
# print("1. Création de volumes synthétiques")
# print("=" * 60)

# affine = np.eye(4)
# affine[:3, 3] = [-90, -126, -72]  # origin MNI-like
# shape = (91, 109, 91)

# # T1 : sphère pleine (simule un cerveau)
# t1_data = np.zeros(shape, dtype=np.float32)
# center = np.array(shape) // 2
# Z, Y, X = np.ogrid[: shape[0], : shape[1], : shape[2]]
# dist = np.sqrt((X - center[2]) ** 2 + (Y - center[1]) ** 2 + (Z - center[0]) ** 2)
# t1_data[dist < 35] = 1000.0
# t1_data[dist < 30] = 2000.0
# t1_img = nib.Nifti1Image(t1_data, affine)
# print(
#     f"  T1:     shape={t1_data.shape}, range=[{t1_data.min():.0f}, {t1_data.max():.0f}]"
# )

# # E-field : gradient dans la sphère
# ef_data = np.zeros(shape, dtype=np.float32)
# ef_data[dist < 25] = (25 - dist[dist < 25]) / 25.0  # 0→1
# ef_img = nib.Nifti1Image(ef_data, affine)
# print(
#     f"  EField: shape={ef_data.shape}, range=[{ef_data.min():.2f}, {ef_data.max():.2f}]"
# )
# print(f"          nonzero voxels: {np.count_nonzero(ef_data)}")

# # Mask : petite sphère excentrée
# mask_data = np.zeros(shape, dtype=np.float32)
# mask_center = center + np.array([10, 5, 0])
# dist_mask = np.sqrt(
#     (X - mask_center[2]) ** 2 + (Y - mask_center[1]) ** 2 + (Z - mask_center[0]) ** 2
# )
# mask_data[dist_mask < 8] = 1.0
# mask_img = nib.Nifti1Image(mask_data, affine)
# print(f"  Mask:   shape={mask_data.shape}, nonzero={np.count_nonzero(mask_data)}")

# # ── 2. Test _to_pv_image ─────────────────────────────────────────────────
# print("\n" + "=" * 60)
# print("2. Test _to_pv_image")
# print("=" * 60)

# try:
#     import pyvista as pv

#     print(f"  PyVista version: {pv.__version__}")
# except ImportError:
#     print("  ❌ PyVista non installé !")
#     exit(1)

# from visualizer import ColocVolume

# canonical = nib.as_closest_canonical(ef_img)
# data = np.squeeze(canonical.get_fdata()).astype(np.float32)

# print(f"  Canonical affine diagonal: {np.diag(canonical.affine)}")
# print(f"  Canonical zooms: {canonical.header.get_zooms()[:3]}")
# print(f"  Canonical origin: {canonical.affine[:3, 3]}")

# grid = ColocVolume._to_pv_image(canonical, data, key="test")
# print(f"  Grid dimensions: {grid.dimensions}")
# print(f"  Grid spacing:    {grid.spacing}")
# print(f"  Grid origin:     {grid.origin}")
# print(f"  Grid bounds:     {grid.bounds}")
# print(f"  Cell data keys:  {list(grid.cell_data.keys())}")
# print(f"  Cell data shape: {grid.cell_data['test'].shape}")
# print(
#     f"  Cell data range: [{np.nanmin(grid.cell_data['test']):.3f}, {np.nanmax(grid.cell_data['test']):.3f}]"
# )

# # ── 3. Test chaque type de rendu isolément ────────────────────────────────
# print("\n" + "=" * 60)
# print("3. Test rendu isolé — background (isosurface)")
# print("=" * 60)

# try:
#     p = pv.Plotter(off_screen=True)
#     canonical_t1 = nib.as_closest_canonical(t1_img)
#     d = np.squeeze(canonical_t1.get_fdata()).astype(np.float32)
#     g = ColocVolume._to_pv_image(canonical_t1, d, key="t1")
#     pts = g.cell_data_to_point_data()
#     thresh = float(np.percentile(d[d > 0], 15))
#     print(f"  Contour threshold: {thresh:.1f}")
#     print(f"  Point data keys: {list(pts.point_data.keys())}")
#     print(
#         f"  Point data range: [{pts.point_data['t1'].min():.1f}, {pts.point_data['t1'].max():.1f}]"
#     )
#     surface = pts.contour([thresh], scalars="t1")
#     print(f"  Surface n_points: {surface.n_points}, n_cells: {surface.n_cells}")
#     if surface.n_points > 0:
#         p.add_mesh(surface, color="white", opacity=0.15)
#         img = p.screenshot(return_img=True)
#         print(f"  ✅ Screenshot shape: {img.shape}, mean pixel: {img.mean():.1f}")
#     else:
#         print("  ❌ Isosurface vide !")
#     p.close()
# except Exception as e:
#     print(f"  ❌ Erreur: {e}")

# print("\n" + "=" * 60)
# print("4. Test rendu isolé — stat_map (volume rendering)")
# print("=" * 60)

# try:
#     p = pv.Plotter(off_screen=True)
#     canonical_ef = nib.as_closest_canonical(ef_img)
#     d = np.squeeze(canonical_ef.get_fdata()).astype(float)
#     print(f"  Data range avant NaN: [{d.min():.3f}, {d.max():.3f}]")
#     print(f"  Zeros: {np.count_nonzero(d == 0)}, Non-zeros: {np.count_nonzero(d != 0)}")
#     d_render = d.copy()
#     d_render[d_render == 0] = np.nan
#     print(
#         f"  Data range après NaN: [{np.nanmin(d_render):.3f}, {np.nanmax(d_render):.3f}]"
#     )
#     print(f"  NaN count: {np.count_nonzero(np.isnan(d_render))}")

#     g = ColocVolume._to_pv_image(canonical_ef, d_render, key="values")
#     print(f"  Grid bounds: {g.bounds}")

#     # Test add_volume
#     p.add_volume(g, cmap="hot")
#     img = p.screenshot(return_img=True)
#     print(f"  ✅ Screenshot shape: {img.shape}, mean pixel: {img.mean():.1f}")
#     # mean pixel > 0 et < 255 = quelque chose a été rendu
#     if img.mean() < 1.0:
#         print("  ⚠️  Image quasi-noire — volume probablement invisible")
#     elif img.mean() > 254.0:
#         print("  ⚠️  Image quasi-blanche — problème de clim ?")
#     else:
#         print("  ✅ Rendu semble OK")
#     p.close()
# except Exception as e:
#     print(f"  ❌ Erreur: {e}")
#     import traceback

#     traceback.print_exc()

# print("\n" + "=" * 60)
# print("5. Test rendu isolé — overlay (surface seuillée)")
# print("=" * 60)

# try:
#     p = pv.Plotter(off_screen=True)
#     canonical_m = nib.as_closest_canonical(mask_img)
#     d = np.squeeze(canonical_m.get_fdata()).astype(np.float32)
#     g = ColocVolume._to_pv_image(canonical_m, d, key="mask")
#     threshed = g.threshold(0.5)
#     print(f"  Threshold result: n_cells={threshed.n_cells}")
#     surface = threshed.extract_surface()
#     print(f"  Surface n_points: {surface.n_points}")
#     if surface.n_points > 0:
#         p.add_mesh(surface, color="cyan", opacity=0.5)
#         img = p.screenshot(return_img=True)
#         print(f"  ✅ Screenshot shape: {img.shape}, mean pixel: {img.mean():.1f}")
#     else:
#         print("  ❌ Surface vide !")
#     p.close()
# except Exception as e:
#     print(f"  ❌ Erreur: {e}")

# # ── 6. Test ColocVolume complet ───────────────────────────────────────────
# print("\n" + "=" * 60)
# print("6. Test ColocVolume.plot_3d() complet")
# print("=" * 60)

# try:
#     vol = ColocVolume()
#     vol.add_layer(t1_img, role="background", cmap="gray", opacity=0.15, label="T1")
#     vol.add_layer(ef_img, role="stat_map", cmap="hot", opacity=1.0, label="EField")
#     vol.add_layer(mask_img, role="overlay", color="cyan", opacity=0.5, label="ROI")
#     print(f"  {vol}")

#     frame = vol.plot_3d(camera_position="xy")
#     print(f"  Screenshot shape: {frame.shape}")
#     print(
#         f"  Pixel stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}"
#     )

#     if frame.mean() < 1.0:
#         print("  ❌ Image noire — rien n'a été rendu")
#     elif frame.mean() > 254.0:
#         print("  ❌ Image blanche")
#     else:
#         print("  ✅ Rendu semble contenir du contenu")

#     # Sauvegarder pour inspection visuelle
#     import matplotlib

#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.imshow(frame)
#     ax.axis("off")
#     ax.set_title("ColocVolume.plot_3d() — synthetic test")
#     fig.savefig("test_3d_output.png", dpi=100, bbox_inches="tight")
#     print("  Saved: test_3d_output.png")
#     plt.close(fig)

# except Exception as e:
#     print(f"  ❌ Erreur: {e}")
#     import traceback

#     traceback.print_exc()

# print("\n" + "=" * 60)
# print("7. Test avec NaN dans add_volume (source fréquente de problèmes)")
# print("=" * 60)

# try:
#     p = pv.Plotter(off_screen=True)
#     # Volume SANS NaN
#     d_no_nan = np.squeeze(nib.as_closest_canonical(ef_img).get_fdata()).astype(float)
#     g = ColocVolume._to_pv_image(nib.as_closest_canonical(ef_img), d_no_nan, key="v")
#     p.add_volume(g, cmap="hot")
#     img = p.screenshot(return_img=True)
#     print(f"  Sans NaN — mean pixel: {img.mean():.1f}")
#     p.close()

#     p = pv.Plotter(off_screen=True)
#     # Volume AVEC NaN
#     d_nan = d_no_nan.copy()
#     d_nan[d_nan == 0] = np.nan
#     g = ColocVolume._to_pv_image(nib.as_closest_canonical(ef_img), d_nan, key="v")
#     p.add_volume(g, cmap="hot")
#     img = p.screenshot(return_img=True)
#     print(f"  Avec NaN — mean pixel: {img.mean():.1f}")
#     p.close()

#     # Comparer
#     print("  → Si 'Avec NaN' est plus sombre, les NaN masquent peut-être tout")
# except Exception as e:
#     print(f"  ❌ Erreur: {e}")
#     import traceback

#     traceback.print_exc()

# print("\n✅ Diagnostic terminé.")
