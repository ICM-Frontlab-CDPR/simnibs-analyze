# Configuration

Each command takes its own YAML schema. They are **not interchangeable**: passing
a viz config to `simnibs-analyze` (or the reverse) fails validation immediately.

Both are validated with pydantic before any work starts, so a malformed config
raises a precise error rather than crashing mid-run.

| Command | Schema | Example |
|---|---|---|
| `simnibs-analyze` | `PipelineConfig` | `docs/examples/config/config-analyze_htacs.yaml` |
| `simnibs-viz` | `VizConfig` | `docs/examples/config/config-viz_stimSD.yaml` |

---

## `simnibs-analyze` config

### Top-level keys

| Key | Type | Default | Description |
|---|---|---|---|
| `subjects` | list of str | — | Subject IDs, e.g. `["0001", "0002"]` |
| `stim_conditions` | list of str | — | Conditions to process. **Every entry must have a matching key in `target_generation.rois`** |
| `mode` | list | — | Any of `simulation`, `optimization` |
| `space` | str | `mni` | `mni` or `native`. Drives the `space-<...>` output suffix |
| `running` | mapping | see below | Output-collision policy |
| `target_generation` | mapping | — | ROI definitions |
| `paths` | mapping | — | Input and output locations |
| `preprocessing` | mapping | see below | Smoothing and outlier removal |
| `feature_extraction` | mapping | see below | Which metrics to compute |
| `analysis` | mapping | see below | Group statistics and clustering |

### `running`

| Key | Type | Default | Description |
|---|---|---|---|
| `if_exists` | str | `skip` | `skip`, `overwrite`, or `error` when an output file already exists |

### `paths`

Key names are the **same in both schemas**, so a `paths` block can be copied
between an analyse config and a viz config unchanged.

| Key | Type | Description |
|---|---|---|
| `sim_base` | path | Simulation / optimization results — `{sub}/simulations/` |
| `seg_base` | path | charm / segmentation results — `{sub}/m2m_{sub}/` |
| `out_root` | path | **Required.** Where CSVs, figures and statistics are written |

!!! info "`seg_base` is not optional in practice"
    It carries two things the pipeline needs: the tissue map used to compute
    extra-ROI statistics (and therefore the focality ratio and the clustering),
    and the deformation field used to place MNI targets in native space.
    Without it you get intra-ROI stats only, and native-space runs cannot warp.

**Deprecated aliases**, still accepted so existing configs keep working:

| Deprecated | Use instead |
|---|---|
| `simnibs_simu` | `sim_base` |
| `simnibs_preps` | `seg_base` |
| `results_dir` | `out_root` |
| `simnibs_output` | `sim_base` + `seg_base` (single root holding both) |

If both a new name and its alias are given, the new name wins.

!!! warning "Removed in 0.2.0"
    `mni_template` and `mni_brain_mask` are gone. They were accepted by the
    schema and then never read by any code — the brain mask is now resolved
    from the segmentation automatically, in whichever space the analysis runs.

### `target_generation`

| Key | Type | Default | Description |
|---|---|---|---|
| `radius_mm` | float | `10.0` | Sphere radius, must be `> 0`. Used by `method: sphere` |
| `rois` | mapping | — | ROI name → definition |

!!! note "Coordinates are always MNI"
    `coords` are given in MNI millimetres regardless of `space`. When `space:
    native`, they are warped onto the subject grid using the deformation field
    in `seg_base/{sub}/m2m_{sub}/toMNI/`. Atlas parcels are warped the same way.

!!! danger "ROI names cannot contain underscores"
    Use hyphens: `ips-left`, not `ips_left`. Underscores are rejected at
    validation because they collide with the output filename convention. If the
    SimNIBS folder on disk *does* use an underscore, keep the hyphen in the ROI
    name and set `folder_pattern` to the on-disk spelling.

Each ROI is one of two methods, discriminated on the `method` key.

**Sphere** — centred on MNI coordinates:

```yaml
fef:
  method: sphere
  coords: [28, -8, 54]     # exactly 3 floats, MNI mm
ips-left:
  method: sphere
  coords: [-25, -60, 52]
  folder_pattern: ips_left  # folder spelling differs from the ROI key
```

**Atlas** — one or more parcels:

```yaml
HA-fef:
  method: atlas
  atlas: harvard-oxford     # harvard-oxford | aal | destrieux
  regions: "Precentral Gyrus"   # str or list of str
```

| Key | Applies to | Description |
|---|---|---|
| `coords` | sphere | `[x, y, z]` in MNI mm. Exactly three values |
| `atlas` | atlas | `harvard-oxford`, `aal`, or `destrieux` |
| `regions` | atlas | One label or a list of labels |
| `folder_pattern` | both | Glob fragment for finding SimNIBS folders when the name differs from the ROI key |

### `preprocessing`

| Key | Type | Default | Description |
|---|---|---|---|
| `smooth_fwhm` | float | `2.0` | Smoothing FWHM in mm, `>= 0` |
| `outlier_method` | str | `iqr` | `iqr` or `zscore` |
| `portion` | float | `null` | Optional fraction of voxels to keep, in `(0, 1]` |

### `feature_extraction`

| Key | Type | Default | Description |
|---|---|---|---|
| `metrics` | list of str | `[mean, median, std, min, max]` | Statistics computed per ROI |

### `analysis`

| Key | Type | Default | Description |
|---|---|---|---|
| `metric` | str | `mean` | Column used for group statistics |
| `subject_col` | str | `subject` | Subject column name in the features CSV |
| `condition_col` | str | `condition` | Condition column name |
| `clustering.method` | str | `mean` | Ratio-column suffix, reads `efield_ratio_<method>` |
| `clustering.specificity_threshold` | float | `1.5` | Ratio above which a simulation counts as *specific*. Must be `> 0` |
| `clustering.intensity_col` | str | `mean` | Column whose group median splits *high* from *low* |

Clusters are labelled `specific_high`, `specific_low`, `diffuse_high`, `diffuse_low`.

### Minimal example

```yaml
subjects: ["0001", "0002"]
stim_conditions: [fef]
mode: [simulation]
space: native

target_generation:
  radius_mm: 5.0
  rois:
    fef:
      method: sphere
      coords: [28, -8, 54]

paths:
  sim_base: /data/derivatives/simnibs-simu
  seg_base: /data/derivatives/simnibs-preps
  out_root: /data/derivatives/results
```

---

## `simnibs-viz` config

The viz schema is built around a **named registry of reusable layers** (`vols`)
that figure blocks reference by name. This keeps the raw rendering dicts out of
the figure definitions.

!!! note "Unknown keys are rejected"
    Every viz model sets `extra="forbid"`. A typo in a key name raises a
    validation error instead of being silently ignored.

### Top-level keys

| Key | Type | Description |
|---|---|---|
| `subjects` | mapping | Cohort selection |
| `paths` | mapping | Input and output roots |
| `fields_scale` | mapping | Colour scale shared across the whole cohort |
| `vols` | mapping | Named layer registry |
| `figures` | list | Figure blocks |

### `subjects`

| Key | Type | Description |
|---|---|---|
| `ids` | list of str | Subject IDs |
| `stim_pattern` | str | Substring matched against `simulation_*` folder names |

### `paths`

| Key | Type | Description |
|---|---|---|
| `sim_base` | path | Simulation results root |
| `seg_base` | path | Segmentation root, containing `{sub}/m2m_{sub}/` |
| `out_root` | path | Where figures are written |

### `fields_scale`

| Key | Type | Description |
|---|---|---|
| `min` | float | Lower bound in V/m. Must be `<` `max` |
| `max` | float | Upper bound in V/m |

Locked manually for the whole cohort so panels stay comparable across subjects.

### `vols` — the layer registry

Each entry is `name: {kind: ...}`, discriminated on `kind`. All layers accept
`colormap` (default `gray`) and `opacity` (`0.0`–`1.0`).

**`kind: anat`** — structural background:

| Key | Default | Description |
|---|---|---|
| `source` | `t1` | `t1`, `brain_mask`, `label_prep`, `lesion_native`, `lesion_mni` |
| `render` | `fill` | `fill` or `contour` |

**`kind: roi`** — exactly **one** source among `atlas`, `coords`, `file`:

| Key | Default | Description |
|---|---|---|
| `atlas` + `regions` | — | Atlas name plus a list of labels. `regions` is mandatory with `atlas` |
| `coords` | — | `[x, y, z]` MNI mm, converted to native at runtime |
| `radius` | `10.0` | Sphere radius when using `coords` |
| `file` | — | Path to a mask file |
| `colormap` | `blue` | |
| `render` | `fill` | `fill` or `contour` |

**`kind: field`** — continuous e-field or current density:

| Key | Default | Description |
|---|---|---|
| `name` | `magnE` | `E`, `J`, `magnE`, `magnJ` |
| `colormap` | `hot` | |
| `opacity` | `0.6` | |

```yaml
vols:
  anat_brain:
    kind: anat
    source: brain_mask
    opacity: 0.5
  roi_fef:
    kind: roi
    coords: [28, -8, 54]
    radius: 10
  field_E:
    kind: field
    name: magnE
```

### `figures`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | — | Figure name, used in output filenames |
| `type` | str | — | `2D` or `3D` |
| `vols` | list of str | — | Layer names from the registry. **Must not be empty, and every name must exist** |
| `cohort` | bool | `false` | Also include this figure in a cohort montage |
| `if_exists` | str | `overwrite` | `overwrite`, `skip`, `error` |

**2D only:**

| Key | Default | Description |
|---|---|---|
| `subtype` | `ortho` | `ortho` (single crosshair view) or `parallel` (slice series) |
| `cut_coords` | `null` | `[x, y, z]` MNI mm. **Mutually exclusive with `cut_center_vol`** |
| `cut_center_vol` | `null` | Layer name whose centre of mass becomes the cut centre. Takes priority |
| `axis` | `z` | `x`, `y` or `z`. `parallel` only |
| `n_cuts` | `7` | Number of slices when `half_width` is not given |
| `half_width` | `null` | Half-extent in mm around the centre. **Requires `spacing`** |
| `spacing` | `null` | Millimetres between slices |
| `contour_vols` | `[]` | Layer names to render as contour in this figure, overriding their `render` |
| `montage_ncols` | `null` | Columns in the cohort grid. `null` = auto |
| `montage_panel_h` | `4.0` | Panel height in inches; width follows the aspect ratio |

**3D only:**

| Key | Default | Description |
|---|---|---|
| `camera` | `[225.0, 15.0]` | `[azimuth, elevation]`. Exactly two values |
| `electrodes_cap` | `null` | EEG cap name without `.csv`. `null` = no electrodes |

### Minimal example

```yaml
subjects:
  ids: ["0001", "0003"]
  stim_pattern: "AFFT"

paths:
  sim_base: /data/2-simnibs-simu-left
  seg_base: /data/1-simnibs-preps
  out_root: /data/figures

fields_scale:
  min: 0.05
  max: 0.4

vols:
  anat_brain:
    kind: anat
    source: brain_mask
  field_E:
    kind: field
    name: magnE

figures:
  - name: efield_ortho
    type: 2D
    subtype: ortho
    vols: [anat_brain, field_E]
    cohort: true
```

---

## Validating a config without running

The analysis schema can be checked standalone:

```bash
python -m simnibs_analyze._config_schema_analyze --config config-analyze.yaml
```

It prints a one-line summary on success, or the validation error and exit code
`1` on failure.
