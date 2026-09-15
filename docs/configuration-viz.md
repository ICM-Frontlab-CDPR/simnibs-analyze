# `simnibs-viz` config

Validated against `VizConfig` before any work starts. Built around a **named
registry of reusable layers** (`vols`) that figure blocks reference by name,
keeping rendering details out of the figure definitions.

[:material-download: config_viz_example.yaml](examples/config/config_viz_example.yaml){ .md-button .md-button--primary download="config_viz_example.yaml" }

!!! note "Unknown keys are rejected"
    Every viz model sets `extra="forbid"`, so a typo raises a validation error
    instead of being silently ignored.

## Top-level keys

| Key | Type | Description |
|---|---|---|
| `subjects` | mapping | Cohort selection |
| `paths` | mapping | Input and output roots |
| `running` | mapping | Execution policy |
| `fields_scale` | mapping | Colour scale shared across the cohort |
| `vols` | mapping | Named layer registry |
| `figures` | list | Figure blocks |

## `subjects`

| Key | Type | Description |
|---|---|---|
| `ids` | list of str | Subject IDs |
| `stim_pattern` | str | Substring matched against `simulation_*` folder names |

## `paths`

Key names are shared with the [analyse config](configuration-analyze.md), so a
`paths` block can be copied between the two.

| Key | Type | Description |
|---|---|---|
| `sim_base` | path | Simulation results root |
| `seg_base` | path | Segmentation root, containing `{sub}/m2m_{sub}/` |
| `out_root` | path | Where figures are written |

## `running`

| Key | Type | Default | Description |
|---|---|---|---|
| `if_exists` | str | `skip` | `skip`, `overwrite`, or `error` |

An existing figure is skipped **before** any volume is loaded or warped, so
re-running a cohort to add one figure costs almost nothing.

## `fields_scale`

| Key | Type | Description |
|---|---|---|
| `min` | float | Lower bound in V/m, must be `<` `max` |
| `max` | float | Upper bound in V/m |

Locked manually so panels stay comparable across subjects.

## `vols` — the layer registry

Each entry is `name: {kind: ...}`, discriminated on `kind`. All layers accept
`colormap` and `opacity` (`0.0`–`1.0`).

**`kind: anat`** — structural background:

| Key | Default | Description |
|---|---|---|
| `source` | `t1` | `t1`, `brain_mask`, `label_prep`, `lesion_native`, `lesion_mni` |
| `render` | `fill` | `fill` or `contour` |

**`kind: roi`** — exactly **one** source among `atlas`, `coords`, `file`:

| Key | Default | Description |
|---|---|---|
| `atlas` + `regions` | — | Atlas name plus labels. `regions` is mandatory with `atlas` |
| `coords` | — | `[x, y, z]` MNI mm, warped to native at runtime |
| `radius` | `10.0` | Sphere radius when using `coords` |
| `file` | — | Path to a mask file |
| `colormap` | `blue` | |
| `render` | `fill` | `fill` or `contour` |

**`kind: field`** — e-field or current density:

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
  roi_target:
    kind: roi
    coords: [28, -8, 54]
    radius: 10
  efield:
    kind: field
    name: magnE
```

## `figures`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | — | Used in output filenames |
| `type` | str | — | `2D` or `3D` |
| `vols` | list of str | — | Layer names from the registry. Must not be empty, and every name must exist |
| `cohort` | bool | `false` | Also include this figure in a cohort montage |
| `if_exists` | str | inherits `running.if_exists` | Override for this figure only |

**2D only:**

| Key | Default | Description |
|---|---|---|
| `subtype` | `ortho` | `ortho` (crosshair view) or `parallel` (slice series) |
| `cut_coords` | `null` | `[x, y, z]` MNI mm. Mutually exclusive with `cut_center_vol` |
| `cut_center_vol` | `null` | Layer whose centre of mass becomes the cut centre. Takes priority |
| `axis` | `z` | `x`, `y` or `z`. `parallel` only |
| `n_cuts` | `7` | Number of slices when `half_width` is not given |
| `half_width` | `null` | Half-extent in mm around the centre. Requires `spacing` |
| `spacing` | `null` | Millimetres between slices |
| `contour_vols` | `[]` | Layers to render as contour in this figure, overriding their `render` |
| `montage_ncols` | `null` | Columns in the cohort grid. `null` = auto |
| `montage_panel_h` | `4.0` | Panel height in inches |

**3D only:**

| Key | Default | Description |
|---|---|---|
| `camera` | `[225.0, 15.0]` | `[azimuth, elevation]`, exactly two values |
| `electrodes_cap` | `null` | EEG cap name without `.csv`. `null` = no electrodes |

!!! warning "3D figures need Playwright"
    Rendered by driving a headless browser over a NiiVue WebGL scene:

    ```bash
    pip install "simnibs-analyze[viz3d]"
    playwright install chromium
    ```

    The second step downloads the browser and is easy to forget. 2D figures
    have no such requirement.

## Minimal example

```yaml
subjects:
  ids: ["0001", "0003"]
  stim_pattern: "AFFT"

paths:
  sim_base: /data/derivatives/simnibs-simu
  seg_base: /data/derivatives/simnibs-preps
  out_root: /data/figures

fields_scale:
  min: 0.05
  max: 0.4

vols:
  anat_brain:
    kind: anat
    source: brain_mask
  efield:
    kind: field
    name: magnE

figures:
  - name: efield_ortho
    type: "2D"
    subtype: ortho
    vols: [anat_brain, efield]
    cohort: true
```
