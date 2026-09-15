# `simnibs-analyze` config

Validated against `PipelineConfig` before any work starts, so a malformed file
fails immediately with a precise message.

[:material-download: config_analyze_example.yaml](examples/config/config_analyze_example.yaml){ .md-button .md-button--primary download="config_analyze_example.yaml" }

## Top-level keys

| Key | Type | Default | Description |
|---|---|---|---|
| `subjects` | list of str | — | Subject IDs |
| `stim_conditions` | list of str | — | Conditions to process. Every entry needs a matching key in `target_generation.rois` |
| `mode` | list | — | Any of `simulation`, `optimization` |
| `space` | str | `mni` | `mni` or `native`. Drives the `space-<...>` output suffix |
| `running` | mapping | see below | Output-collision policy |
| `paths` | mapping | — | Input and output locations |
| `target_generation` | mapping | — | ROI definitions |
| `preprocessing` | mapping | see below | Smoothing and outlier removal |
| `feature_extraction` | mapping | see below | Which metrics to compute |
| `analysis` | mapping | see below | Group statistics and clustering |

## `running`

| Key | Type | Default | Description |
|---|---|---|---|
| `if_exists` | str | `skip` | `skip`, `overwrite`, or `error` |

## `paths`

Key names are shared with the [viz config](configuration-viz.md), so a `paths`
block can be copied between the two.

| Key | Type | Description |
|---|---|---|
| `sim_base` | path | Simulation / optimization results — `{sub}/simulations/` |
| `seg_base` | path | Segmentation results — `{sub}/m2m_{sub}/` |
| `out_root` | path | **Required.** Where CSVs and statistics are written |

Without `seg_base` you get intra-ROI stats only: no focality ratio, no
clustering, and native-space runs cannot warp MNI targets.

## `target_generation`

| Key | Type | Default | Description |
|---|---|---|---|
| `radius_mm` | float | `10.0` | Sphere radius, must be `> 0` |
| `rois` | mapping | — | ROI name → definition |

!!! note "Coordinates are always MNI"
    `coords` are MNI millimetres regardless of `space`. In `native`, they are
    warped onto the subject grid using the field in
    `seg_base/{sub}/m2m_{sub}/toMNI/`. Atlas parcels are warped the same way.

!!! danger "ROI names cannot contain underscores"
    Use hyphens — `ips-left`, not `ips_left` — they collide with the
    `space-<value>` filename convention. SimNIBS folders on disk often use
    underscores anyway; both spellings are tried automatically.

Each ROI is one of two methods, discriminated on `method`:

```yaml
rois:
  AFFT:
    method: sphere
    coords: [28, -8, 54]        # exactly 3 floats, MNI mm

  HA-fef:
    method: atlas
    atlas: harvard-oxford       # harvard-oxford | aal | destrieux
    regions: "Precentral Gyrus" # str or list of str
```

| Key | Applies to | Description |
|---|---|---|
| `coords` | sphere | `[x, y, z]` in MNI mm |
| `atlas` | atlas | `harvard-oxford`, `aal`, or `destrieux` |
| `regions` | atlas | One label or a list of labels |

## `preprocessing`

| Key | Type | Default | Description |
|---|---|---|---|
| `smooth_fwhm` | float | `2.0` | Smoothing FWHM in mm, `>= 0` |
| `outlier_method` | str | `iqr` | `iqr` or `zscore` |
| `portion` | float | `null` | Fraction of voxels to keep, in `(0, 1]` |

## `feature_extraction`

| Key | Type | Default | Description |
|---|---|---|---|
| `metrics` | list of str | `[mean, median, std, min, max]` | Statistics computed per ROI |

## `analysis`

| Key | Type | Default | Description |
|---|---|---|---|
| `metric` | str | `mean` | Column used for group statistics |
| `subject_col` | str | `subject` | Subject column in the features CSV |
| `condition_col` | str | `condition` | Condition column |
| `clustering.method` | str | `mean` | Reads `efield_ratio_<method>` |
| `clustering.specificity_threshold` | float | `1.5` | Ratio above which a simulation is *specific*, `> 0` |
| `clustering.intensity_col` | str | `mean` | Column whose group median splits *high* from *low* |

Clusters are labelled `specific_high`, `specific_low`, `diffuse_high`,
`diffuse_low`.

## Minimal example

```yaml
subjects: ["0001", "0002"]
stim_conditions: [AFFT]
mode: [simulation]
space: native

target_generation:
  radius_mm: 10.0
  rois:
    AFFT:
      method: sphere
      coords: [28, -8, 54]

paths:
  sim_base: /data/derivatives/simnibs-simu
  seg_base: /data/derivatives/simnibs-preps
  out_root: /data/derivatives/results
```

## Checking a config

```bash
python -m simnibs_analyze._config_schema_analyze --config my-config.yaml
```
