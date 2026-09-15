# Outputs

What each command writes to `paths.out_root`.

## `simnibs-analyze`

```
out_root/
├── all_features_space-<space>.csv        ← the extraction table
├── inter_subject_space-<space>.csv       ← group summary
└── intra_subject_<cond>_space-<space>.csv  (only when both modes ran)
```

The `space-<space>` suffix is `space-mni` or `space-native`, so running both
spaces against the same `out_root` never overwrites anything.

### `all_features_space-<space>.csv`

**One row per (subject × condition × mode).** This is the file to open first,
and the only one you need to reproduce every downstream number.

| Column | Description |
|---|---|
| `subject` | Subject ID |
| `condition` | `<roi-name>_<mode>`, e.g. `AFFT_simulation` |
| `space` | `mni` or `native` |
| `efield_path` | Absolute path to the e-field volume actually read |
| `roi_path` | Reserved — the ROI mask is not persisted |
| `mean`, `median`, `std`, `min`, `max` | Intra-ROI statistics, one column per entry in `feature_extraction.metrics` |
| `n_voxels` | Voxels inside the ROI after post-processing |
| `extra_*` | The same statistics **outside** the ROI but inside the brain |
| `extra_n_voxels` | Voxels in that complement |
| `efield_ratio_mean` | Focality: intra mean ÷ extra mean |
| `cluster` | `specific_high`, `specific_low`, `diffuse_high` or `diffuse_low` |

!!! info "The `extra_*`, ratio and cluster columns need `seg_base`"
    Computing statistics outside the ROI requires a brain mask, which comes
    from the segmentation. Without `seg_base` you get the intra-ROI columns
    only, and clustering is skipped — the log says
    `'efield_ratio_mean' absent — clustering skipped`.

### `inter_subject_space-<space>.csv`

**One row per condition**, aggregating across subjects.

| Column | Description |
|---|---|
| `condition` | Condition name |
| `mean`, `std`, `count` | Across-subject statistics of `analysis.metric` |
| `sem` | Standard error, `std / sqrt(count)` |

### `intra_subject_<cond>_space-<space>.csv`

Written **only** when both `simulation` and `optimization` ran for that
condition, since it contrasts the two. One row per subject.

---

## `simnibs-viz`

```
out_root/
├── <subject>_<simulation>/
│   ├── <figure-name>.png
│   └── roi_<name>_native.nii.gz     (when a ROI layer is warped)
└── _cohort/
    └── <figure-name>.png
```

| Path | Contents |
|---|---|
| `<subject>_<simulation>/` | One directory per subject and montage, holding one PNG per figure block |
| `<figure-name>.png` | Named after the `name` key of the figure block |
| `roi_<name>_native.nii.gz` | ROI mask warped to subject space, written when a `kind: roi` layer is resolved in native space — reusable as a `file:` layer later |
| `_cohort/` | One montage per figure block flagged `cohort: true`, all subjects on the shared `fields_scale` |

Existing figures are skipped according to
[`running.if_exists`](configuration-viz.md#running), checked before any volume
is loaded, so adding one figure to a finished cohort is nearly free.
