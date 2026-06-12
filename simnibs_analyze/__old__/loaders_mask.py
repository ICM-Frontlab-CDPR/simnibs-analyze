from pathlib import Path

import nibabel as nib

### TODO A DEPLACER DANS LE READER !!

# ── Dataset loaders ────────────────────────────────────────────────────────────


class SynthStrokeDataset:
    """
    Reads SynthStroke outputs from each subject folder.

    space="native" : *_lesion.nii.gz  + *_reslice.nii.gz
    space="mni"    : *_lesion_mni.nii.gz + *_reslice_mni.nii.gz
    """

    def __init__(self, root: Path, space: str = "native"):
        self.root = root
        self.space = space
        self.subjects = sorted(d.name for d in root.iterdir() if d.is_dir())

    def get(self, subject: str) -> dict | None:
        # Ajoute automatiquement le préfixe 'sub-' si le dossier existe sous cette forme
        subj_dir = self.root / subject
        if not subj_dir.exists():
            alt_dir = self.root / f"sub-{subject}"
            if alt_dir.exists():
                subj_dir = alt_dir
        if self.space == "mni":
            masks = sorted(subj_dir.glob("*_lesion_mni.nii.gz"))
            t1s = sorted(subj_dir.glob("*_reslice_mni.nii.gz"))
        else:
            # native: exclude *_mni files to avoid ambiguity
            masks = [
                p
                for p in sorted(subj_dir.glob("*_lesion.nii.gz"))
                if "_mni" not in p.name
            ]
            t1s = [
                p
                for p in sorted(subj_dir.glob("*_reslice.nii.gz"))
                if "_mni" not in p.name
            ]
        if not masks or not t1s:
            return None
        return {
            "subject": subject,
            "mask_ss": nib.load(masks[0]),
            "t1_ss": nib.load(t1s[0]),
        }


class OriginalDataset:
    """
    Reads original lesion masks from the clean folder.

    space="native" : <subject>/Espace_natif/lesion.nii[.gz]
    space="mni"    : <subject>/Espace_MNI/lesion_mni.nii.gz
    """

    def __init__(self, root: Path, space: str = "native"):
        self.root = root
        self.space = space
        self.subjects = sorted(d.name for d in root.iterdir() if d.is_dir())

    def get(self, subject: str) -> dict | None:
        if self.space == "mni":
            mask_path = self.root / subject / "Lesion_normalisee" / "lesion.nii.gz"
            if not mask_path.exists():
                # try uncompressed
                mask_path = self.root / subject / "Lesion_normalisee" / "lesion.nii"
            if not mask_path.exists():
                return None
            return {
                "subject": subject,
                "mask_orig": nib.load(mask_path),
            }
        else:
            native_dir = self.root / subject / "Espace_natif"
            if not native_dir.exists():
                return None
            candidates = sorted(native_dir.glob("lesion.nii*"))
            if not candidates:
                candidates = sorted(native_dir.glob("*lesion*.nii*"))
            if not candidates:
                return None
            return {
                "subject": subject,
                "mask_orig": nib.load(candidates[0]),
            }
