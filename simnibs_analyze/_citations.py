"""
Dependency citations for simnibs-analyze.

Called once at pipeline startup (run.py) to print BibTeX references
for all tools used, so users can cite them properly.

Uses duecredit when available; falls back to a plain-text summary.
"""

from __future__ import annotations

_CITATIONS = [
    {
        "tool": "SimNIBS",
        "doi": "10.1016/j.neuroimage.2019.116183",
        "ref": (
            "Thielscher A, Antunes A, Saturnino GB (2015). "
            "Field modeling for transcranial magnetic stimulation: a useful tool "
            "to understand the physiological effects of TMS? "
            "IEEE EMBC 2015. doi:10.1109/EMBC.2015.7318340"
        ),
    },
    {
        "tool": "ANTsPy",
        "doi": "10.5281/zenodo.2629946",
        "ref": (
            "Avants BB et al. (2009). "
            "A reproducible evaluation of ANTs similarity metric performance in "
            "brain image registration. NeuroImage 54(3):2033-2044. "
            "doi:10.1016/j.neuroimage.2010.09.025"
        ),
    },
    {
        "tool": "nilearn",
        "doi": "10.3389/fninf.2014.00014",
        "ref": (
            "Abraham A et al. (2014). "
            "Machine learning for neuroimaging with scikit-learn. "
            "Frontiers in Neuroinformatics 8:14. doi:10.3389/fninf.2014.00014"
        ),
    },
    {
        "tool": "nibabel",
        "doi": "10.5281/zenodo.3269256",
        "ref": (
            "Brett M et al. (2024). nipy/nibabel. Zenodo. " "doi:10.5281/zenodo.3269256"
        ),
    },
    {
        "tool": "NumPy",
        "doi": "10.1038/s41586-020-2649-2",
        "ref": (
            "Harris CR et al. (2020). "
            "Array programming with NumPy. Nature 585:357-362. "
            "doi:10.1038/s41586-020-2649-2"
        ),
    },
    {
        "tool": "pandas",
        "doi": "10.25080/Majora-92bf1922-00a",
        "ref": (
            "McKinney W (2010). "
            "Data Structures for Statistical Computing in Python. "
            "Proceedings of the 9th Python in Science Conference, 51-56. "
            "doi:10.25080/Majora-92bf1922-00a"
        ),
    },
    {
        "tool": "matplotlib",
        "doi": "10.1109/MCSE.2007.55",
        "ref": (
            "Hunter JD (2007). "
            "Matplotlib: A 2D graphics environment. "
            "Computing in Science & Engineering 9(3):90-95. "
            "doi:10.1109/MCSE.2007.55"
        ),
    },
    {
        "tool": "PyVista",
        "doi": "10.21105/joss.01450",
        "ref": (
            "Sullivan CB, Kaszynski A (2019). "
            "PyVista: 3D plotting and mesh analysis through a streamlined "
            "interface for the Visualization Toolkit (VTK). "
            "Journal of Open Source Software 4(37):1450. doi:10.21105/joss.01450"
        ),
    },
]


def print_citations() -> None:
    """Print dependency citation notices to stdout."""
    # Try duecredit first
    try:
        from duecredit import due, Doi, BibTeX  # noqa: F401

        for entry in _CITATIONS:
            due.cite(
                Doi(entry["doi"]),
                description=entry["tool"],
                path="simnibs_analyze",
                cite_module=True,
            )
        # duecredit handles printing at process exit automatically
        return
    except ImportError:
        pass

    # Fallback: plain text
    separator = "-" * 60
    print(separator)
    print("simnibs-analyze uses the following tools — please cite them:")
    print(separator)
    for entry in _CITATIONS:
        print(f"\n[{entry['tool']}]\n  {entry['ref']}")
    print(f"\n{separator}")
    print("Install duecredit for BibTeX output:  pip install duecredit")
    print(separator)
