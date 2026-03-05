    
    
    
class Visualizer:    
    
    
    def efield_comparison(
        self,
        stim_efield_path: Path,
        opti_efield_path: Path,
        subject_id: str,
        roi_name: str,
        slice_idx: Optional[int] = None,
        output_path: Optional[Path] = None
    ) -> Figure:
        """
        Figure de comparaison des e-fields stim vs opti (NIVEAU SUJET).
        
        Parameters
        ----------
        stim_efield_path : Path
            Chemin vers l'e-field de stimulation
        opti_efield_path : Path
            Chemin vers l'e-field d'optimisation
        subject_id : str
            ID du sujet
        roi_name : str
            Nom de la ROI
        slice_idx : int, optional
            Index de la coupe axiale
        output_path : Path, optional
            Chemin de sauvegarde
            
        Returns
        -------
        fig : Figure
            Figure matplotlib
        """
        # Charger les e-fields
        stim_data, _ = load_nifti(stim_efield_path)
        opti_data, _ = load_nifti(opti_efield_path)
        
        # Déterminer le vmax commun
        vmax = max(np.nanmax(stim_data), np.nanmax(opti_data))
        
        # Extraire les coupes
        stim_slice = extract_slice(stim_data, axis=2, slice_idx=slice_idx)
        opti_slice = extract_slice(opti_data, axis=2, slice_idx=slice_idx)
        diff_slice = opti_slice - stim_slice
        
        # Créer la figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plots
        plot_efield_slice(axes[0], stim_slice, title='Stimulation', vmax=vmax)
        plot_efield_slice(axes[1], opti_slice, title='Optimization', vmax=vmax)
        plot_efield_difference(axes[2], diff_slice, title='Difference (Opti - Stim)')
        
        fig.suptitle(f'E-field Comparison - {subject_id} - {roi_name}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()