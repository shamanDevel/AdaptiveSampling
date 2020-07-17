"""
Creates error plots for different network architectures.
This evaluates the results of mainAdaptiveIsoStats.py
"""

import numpy as np
import matplotlib.pyplot as plt

from statsLoader import StatsConfig, StatsLoader, StatField

if __name__ == "__main__":
    """
    Compare network architectures (SSIM, PSNR, REC-curve),
    averaged over heatmin, heatmean, pattern.
    Future: compare different test sets
    """
    SETS = [
        ('Ejecta', StatsLoader("../result-stats/adaptiveIsoEnhance6/Ejecta.hdf5")),
        #('RM', StatsLoader("../result-stats/adaptiveIsoEnhance4Big/RM")),
        #('Thorax', StatsLoader("../result-stats/adaptiveIso/Thorax")),
        #('Human', StatsLoader("../result-stats/adaptiveIso/Human")),
        ]

    FIELDS = [
        ("PSNR", [
        (StatField.PSNR_MASK, "mask"), (StatField.PSNR_NORMAL, "normal"), (StatField.PSNR_AO, "AO"),
        (StatField.PSNR_COLOR_NOAO, "color (no AO)"), (StatField.PSNR_COLOR_WITHAO, "color (with AO)")
        ], "col"),
        ("SSIM", [
        (StatField.SSIM_MASK, "mask"), (StatField.SSIM_NORMAL, "normal"), (StatField.SSIM_AO, "AO"),
        (StatField.SSIM_COLOR_NOAO, "color (no AO)"), (StatField.SSIM_COLOR_WITHAO, "color (with AO)")
        ], True),
        ("LPIPS (lower is better)", [
        (StatField.LPIPS_COLOR_NO_AO, "color (no AO)"), (StatField.LPIPS_COLOR_WITHAO, "color (with AO)")
        ], "col"),
        ]
    #HEATMAP_CONFIGS = None
    HEATMAP_CONFIGS = [(0.002, 0.05)]

    # Error Plots
    for (title, FIELD, sharex) in FIELDS:

        fig, ax = plt.subplots(
            nrows=len(SETS), ncols=len(FIELD), 
            sharey='row', sharex=sharex,
            squeeze = False)
        # loop over datasets
        for a,(name, loader) in enumerate(SETS):
            assert isinstance(loader, StatsLoader)
            # get model combinations (importance + reconstruction networks)
            models = loader.allNetworks()
            modelNames = ["%s - %s"%model for model in models]
            def modelFilter(model, column : StatField): # inpainting has no AO
                if model[1] == "inpainting":
                    if column.name.find("WITHAO")>=0:
                        return False
                return True
            # average data over smallest heatmap cfg and all pattern
            # for each model
            heatmap_configs = HEATMAP_CONFIGS or loader.allHeatmapConfigs()
            data = []
            for importance, reconstruction in models:
                d = [loader.getStatistics(StatsConfig(
                    importance, reconstruction, heatmin, heatmean, pattern))
                     for (heatmin, heatmean) in heatmap_configs
                     for pattern in loader.samplingPattern()]
                data.append(np.concatenate(d, axis=0))
            print(data[0].dtype)
            l = min([len(data[i]) for i in range(len(data))])
            print(l)

            model_indices = []
            for model_idx in range(len(models)):
                has_entry = False
                for (field_id, field_name) in FIELD:
                    if modelFilter(models[model_idx], field_id):
                        has_entry = True
                if has_entry:
                    model_indices.append(model_idx)

            for b,(field_id, field_name) in enumerate(FIELD):
                #ax.set_title(field)
                field_data = [ ((np.array([data[model_idx][i][field_id] for i in range(l) \
                                         if (not np.isnan(data[model_idx][i][field_id]) and data[model_idx][i][field_id]>0)])) \
                                   if modelFilter(models[model_idx], field_id) else []) \
                               for model_idx in model_indices]
                medians = [np.median(d) for d in field_data]
                ax[a,b].boxplot(field_data, vert=False, 
                                labels=[modelNames[model_idx] for model_idx in model_indices], 
                                showfliers=False)

                for tick in range(len(model_indices)):
                    ax[a,b].text(medians[tick], tick+1.26, str(np.round(medians[tick], 4)), horizontalalignment='center')

                if a==0:
                    ax[a,b].set_title(field_name)
                if b==0:
                    ax[a,b].set_ylabel(name)

        fig.suptitle(title + " - heatmap cfg: " + str(heatmap_configs))
        fig.show()

        fig.subplots_adjust(
            bottom = 0.025, top = 0.94, right = 0.99, left = 0.2, 
            wspace = 0.04, hspace=0.06)

plt.show()