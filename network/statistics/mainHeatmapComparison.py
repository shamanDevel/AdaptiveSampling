"""
Creates error plots for different heatmap values.
This evaluates the results of mainAdaptiveIsoStats.py
"""

import numpy as np
import matplotlib.pyplot as plt

from statsLoader import StatsConfig, StatsLoader, StatField

if __name__ == "__main__":
    """
    Compare convergence with increasing heatmap mean.
    """

    SETS = [
        ('Ejecta', StatsLoader("../result-stats/adaptiveIsoEnhance4Big/Ejecta.hdf5")),
        #('RM', StatsLoader("../result-stats/adaptiveIso/RM")),
        #('Thorax', StatsLoader("../result-stats/adaptiveIso/Thorax")),
        #('Human', StatsLoader("../result-stats/adaptiveIso/Human")),
        ]

    FIELDS = [
        (StatField.SSIM_MASK, "DSSIM mask", lambda x: (1-x)/2), 
        (StatField.SSIM_NORMAL, "DSSIM normal", lambda x: (1-x)/2), 
        (StatField.SSIM_COLOR_NOAO, "DSSIM color (no AO)", lambda x: (1-x)/2),
        (StatField.LPIPS_COLOR_NO_AO, "LPIPS color (no AO)", lambda x: x)
        ]
    TITLE = "Mean stats - different heatmap mean / number of samples"
    PATTERN = "plastic"

    fig, ax = plt.subplots(
            nrows=len(SETS), ncols=len(FIELDS), 
            sharey='row', sharex=True,
            squeeze = False)
    for a,(name, loader) in enumerate(SETS):
        assert isinstance(loader, StatsLoader)
        minHeatmapMin = min(loader.heatmapMin())
        heatmapMeans = loader.heatmapMean()

        # get model combinations (importance + reconstruction networks)
        models = loader.allNetworks()
        modelNames = ["%s - %s"%model for model in models]
        # draw plots
        for b,(field_id, field_name, transform) in enumerate(FIELDS):
            field_data = []
            handles = []
            for model_idx, (importance, reconstruction) in enumerate(models):
                results = []
                for mean in heatmapMeans:
                    cfg = StatsConfig(importance, reconstruction, minHeatmapMin, mean, PATTERN)
                    data = loader.getStatistics(cfg)
                    l = len(data)
                    field_data = np.array([data[i][field_id] for i in range(l)])
                    results.append(transform(np.mean(field_data)))
                print(field_name, ";", modelNames[model_idx], "->", results)
                handles.append(ax[a,b].plot(heatmapMeans, results))
            if a==0:
                ax[a,b].set_title(field_name)
            if b==0:
                ax[a,b].set_ylabel(name)

                fig.legend(handles,labels=modelNames,loc='center left')
                handles = []

    fig.suptitle(TITLE)
    #fig.legend(handles, 
    #           labels=modelNames,
    #           loc="lower center",
    #           borderaxespad=0.1,
    #           ncol=max(5,len(models)))
    fig.subplots_adjust(
        left=0.03, bottom=0.05, right=0.99, top=0.93,
        wspace=0.03, hspace=0.09)
    plt.show()