"""
Creates a histogram showing the percentag of pixels with a certain error bound.
This evaluates the results of mainAdaptiveIsoStats.py
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from statsLoader import StatsConfig, StatsLoader, HistoField

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

    HEATMAP_CONFIG = (0.002, 0.05)

    FIELDS = [
        (HistoField.L1_ERROR_MASK, "mask"), 
        (HistoField.L1_ERROR_NORMAL, "normal"), 
        (HistoField.L1_ERROR_DEPTH, "depth"), 
        (HistoField.L1_ERROR_AO, "ao"), 
        (HistoField.L1_ERROR_COLOR_NOAO, "color (no AO)"), 
        (HistoField.L1_ERROR_COLOR_WITHAO, "color (with AO)"), 
        ]
    HISTO_TITLE = "Percentage of pixels (y-axis) with $|x^{est}-x^{GT}|$ smaller than the specified error bound (x-axis)\n(a higher value is better)"
    PATTERN = "plastic"

    fig, ax = plt.subplots(
            nrows=len(SETS), ncols=len(FIELDS), 
            sharey=False, sharex=True,
            squeeze = False)

    linearFormatter = FuncFormatter(lambda y, _: '{:.2}'.format(y))
    percentageFormatter = FuncFormatter(lambda y, _: '{:.2%}'.format(y))

    for a,(name, loader) in enumerate(SETS):
        assert isinstance(loader, StatsLoader)
        minHeatmapMin = min(loader.heatmapMin())
        heatmapMeans = loader.heatmapMean()

        # get model combinations (importance + reconstruction networks)
        models = loader.allNetworks()
        modelNames = ["%s - %s"%model for model in models]
        # draw plots
        for b,(field_id, field_name) in enumerate(FIELDS):
            handles = []
            for model_idx, (importance, reconstruction) in enumerate(models):
                cfg = StatsConfig(importance, reconstruction, HEATMAP_CONFIG[0], HEATMAP_CONFIG[1], PATTERN)
                histo = loader.getHistogram(cfg)
                data = np.array([histo[i][field_id] for i in range(histo.shape[0])])
                centers = np.array([0.5*(histo[i][HistoField.BIN_END]+histo[i][HistoField.BIN_START]) for i in range(histo.shape[0])])
                heights = np.cumsum(data)
                #if field_id=="L1ErrorDepth":
                #    heights = (heights-0.8)*5
                if len(heights)>10:
                    print(modelNames[model_idx],field_name,": percantage of pixels with less than 10% error: {:.2%}".format(1-heights[10]))
                
                if len(heights)>0:
                    handles.append(ax[a,b].semilogx(centers, heights, label=modelNames[model_idx]))
            ax[a,b].xaxis.set_major_formatter(linearFormatter)
            ax[a,b].yaxis.set_major_formatter(percentageFormatter)
            if a==0:
                ax[a,b].set_title(field_name)
            if b==0:
                ax[a,b].set_ylabel(name)

    fig.suptitle(HISTO_TITLE)
    #fig.legend(handles, 
    #           labels=modelNames,
    #           loc="lower center",
    #           borderaxespad=0.1,
    #           ncol=max(5,len(models)))
    fig.legend(handles,labels=modelNames,loc='lower center', ncol=len(modelNames))
    fig.subplots_adjust(
        left=0.04, bottom=0.06, right=0.99, top=0.91,
        wspace=0.22, hspace=0.09)
    plt.show()