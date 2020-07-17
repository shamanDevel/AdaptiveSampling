import os
import numpy as np
import sys
import enum
import h5py

class StatField(enum.IntEnum):
    PSNR_MASK = 0
    PSNR_NORMAL = 1
    PSNR_DEPTH = 2
    PSNR_AO = 3
    PSNR_COLOR_NOAO = 4
    PSNR_COLOR_WITHAO = 5
    SSIM_MASK = 6
    SSIM_NORMAL = 7
    SSIM_DEPTH = 8
    SSIM_AO = 9
    SSIM_COLOR_NOAO = 10
    SSIM_COLOR_WITHAO = 11
    LPIPS_COLOR_NO_AO = 12
    LPIPS_COLOR_WITHAO = 13
    SAMPLES = 14

class StatFieldDvr(enum.IntEnum):
    PSNR_COLOR = 0
    PSNR_ALPHA = 1
    SSIM_COLOR = 2
    SSIM_ALPHA = 3
    LPIPS_COLOR = 4
    SAMPLES = 5

class HistoField(enum.IntEnum):
    BIN_START = 0
    BIN_END = 1
    L1_ERROR_MASK = 2
    L1_ERROR_NORMAL = 3
    L1_ERROR_DEPTH = 4
    L1_ERROR_AO = 5
    L1_ERROR_COLOR_NOAO = 6
    L1_ERROR_COLOR_WITHAO = 7

class HistoFieldDvr(enum.IntEnum):
    BIN_START = 0
    BIN_END = 1
    L1_ERROR_COLOR = 2
    L1_ERROR_ALPHA = 3

class StatsConfig:
    def __init__(
            self, importance : str = None, reconstruction : str = None,
            heatmin : float = None, heatmean : float = None, pattern : str = None):
        self.importance = importance
        self.reconstruction = reconstruction
        self.heatmin = heatmin
        self.heatmean = heatmean
        self.pattern = pattern

    def __repr__(self):
        return "%s-%s-%.3f-%.3f-%s"%(self.importance, self.reconstruction,
                                     self.heatmin, self.heatmean, self.pattern)
    def __hash__(self):
        return hash(repr(self))
    def __eq__(self, value):
        return repr(self) == repr(value)

class StatsLoader:
    def __init__(self, hdf5_file):
        # open hdf5_file
        with h5py.File(hdf5_file, 'r') as file:
            self._importance = list()
            self._reconstruction = list()
            self._networks = list()
            self._heatmin = list()
            self._heatmean = list()
            self._heatcfg = list()
            self._pattern = list()
            self._stats = dict()
            self._histograms = dict()
            print("Load all statistic files")
            for key in file.keys():
                dset = file[key]
                importance = dset.attrs['importance']
                reconstruction = dset.attrs['reconstruction']
                heatmin = dset.attrs['heatmin']
                heatmean = dset.attrs['heatmean']
                pattern = dset.attrs['pattern']

                if not importance in self._importance:
                    self._importance.append(importance)
                if not reconstruction in self._reconstruction:
                    self._reconstruction.append(reconstruction)
                if not (importance, reconstruction) in self._networks:
                    self._networks.append((importance, reconstruction))
                if not heatmin in self._heatmin:
                    self._heatmin.append(heatmin)
                if not heatmean in self._heatmean:
                    self._heatmean.append(heatmean)
                if not (heatmin, heatmean) in self._heatcfg:
                    self._heatcfg.append((heatmin, heatmean))
                if not pattern in self._pattern:
                    self._pattern.append(pattern)

                cfg = StatsConfig(importance, reconstruction, heatmin, heatmean, pattern)

                if key.startswith('Stats'):
                    self._stats[cfg] = dset[...]
                elif key.startswith('Histogram'):
                    self._histograms[cfg] = dset[...]
                else:
                    print("Unknown dataset:", key)

            print("importance networks:", self._importance)
            print("reconstruction networks:", self._reconstruction)
            print("all networks:", self._networks)
            print("heatmap min:", self._heatmin)
            print("heatmap mean:", self._heatmean)
            print("all heatmap configs:", self._heatcfg)
            print("sampling pattern:", self._pattern)

    def importanceNetworks(self):
        return list(self._importance)
    def reconstructionNetworks(self):
        return list(self._reconstruction)
    def allNetworks(self):
        return list(self._networks)
    def heatmapMin(self):
        return list(self._heatmin)
    def heatmapMean(self):
        return list(self._heatmean)
    def allHeatmapConfigs(self):
        return list(self._heatcfg)
    def samplingPattern(self):
        return list(self._pattern)

    def getStatistics(self, cfg : StatsConfig):
        return self._stats.get(cfg, None)
    def getHistogram(self, cfg : StatsConfig):
        return self._histograms.get(cfg, None)

