from __future__ import division

import sys
import math
import numpy as np

import scipy.ndimage as ndi
import scipy.linalg.lapack as lp
import scipy.linalg.blas as bl
import scipy.linalg as sla

from scipy.optimize import fmin, fmin_powell, fmin_cg
from scipy.stats import norm
from scipy.ndimage import median_filter as mf
from scipy.ndimage import label

from numpy import asfortranarray as afa
from numpy import s_, inf, median, zeros_like, array, argmax, where

try:
    import pandas as pd
    with_pandas = True
except ImportError:
    with_pandas = False

try:
    import matplotlib.pyplot as pl
    with_matplotlib = True
except ImportError:
    with_matplotlib = False


## Utility functions
## =================
def amax(v):
    return v[np.argmax(np.abs(v))]

class KData(object):
    def __init__(self, cadence, flux, quality):
        self._cadence = cadence.copy()
        self._flux = flux.copy()
        self._quality = quality.copy()
        self._flux_o = flux.copy()
        self._mask = np.isfinite(cadence) & np.isfinite(flux)
        
    @property
    def cadence(self):
        return self._cadence[self._mask]

    @property
    def flux(self):
        return self._flux[self._mask]

    @property
    def mf_flux(self):
        f1 = self.flux
        f2 = mf(f1, 3)
        fr = f1-f2
        mad = np.median(abs(fr))
        return np.where(fr > 20*mad, f2, f1)
    
    @property
    def normalized_flux(self):
        return self.flux / self.median - 1.

    @property
    def mf_normalized_flux(self):
        f1 = self.normalized_flux
        f2 = mf(f1, 3)
        fr = f1-f2
        mad = np.median(abs(fr))
        return np.where(fr > 20*mad, f2, f1)
        
    @property
    def median(self):
        return np.median(self.flux)

    @property
    def original_flux(self):
        return self._flux_o[self._mask]

    
    def calculate_exclusion_ranges(self, b):
        a = array([-b,b])
        lids, nreg  = label((self._quality & 2**3).astype(np.bool))
        return [self._cadence[lids == i][[0,-1]] + a for i in range(1,nreg+1)]
    
    
    def plot(self, jumps=None, ax=None):
        if ax is None:
            fig,ax = pl.subplots(1,1)

        ax.plot(self._cadence, self._flux, 'k')

        if jumps is not None:
            [ax.axvline(j.pos, ls=':', c='k', alpha=0.25) for j in jumps if j.type=='noise'] 
            [ax.axvline(j.pos, ls='--', c='k', alpha=0.5) for j in jumps if j.type=='jump']
            [ax.axvline(j.pos, 0.9, 1.0, c='k', lw=3)     for j in jumps if j.type=='transit']
            [ax.axvline(j.pos, ls='-', c='k', alpha=0.5) for j in jumps if j.type=='flare']

        pl.setp(ax, xlim=self._cadence[[0,-1]], xlabel='Cadence', ylabel='Flux')

        if ax is None:
            fig.tight_layout()
            
            
