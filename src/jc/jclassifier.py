from __future__ import division
from .core import *
from .discontinuity import Discontinuity
from .mugp import MuGP


class JumpClassifier(object):
    
    def __init__(self, kdata, hp, window_width=100, kernel='e', use_gp=False):
        assert isinstance(kdata, KData)
        assert isinstance(use_gp, bool)
        assert kernel in ['e', 'g']

        self._kdata = kdata
        self.cadence = self._kdata.cadence
        self.flux = self._kdata.mf_normalized_flux

        self.use_gp = use_gp
        self.gp = MuGP(kernel=kernel)
        self.hp = hp
        
            
    def classify(self, jumps, use_de=True, de_niter=200, de_npop=30, method='Nelder-Mead'):
        """Classify flux discontinuities

        Classifies given flux discontinuities (jumps) as noise, jump, transit, or flare. 

        parameters
        ----------
        jumps    : Jump or a list of Jumps

        de_niter : int, optional
                   number of differential evolution iterations

        de_npop  : int, optional,
                   size of the differential evolution population

        Notes
        -----
        The classification is based on a simple BIC comparison. That is, we fit several
        models to the discontinuities and select the one with the lowest BIC value.

        The ln likelihood space can be pretty nasty even when our models are simple
        (thanks to GPs), so we do a small global optimisation run using Differential
        Evolution (DE) before a local optimisation. The DE run is the most time consuming
        part of the process, but necessary for realiable classification.
        """
        if isinstance(jumps, list):
            [self._classify_single(j, use_de, de_niter, de_npop, method) for j in jumps]
        elif isinstance(jumps, Discontinuity):
            self._classify_single(jumps, use_de, de_niter, de_npop)
        else:
            raise NotImplementedError('jumps must be a list of jumps or a single jump.')
        
            
    def _classify_single(self, jump, use_de=True, de_niter=100, de_npop=30, method='Nelder-Mead'):
        jump.classify(use_de, de_npop, de_niter, method,
                          wn_estimate = self.hp[2],
                          gp = self.gp if self.use_gp else None,
                          hp = self.hp if self.use_gp else None)
