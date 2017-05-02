from __future__ import division
from math import log, pi
from numpy import ndarray, array, zeros
from scipy.optimize import minimize
from pyde.de import DiffEvol
from .core import *
from .fmodels import models as fm

ln_two_pi = log(2*pi)


def lnlikelihood(obs, mod, err):
    n = float(obs.size)
    return -0.5 * (n*ln_two_pi + n*log(err**2) + ((obs-mod)**2).sum()/err**2)


class DiscontinuityType(object):
    name    = ''
    pnames  = []
    npar    = len(pnames)
    penalty = 0.
    
    def __init__(self, discontinuity):
        self.discontinuity = self.d = discontinuity
        self.best_fit_pv = None
        self.best_fit_model = None
        self.bic = None
        self._optimization_result = None
            
#    def __str__(self):
#        return '{:s} {:4.1f}  {:4.1f}'.format(self.name, self.position, self.amplitude)

#    def __repr__(self):
#        return '{:s}({:4.1f}, {:4.1f})'.format(self.name, self.position, self.amplitude)

            
    def nlnlike_wn(self, pv):
        if not self.is_inside_bounds(pv):
            return inf
        else:
            return -lnlikelihood(self.d.flux, self.model(pv), self.d.wn_estimate)

        
    def nlnlike_gp(self, pv):
        if not self.is_inside_bounds(pv):
            return inf
        else:
            return -self.d.gp.lnlikelihood(self.d.cadence, self.d.flux-self.model(pv), freeze_k=True)    

        
    def fit(self, use_de=False, de_npop=30, de_niter=100, method='Powell'):
        jamp, jpos, fstd = self.d.amplitude, self.d.position, self.d.flux.std()
        nlnlike = self.nlnlike_gp if self.d.use_gp else self.nlnlike_wn
        if use_de:
            self._de = DiffEvol(nlnlike, self._de_bounds(jamp, jpos, fstd), npop=de_npop)
            self._de.optimize(de_niter)
            pv0 = self._de.minimum_location
        else:
            pv0 = self._pv0(jamp, jpos, fstd)
            
        self._optimization_result = r = minimize(nlnlike, pv0, method=method)
        self.best_fit_pv = r.x
        self.best_fit_model = self.model(r.x)
        xx = r.x.copy()
        xx[-2:] = 0
        self.best_fit_model_wo_baseline = self.model(xx)
        self.bic = self.c_bic(r.fun)
        return self.bic

    
    def model(self, pv, cad=None):
        raise NotImplementedError

    
    def c_bic(self, nln):
        return 2*nln + self.npar*log(self.d.npt) + self.penalty

    
class UnclassifiedDiscontinuity(DiscontinuityType):
    name   = 'Unclassified'
    pnames = []
    npar   = len(pnames)

    
    def fit(self, *nargs, **kwargs):
        raise NotImplementedError

    
class Slope(DiscontinuityType):
    name   = 'slope'
    pnames = 'slope intercept'.split()
    npar   = len(pnames)

    def model(self, pv, cad=None):
        return np.poly1d(pv)(self.d.cadence if cad is None else cad)

    def is_inside_bounds(self, pv):
        return True
    
    def fit(self, use_de=False, de_npop=30, de_niter=100, method='Powell'):
        nlnlike = self.nlnlike_gp if self.d.use_gp else self.nlnlike_wn
        pv0 = np.polyfit(self.d.cadence, self.d.flux, 2)
        self._optimization_result = r = minimize(nlnlike, pv0, method=method)
        self.best_fit_pv = r.x
        self.best_fit_model = self.model(r.x)
        self.bic = self.c_bic(r.fun)
        return self.bic

    
class Jump(DiscontinuityType):
    name   = 'jump'
    pnames = 'center width amplitude bl_constant bl_slope'.split()
    npar   = len(pnames)

    def model(self, pv, cad=None):
        return fm.m_jump(*pv, cadence=self.d.cadence if cad is None else cad)

    def is_inside_bounds(self, pv):
        return all(pv[:2] > 0) and (0.5 < pv[1] < 3.0) and (self.d.position-3 <= pv[0] <= self.d.position+3)
    
    def _de_bounds(self, jamp, jpos, fstd):
        return [[     jpos-2,     jpos+2],  # 0 - center
                [          1,          3],  # 1 - width
                [  0.75*jamp,  1.25*jamp],  # 2 - amplitude
                [ -0.20*fstd,  0.20*fstd],  # 3 - baseline constant
                [      -1e-3,       1e-3]]  # 4 - baseline slope 

    def _pv0(self, jamp, jpos, fstd):
        return [jpos, 2, jamp, fstd, 0]

    
class Jump2(DiscontinuityType):
    name    = 'jumpf'
    pnames  = 'center width famp jamp bl_constant bl_slope'.split()
    npar    = len(pnames)
    
    def model(self, pv, cad=None):
        return fm.m_jumpf(*pv, cadence=self.d.cadence if cad is None else cad)

    def is_inside_bounds(self, pv):
        return ((self.d.position-3 <= pv[0] <= self.d.position+3)
                and (2.5 < pv[1] < 20.)
                and (pv[2] < 0.)
                and (pv[3] < 0.)
                and (pv[2] < pv[3]))
    
    def _de_bounds(self, jamp, jpos, fstd):
        return [[     jpos-4,     jpos+4],  # 0 - center
                [          1,         10],  # 1 - width
                [  0.75*jamp,  1.25*jamp],  # 2 - jump amplitude
                [  0.75*jamp,  1.25*jamp],  # 3 - baseline level after jump
                [ -0.20*fstd,  0.20*fstd],  # 4 - baseline constant
                [      -1e-3,       1e-3]]  # 5 - baseline slope 

    def _pv0(self, jamp, jpos, fstd):
        return [jpos, 2, jamp, 0.5*jamp, fstd, 0]
            

class Transit(DiscontinuityType):
    name   = 'transit'
    pnames = 'depth center duration bl_constant bl_slope'.split()
    npar   = len(pnames)

    def model(self, pv, cad=None):
        return fm.m_transit(*pv, cadence=self.d.cadence if cad is None else cad)
    
    def is_inside_bounds(self, pv):
        return ((pv[0] > 0.)
                and (self.d.cadence[1]+0.55*pv[2] < pv[1] < self.d.cadence[-2]-0.55*pv[2])
                and (1. < pv[2] < 50.))

    def _de_bounds(self, jamp, jpos, fstd):
        return [[ -0.8*jamp, -1.2*jamp],  # 0 - transit depth
                [    jpos-5,   jpos+5],  # 1 - center 
                [       1.2,      50.],  # 2 - duration
                [ -0.2*fstd, 0.2*fstd],  # 3 - baseline constant
                [     -1e-3,     1e-3]]  # 4 - baseline slope
                
    def _pv0(self, jamp, jpos, fstd):
        return [-jamp, jpos, 10, fstd, 0]
            
    
class Flare(DiscontinuityType):
    name   = 'flare'
    pnames = 'start duration amplitude bl_constant bl_slope'.split()
    npar   = len(pnames)

    def model(self, pv, cad=None):
        return fm.m_flare(*pv, cadence=self.d.cadence if cad is None else cad)

    def is_inside_bounds(self, pv):
        return all(pv[:3] >= 0.) and (self.d.cadence[0] < pv[0] < self.d.cadence[-1]) and (pv[1] < 10.)

    def _de_bounds(self, jamp, jpos, fstd):
        return [[   jpos-5,   jpos+5],  # 0 - flare start
                [      1.2,       7.],  # 1 - flare duration
                [ 0.8*jamp, 1.2*jamp],  # 2 - amplitude
                [-0.2*fstd, 0.2*fstd],  # 3 - baseline constant
                [    -1e-3,     1e-3]]  # 4 - baseline slope
               
    def _pv0(self, jamp, jpos, fstd):
        return array([jpos, 2.5, jamp, fstd, 0])
    
dmodels = Slope, Jump, Jump2, Transit, Flare
