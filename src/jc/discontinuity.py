from __future__ import division
from numpy import ndarray, array, zeros, nanmedian, isfinite, diff, sqrt
from scipy.signal import medfilt as mf
from .core import *
from .models import *

jump_classes = dmodels

class Discontinuity(object):

    _available_models = Slope, Jump, Jump2, Transit, Flare

    def __init__(self, position, amplitude, kdata, window_width=100):
        """
        Parameters
        ----------
        position : float
            Approximate position of the discontinuity.
        amplitude : float
            Approximate amplitude of the discontinuity.
        cadence : 1D array
            Cadence array
        flux : 1D array
            Flux array
        window_width : int (> 25)
            Size of the local window
        """

        assert isinstance(window_width, int) and window_width > 25
        assert all(kdata.masked_flux[isfinite(kdata.masked_flux)] > 0), "Flux array contains negative values."

        self.data = kdata
        self.position = float(position)        # Approximate discontinuity position
        self._amplitude_g = float(amplitude)   # Approximate discontinuity amplitude
        self._cadence_g = kdata.masked_cadence # Global cadence array
        self._flux_g = 1. + kdata.normalized_masked_flux  # Global flux array

        self.use_gp = False
        self.gp = None
        self.hp = None
        
        # Initialize discontinuity models
        # -------------------------------
        self.type = UnclassifiedDiscontinuity(self)
        self.models = [M(self) for M in self._available_models]
        self.bics = zeros(len(self.models))

        # Create local cadence and flux arrays
        # ------------------------------------
        self.cadence, self.flux, self.ids = kdata.get_window(self.position, window_width, masked=True)
        self.flux /= kdata.median
        self.npt = self.cadence.size
        self.local_median = nanmedian(self.flux)
        self.flux[:] = self.global_to_local(self.flux)
        self.wn_estimate = diff(self.flux).std() / sqrt(2.)
        self.amplitude = self._amplitude_g / self.local_median
        
            
    def setup_gp(self, gp, hp):
        self.use_gp = True
        self.gp = gp
        self.hp = hp
        self.gp.set_parameters(hp)
        self.gp.compute(self.cadence)

        
    def classify(self, use_de=True, de_npop=30, de_niter=100, method='Nelder-Mead', wn_estimate=None, gp=None, hp=None):
        if gp is not None and hp is not None:
            self.setup_gp(gp, hp)

        if wn_estimate is not None:
            self.wn_estimate = wn_estimate / self.local_median
            
        self.bics[:] = array([m.fit(use_de, de_npop, de_niter, method) for m in self.models])
        self.bics -= self.bics.min()
        self.type = self.models[self.bics.argmin()]
        self._pv = self.type.best_fit_pv


    def local_to_global(self, a):
        return (1. + a) * self.local_median

    def global_to_local(self, a):
        return a / self.local_median - 1.

    
    def global_model_wo_baseline(self):
        pv = self._pv.copy()
        pv[-2:] = 0
        return self.local_median * self.type.model(pv, self.data.cadence)

    
    @property
    def name(self):
        return self.type.name

    @property
    def global_cadence(self):
        return self._cadence_g

    @property
    def local_cadence(self):
        return self.cadence

    @property
    def global_flux(self):
        return self._flux_g

    @property
    def local_flux(self):
        return self.flux


class DiscontinuitySet(list):
    def __init__(self, values=[]):
        if isinstance(values, Discontinuity):
            super(DiscontinuitySet, self).__init__([values])
        elif isinstance(values, list) and all([isinstance(v, Discontinuity) for v in values]):
            super(DiscontinuitySet, self).__init__(values)
        else:
            raise TypeError('DiscontinuitySet can contain only Jumps')
        
    def append(self, v):
        if isinstance(v, Discontinuity):
            super(DiscontinuitySet, self).append(v)
        else:
            raise TypeError('DiscontinuitySet can contain only Jumps')

    @property
    def types(self):
        return [j.name for j in self]
            
    @property
    def amplitudes(self):
        return [j.amplitude for j in self]
    
    @property
    def bics(self):
        if with_pandas:
            return pd.DataFrame([j.bics for j in self], columns=jump_classes)
        else:
            return np.array([j.bics for j in self])
    
