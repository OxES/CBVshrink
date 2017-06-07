from __future__ import division
from tqdm import tqdm
from scipy.optimize import minimize
from numpy import sign, isfinite, poly1d, polyfit
from astropy.stats import sigma_clip, LombScargle
from .core import *
from .mugp import MuGP
from .discontinuity import Discontinuity, DiscontinuitySet

class JumpFinder(object):
    def __init__(self, kdata, kernel='e', chunk_size=128, **kwargs):
        """Discontinuity detection for photometric time series.

        Parameters
        ----------
        cadence  : 1D array
        flux  : 1D array

        Keyword arguments
        -----------------
        exclude : list of cadence ranges to be excluded

        n_iterations   : int
        min_gap_width  : int
        min_separation : int

        sigma    : float
        csize    :   int
        wnoise   : float
        clength  :   int

        Returns
        -------
        jumps : list of jump objects
        """
        self.gp = MuGP(kernel=kernel)
        self._kdata = kdata
        self.hps = None
        self.hp  = None
        
        self.cadence = self._kdata.cadence
        self.flux    = self._kdata.mf_normalized_flux
        self.period  = self.estimate_period()

        self.gp.pv0[3] = self.period
        self.gp.pv[3] = self.period

        self.chunk_size = cs = chunk_size
        self.n_chunks   = nc = self.flux.size // chunk_size
        self.chunks = [s_[i*cs:(i+1)*cs] for i in range(nc)] + [s_[cs*nc:]]
        
        self.exclude = kwargs.get('exclude', [])
        self.exclude.append(self.cadence[[0,25]])
        self.exclude.extend(kdata.calculate_exclusion_ranges(5))


    def estimate_period(self):
        mask = isfinite(self.cadence) & isfinite(self.flux)
        cd, fl = self.cadence[mask] - self.cadence[mask].mean(), self.flux[mask]
        fl -= poly1d(polyfit(cd, fl, 7))(cd)
        freq = np.linspace(0.005, 0.05, 5000)
        power = LombScargle(cd, fl).power(freq, method='fast')
        return 1. / freq[np.argmax(power)]


    def minfun(self, pv, sl):
        if any(pv <= 0): 
            return inf
        self.gp.pv[[0,1,2,4]] = pv
        self.gp.dirty = True
        return -self.gp.lnlikelihood(self.cadence[sl], self.flux[sl])

    
    def learn_hp(self, max_chunks=50):
        self.hps = []
        for sl in tqdm(self.chunks[:max_chunks], desc='Learning noise hyperparameters'):
            cd, fl = self.cadence[sl], self.flux[sl]
            fl = fl - fl.mean()
            pv0 = [fl.std(), 50, 1, 1.4 * np.diff(fl).std()]

            def minfun(pv):
                if any(pv<=0):
                    return inf
                self.gp.pv[[0, 1, 2, 4]] = pv
                self.gp.dirty = True
                return -self.gp.lnlikelihood(cd, fl)

            res = minimize(minfun, pv0, method='nelder-mead')
            self.hps.append(res.x)
            #self.hps.append(fmin(self.minfun, self.gp.pv0, args=(sl,), disp=False))
        self.hp = median(self.hps, 0)

        
    def compute_lnl(self, wsize=None):
        self.lnlike = lnlike = zeros_like(self.flux)
        self.gp.pv[3] = self.period
        self.gp.pv[[0,1,2,4]] = self.hp
        #self.gp.set_parameters(self.hp)

        npt = self.cadence.size
        nsc = wsize or self.chunk_size
        hsc = nsc // 2

        for i in tqdm(range(5, npt - 5), desc='Scanning for discontinuities'):
            imin = min(max(i - hsc, 0), npt - nsc)
            imax = max(min(i + hsc, npt), nsc)
            cadence = self.cadence[imin:imax]
            flux = self.flux[imin:imax] - self.flux[imin:imax].mean()
            lnlike[i] = self.gp.lnlikelihood(cadence, flux, self.cadence[i]) - self.gp.lnlikelihood(cadence, flux)
        return lnlike

    
    def find_jumps(self, sigma_cut=10, learn=True, cln=True):
        if learn:
            self.learn_hp(max_chunks=15)
        if cln:
            self.compute_lnl()
        
        mlnlike = self.lnlike - mf(self.lnlike, 90)
        lnlrm = sigma_clip(mlnlike, sigma_upper=sigma_cut, sigma_lower=inf)
        labels, nl = label(lnlrm.mask)
        jumps = [self.cadence[argmax(where(labels == i, mlnlike, -inf))] for i in range(1, nl + 1)]

        # Compute the amplitudes
        # ----------------------
        jids   = [self.cadence.searchsorted(j) for j in jumps]
        slices = [s_[max(0,j-self.chunk_size//2):min(self.flux.size-1, j+self.chunk_size//2)] for j in jids]
        
        amplitudes = []
        for jump,sl in zip(jumps,slices):  
            cad = self.cadence[sl]
            self.gp.compute(cad, jump)
            pr = self.gp.predict(self.flux[sl])
            k = np.argmin(np.abs(cad-jump))
            amplitudes.append(pr[k]-pr[k-1])

        jumps = [Discontinuity(j,a, self.cadence, 1.+self.flux) for j,a in zip(jumps, amplitudes)]
        jumps = [j for j in jumps if not any([e[0] <= j.position <= e[1] for e in self.exclude])]

        # Merge likely transits
        # ---------------------
        merge_limit = 12
        jt = []
        skip = False
        for i, j1 in enumerate(jumps):
            if not skip:
                if (i < len(jumps) - 1):
                    j2 = jumps[i + 1]
                    if (j2.position - j1.position) < merge_limit and j1.amplitude < 0 and j2.amplitude > 0:
                        jt.append(Discontinuity(0.5 * (j1.position + j2.position), j1.amplitude, self.cadence, 1.+self.flux))
                        skip = True
                        continue
                jt.append(j1)
            else:
                skip = False
        jumps = jt
        return DiscontinuitySet(jumps)

    
    def plot(self, chunk=0, ax=None):
        if ax is None:
            fig, ax = subplots(1,1)
            
        sl = self.chunks[chunk]
        self.gp.set_parameters(self.hp)
        self.gp.compute(self.cadence[sl])
        ax.plot(self.cadence[sl], self.flux[sl], '.', c='0.5')
        ax.plot(self.cadence[sl], self.gp.predict(self.flux[sl]), c='w', lw=4, alpha=0.7)
        ax.plot(self.cadence[sl], self.gp.predict(self.flux[sl]), c='k', lw=2)
