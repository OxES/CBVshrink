from .core import *

class MuGP(object):
    """ A minimalistic Gaussian process class.
    """
    def __init__(self, kernel='e', pv0=None):
        """
        Parameters
        ----------
        kernel  : char   GP kernel, either (e)xponential or (g)aussian
        """
        self.kernels = {'e': lambda isc,cl:afa(isc**2*np.exp(-self.D/cl)),
                        'g': lambda isc,cl:afa(isc**2*np.exp(-0.5*(self.D/cl)**2))}
        self.kernel = self.kernels[kernel]
        self.pv0 = np.asarray(pv0) if pv0 is not None else np.array([0.01, 5000, 8e-4])
        self.pv = self.pv0.copy()
        self.x = None
        self.dirty = True

    def set_parameters(self, pv):
        self.pv[:] = pv
        self.dirty = True

    def _compute_alpha(self, y):
        self._alpha = sla.cho_solve(self.L, y)
        
    def _compute_k0(self, x=None):
        self.x   = x
        self.D   = (np.abs(np.subtract(*np.meshgrid(self.x, self.x))))
        self.K00 = self.kernel(self.pv[0], self.pv[1])

    def compute(self, x, split=None):
        if x is None or not np.array_equal(x, self.x) or self.dirty:
            self._compute_k0(x)

        self.K0 = self.K00*self.get_mask(split)
        self.K  = self.K0.copy() + self.pv[2]**2 * np.identity(self.K0.shape[0])
        self.L = sla.cho_factor(self.K)
        self.dirty = True


    def lnlikelihood(self, x, y, split=None, freeze_k=False):
        if not freeze_k:
            self.compute(x, split)
        self._compute_alpha(y)
        return -(np.log(np.diag(self.L[0])).sum() + 0.5 * np.dot(y,self._alpha))
    
    def get_mask(self, split=None):
        npt = self.x.size
        if split is None:
            return np.ones((npt,npt))
        else:
            isplit = np.argmin(np.abs(self.x-split))
            mask = np.zeros((npt,npt))
            mask[:isplit,:isplit] = 1.0
            mask[isplit:,isplit:] = 1.0
            return mask

    def __call__(self, y):
        vtmp = np.zeros_like(y)
        ymean = y.mean()
        y = y - ymean
        vtmp = bl.dsymv(1., self.P, y, 0, vtmp)
        return bl.dsymv(1., self.K0, vtmp, 0, vtmp) + ymean

    def predict(self, y, mean_only=True):
        ymean = y.mean()
        y = y-ymean
        self._compute_alpha(y)
        Ks  = self.K0.copy()
        Kss = self.K.copy()
        mu = np.dot(Ks, self._alpha)

        if mean_only:
            return ymean+mu
        else:
            b = sla.cho_solve(self.L, Ks.T)
            cov = Kss - np.dot(Ks, b)
            err = np.sqrt(np.diag(cov))
            return ymean+mu, err
