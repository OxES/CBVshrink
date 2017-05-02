import numpy as np
import pylab as pl
from VBLinRegARD import bayes_linear_fit_ard as VBF
from stats import cdpp, medransig
import astropy.io.fits as pyfits

def fit_basis(flux, basis, scl = None):
    '''
    weights = fit_basis(flux, basis, scl = None)
    fit VB linear basis model to one or more light curves

    Inputs:
        flux: (nobj x nobs) light curve(s) 
        basis: (nobs x nb) basis trends
        scl: (nb) prior scaling factors for the basis trends
    Outputs:
        weights: (nobj x nb) weights
    '''
    # pre-process basis
    nb,nobs = basis.shape
    B = np.matrix(basis.T)
    if scl == None: scl = np.ones(nb)
    Bnorm = np.multiply(B, scl)
    Bs = Bnorm.std()
    Bnorm /= Bs
    Bnorm = np.concatenate((Bnorm, np.ones((nobs,1))), axis=1)
    # array to store weights
    nobj = flux.shape[0]
    weights = np.zeros((nobj,nb))
    for iobj in np.arange(nobj): 
        # pre-process flux
        F = np.matrix(flux[iobj,:]).T
        l = np.isfinite(F)
        Fm = F.mean()
        Fs = F.std()
        Fnorm = (F - Fm) / Fs
        res = VBF(Bnorm, Fnorm)
        w, V, invV, logdetV, an, bn, E_a, L = res
        weights[iobj,:] = np.array(res[0][:-1]).flatten() * scl * Fs / Bs
    return weights

def apply_basis(weights, basis):
    '''
    model = apply_basis(weights, basis) 
    Compute linear basis model given weights and basis matrix

    Inputs:
        weights: (nobj x nb) weights
        basis: (nobs x nb) basis trends
    Outputs:
        corr: (nobj x nobs) correction to apply to light curves
    '''
    return np.dot(weights, basis)

def fixed_nb(flux, cbv, nB = 4, use = None, doPlot = True):
    '''
    corrected_flux = fixed_nb(flux, basis, nB = 4, use = None, \
                              doPlot = True)
    Correct light curve for systematics using first nB CBVs.

    Inputs:
        flux: (1-D array) light curves 
        cbv: (2-D array) co-trending basis vectors trends
    Optional inputs:
        nB: number of CBVs to use (the first nB are used)
        use: boolean array, True for data points to use in evaluating correction, 
             False for data points to ignore (NaNs are also ignored)
        doPlot: set to False to suppress plot
    Outputs:
        corrected_flux: (same shape as flux) corrected light curves
        weights: (nB array) basis vector coefficients
    '''
    nobs = len(flux)    
    if cbv.shape[1] == nobs: cbv_ = cbv[:nB,:]
    else: cbv_ = cbv[:,:nB].T
    corrected_flux = np.copy(flux)
    l = np.isfinite(flux)
    if not use is None: l *= use
    weights = fit_basis(flux[l].reshape((1,l.sum())), cbv_[:,l])
    corr = apply_basis(weights, cbv_).reshape(flux.shape)
    corrected_flux = flux - corr
    if doPlot == True:
        pl.clf()
        x = np.arange(nobs)
        pl.plot(x, flux, '-', c = 'grey')
        pl.plot(x[l], flux[l], 'k-')
        pl.plot(x, corr, 'c-')
        pl.plot(x, corrected_flux, 'm-')
        pl.xlabel('Observation number')
        pl.xlabel('Flux')
    return corrected_flux, weights

def sel_nb(flux, cbv, nBmax = None, use = None):
    '''
    (nb_opt, flux_opt, weights_opt), (corr_flux_multi, weights_multi)
                = sel_nb(flux, basis, nBmax = 8, use = None)
    Correct light curve for systematics using upt to nB CBVs 
    (automatically select best number).

    Inputs:
        flux: (1-D array) light curves 
        cbv: (2-D array) co-trending basis vectors trends
    Optional inputs:
        nBmax: maximum number of CBVs to use (starting with the first)
        use: boolean array, True for data points to use in evaluating 
             correction, False for data points to ignore (NaNs are also ignored)
    Outputs:
        nBopt: automatically selected number of CBVs used (<= nBmax)
        corr_flux: (same shape as flux) corrected light curves
        weights: (nBopt array) basis vector coefficients
    '''
    nobs = len(flux)
    if cbv.shape[1] == nobs: cbv_ = np.copy(cbv)
    else: cbv_ = cbv.T
    if nBmax is None: nBmax = cbv.shape[0]
    else: cbv_ = cbv_[:nBmax,:]
        
    corr_flux = np.zeros(nobs)
    corr_flux_multi = np.zeros((nBmax,nobs))
    weights_multi = np.zeros((nBmax,nBmax))
    ran_multi = np.zeros(nBmax)
    sig_multi = np.zeros(nBmax)

    l = np.isfinite(flux)
    if not use is None: l *= use

    med_raw, ran_raw, sig_raw = medransig(flux[l])

    for i in range(nBmax):
        cbv_c = cbv_[:i+1,:]
        w_c = fit_basis(flux[l].reshape((1,l.sum())), cbv_c[:,l])
        w_ext = np.zeros(nBmax)
        w_ext[:i+1] = w_c
        weights_multi[i,:] = w_ext
        corr = apply_basis(w_c, cbv_c).reshape(flux.shape)
        c = flux - corr
        med, ran, sig = medransig(c[l])
        corr_flux_multi[i,:] = c - med + med_raw
        ran_multi[i] = ran
        sig_multi[i] = sig

    # Select the best number of basis functions
    # (smallest number that significantly reduces range)
    med_ran = np.median(ran_multi)
    sig_ran = 1.48 * np.median(abs(ran_multi - med_ran))
    jj = np.where(ran_multi < med_ran + 3 * sig_ran)[0][0]
    # Does that introduce noise? If so try to reduce nB till it doesn't
    while (sig_multi[jj] > 1.1 * sig_raw) and (jj > 0): jj -= 1

    nb_opt = jj + 1
    flux_opt = corr_flux_multi[jj,:].flatten()
    weights_opt = weights_multi[jj,:][:jj+1].flatten()
    ran_opt = ran_multi[jj]
    sig_opt = sig_multi[jj]
    return (nb_opt, flux_opt, weights_opt), \
      (corr_flux_multi, weights_multi)

# def correct_file(infile, cbvfile, outfile, input_type = 'SAP', \
#                  verbose = False, doplot = False, #
#                  exclude_func = None, exclude_func_par = None):
#     '''
#     time, cadence, corrected_flux = correct_file_nB(infile, cbvfile, outfile, \
#         input_type = 'SAP', exclude_func = None, \
#         exclude_func_par = None)

#     Correct light curve containined in infile using CBVs contained in
#     cbvfile, using up to nBmax CBVs

#     Inputs:
#         infile: input (FITS) light curve file
#         cbvfile: input (FITS) CBV file
#         outfile: output (FITS) file to save results in. This is a copy of the input
#             file with extra column 'CBV_FLUX', containing the systematics-corrected fluxes.
#             The weights associated with each CBV are saved in the header (CBVW_X, where
#             X is the index of the CBV to which the weight is applied). Also stored are
#             the range, point-to-point scatter and 6.5-hour CDPPs after correction
#             (RAN_CBV, SIG_CBV, and CDPP_CBV).
#     Optional inputs:
#         input_type: type of data to use as input. Options are:
#             SAP: "raw" (simple aperture photometry) data
#             JCR: "jump-corrected" data 
#         verbose: toggle to produce more / less text output
#         doplot: toggle to produce plots on screen showing the evolution of the correction
#         exclude_func: function f(t,par), which returns list of indices to ignore
#         exclude_func_par: parameters of exclude function
#             as more CBVs are added
#     '''
#     nBmax = 8
#     # Read in light curve data
#     h1 = pyfits.open(infile, mode = 'readonly')
#     kic = h1[0].header['KEPLERID']
#     quarter = h1[0].header['QUARTER']
#     module = h1[0].header['MODULE']
#     output = h1[0].header['OUTPUT']
#     if verbose:
#         print 'Reading in quarter %d light curve data for KIC %d.' % \
#             (quarter, kic)
#         print 'Object is located on module %d, output channel %d.' \
#       % (module, output)
#     if input_type == 'SAP':
#         if verbose: print 'Reading SAP data'
#         flux = h1[1].data.field('SAP_FLUX').astype('float64')
#     elif input_type == 'JCR':
#         if verbose: print 'Reading JCR data'
#         flux = h1[1].data.field('JCR_FLUX').astype('float64')
#     else:
#         print 'Error: input type %s not supported'
#         return
#     time = h1[1].data.field('TIME').astype('float64')
#     pdc = h1[1].data.field('PDCSAP_FLUX').astype('float64')
#     if doplot == True:
#         pl.clf()
#         l = np.isfinite(time)
#         tmin = time[l].min()
#         tmax = time[l].max()
#     nobs = len(flux)
#     l = np.isfinite(flux)
#     nval = l.sum()
#     print 'Read in %d observations of which %d valid.' % (nobs, nval)
#     # Read in CBV data
#     cbv = np.zeros((nobs, 16))
#     h2 = pyfits.open(cbvfile)
#     if h2[0].header['QUARTER'] != quarter:
#         print 'Error: CBV file is for quarter %d.' % h2[0].header['QUARTER']
#         return
#     n_ext = len(h2) - 1
#     for i in np.arange(n_ext)+1:
#         if h2[i].header['MODULE'] != module: continue
#         if h2[i].header['OUTPUT'] != output: continue
#         for j in np.arange(16):
#             cbv[:,j] = h2[i].data.field('VECTOR_%d' % (j+1)).astype('float64')
#         break
#     h2.close()
#     # Identify any observations to ignore
#     if exclude_func != None:
#         if exclude_func_par == None:
#             exclude_indices = exclude_func(time)
#         else:
#             exclude_indices = exclude_func(time, exclude_func_par)
#         use = np.ones(nobs, 'bool')
#         use[exclude_indices] = False
#     else:
#         use = None
#     # Stats before correction - store in header keywords
#     mms, sap_ran, sap_sig = medransig(flux[np.isfinite(flux)])
#     if verbose: print 'Median flux: %f' % mms
#     h1[1].header['MED_FLUX'] = repr(mms)
#     if verbose: print 'Input range: %f' % sap_ran
#     h1[1].header['SAP_RAN'] = repr(sap_ran)
#     if verbose: print 'Input p2p scatter: %f' % sap_sig  
#     h1[1].header['SAP_SIG'] = repr(sap_sig)
#     sap_cdpp = cdpp(time, flux)
#     if verbose: print 'Input CDPP: %f' % sap_cdpp
#     h1[1].header['SAP_CDPP'] = repr(sap_cdpp)
#     mmp = np.median(pdc[np.isfinite(pdc)])
#     pdc = pdc - mmp + mms
#     _, pdc_ran, pdc_sig = medransig(pdc[np.isfinite(pdc)])
#     if verbose: print 'PDC range: %f' % pdc_ran
#     h1[1].header['PDC_RAN'] = repr(pdc_ran)
#     if verbose: print 'PDC p2p scatter: %f' % pdc_sig  
#     h1[1].header['PDC_SIG'] = repr(pdc_sig)
#     pdc_cdpp = cdpp(time, pdc)
#     if verbose: print 'PDC CDPP: %f' % pdc_cdpp
#     h1[1].header['PDC_CDPP'] = repr(pdc_cdpp)
#     # Preliminary plotting commands
#     if doplot == True:
#         ax1 = pl.subplot(211)
#         diff1 = flux[1:] - flux[:-1]
#         ll1 = np.isfinite(diff1)
#         mm1 = np.median(diff1[ll1])
#         offset1 = 5 * 1.48 * np.median(abs(diff1[ll1] - mm1))
#         pl.plot(time, flux, 'k-')
#         pl.plot(time, flux - pdc + mms - offset1, 'g-')
#         pl.ylabel('raw flux')
#         pl.title('KID%d Q%d (module %d output %d)' % \
#                      (kic, quarter, module, output))
#         ax2 = pl.subplot(212, sharex = ax1)
#         pl.plot(time, pdc, 'g-')        
#         diff2 = pdc[1:] - pdc[:-1]
#         ll2 = np.isfinite(diff2)
#         mm2 = np.median(diff2[ll2])
#         offset2 = 5 * 1.48 * np.median(abs(diff2[ll2] - mm2))
#         pl.ylabel('corr. flux')
#         pl.xlabel('time')
#     # Perform correction
#     (nb, flux_cbv, weights, cbv_ran, cbv_sig), \
#       (flux_multi, _, _, _) = \
#       sel_nb(flux, cbv, nBmax = nBmax, use = use)
#     # Plot results for individual nB values, if requested
#     if doplot == True:
#         for i in np.arange(nBmax):
#             flux_cbv = flux_multi[i,:].flatten()
#             mmc = np.median(flux_cbv[np.isfinite(flux_cbv)])
#             flux_cbv = flux_cbv - mmc + mms
#             corr = flux - flux_cbv + mms
#             dr = i/float(nBmax-1)
#             rgb = (1-dr,0,dr)
#             pl.sca(ax1)
#             pl.plot(time, corr - offset1 * (i+2), c = rgb)
#             pl.sca(ax2)
#             pl.plot(time, flux_cbv - offset2 * (i+1), c = rgb)
#         pl.xlim(tmin, tmax)
#     # Store results in relevant FITS column and header keywords
#     if verbose: print 'Optimal no. CBVs: %d' % nb
#     h1[1].header['CBV_NSEL'] = repr(nb)
#     if verbose: print 'Weights:', weights
#     for i in range(nb):
#         h1[1].header['CBVW_%02d' % i] = repr(weights[i])
#     if verbose: print 'CBV range: %f' % cbv_ran
#     h1[1].header['CBV_RAN'] = repr(cbv_ran)
#     if verbose: print 'CBV p2p scatter: %f' % cbv_sig  
#     h1[1].header['CBV_SIG'] = repr(cbv_sig)
#     mmc = np.median(flux_cbv[np.isfinite(flux_cbv)])
#     flux_cbv = flux_cbv - mmc + mms
#     cbv_cdpp = cdpp(time, flux_cbv)
#     if verbose: print 'CBV CDPP: %f' % cbv_cdpp
#     h1[1].header['CBV_CDPP'] = repr(cbv_cdpp)
#     unit = h1[1].header['TUNIT4']
#     cols = h1[1].columns
#     col = pyfits.Column(name = 'CBV_FLUX', format = 'E', disp = 'E14.7', \
#                         unit = unit, array = flux_cbv)
#     cols += col
#     # Save
#     hdr_save = h1
#     h1[1] = pyfits.BinTableHDU.from_columns(cols, header=h1[1].header)
#     if verbose: print 'Saving to file %s' % outfile
#     h1.writeto(outfile, clobber = True)
#     h1.close()
#     return 
