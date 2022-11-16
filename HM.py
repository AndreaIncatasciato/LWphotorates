import os
import numpy as np
from astropy import constants as const
from astropy import units as u

from photorates.utils import nu2lambda,lambda2nu,spec_nu2lambda,spec_lambda2nu,cm2eV,eV2cm,cm2K,K2cm

from scipy.interpolate import InterpolatedUnivariateSpline



def read_sigma():

    '''
    Read the file with the HM detachment cross section from McLaughlin et al. (2017).

    Output:
        dictionary with:
        'energy'                                          [eV]
        'frequency'                                       [Hz]
        'sigma'                                           [cm**2]
    '''

    McL_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/HM/sigma_McLaughlin+2017.txt'
    McL=np.loadtxt(McL_f,unpack=True)
    mask=McL[0]<=(const.Ryd*const.m_p/(const.m_p+const.m_e)*const.h.to(u.eV/u.Hz)*const.c.to(u.m*u.Hz)).value
    en_sigma=McL[0][mask]*u.eV
    freq_sigma=en_sigma/const.h.to(u.eV/u.Hz)
    sigma=McL[1][mask]*1e-18*u.cm**2
    sigma_pd={
        'energy':en_sigma,
        'frequency':freq_sigma,
        'sigma':sigma
    }
    return sigma_pd


def calc_kHM(lambda_array,spectra_lambda,distance,return_sigma=False,return_sigma_only=False):

    '''
    This function computes the HM detachment rate and, if requested, also the cross section.

    Input:
        lambda_array: wavelength array associated with the spectra    [A]
        spectra_lambda: spectra                                       [erg/A/s]
        distance: distance of the radiating source                    [kpc]
    Output:
        photodetachment rate                                          [1/s]
    '''

    sigma_pd=read_sigma()

    if return_sigma_only:
        return sigma_pd

    if spectra_lambda.ndim==1:
        spectra_lambda=np.atleast_2d(spectra_lambda)

    nu_array=lambda2nu(wl=lambda_array)
    spectra_nu=u.quantity.Quantity(value=np.empty_like(spectra_lambda.value),unit=u.erg/u.s/u.Hz)
    for i in range(len(spectra_lambda)):
        spectra_nu[i]=spec_lambda2nu(wl=lambda_array,spec_wl=spectra_lambda[i])

    intensity_nu=spectra_nu/(4.*np.pi*u.sr*4.*np.pi*(distance.to(u.cm))**2)
    units_intensity=intensity_nu.unit
    intensity_nu_HR=np.empty(shape=(len(intensity_nu),len(sigma_pd['frequency'])))
    for i in range(len(intensity_nu)):
        interp=InterpolatedUnivariateSpline(nu_array,intensity_nu[i].value,k=1)
        intensity_nu_HR[i]=interp(sigma_pd['frequency'])
    intensity_nu_HR=intensity_nu_HR*units_intensity

    rate=4.*np.pi*u.sr*np.trapz(sigma_pd['sigma']*intensity_nu_HR/sigma_pd['energy'].to(u.erg),sigma_pd['frequency'])

    if return_sigma:
        return rate,sigma_pd
    else:
        return rate