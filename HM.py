import os
import numpy as np
from astropy import constants as const
from astropy import units as u

from LWphotorates.utils import get_ioniz_energy_hydrogen, nu2lambda, lambda2nu ,spec_lambda2nu

from scipy.interpolate import InterpolatedUnivariateSpline


def get_cross_section(reference='ML_17'):

    '''
    Get the cross section as a function of the photon energy.
    Four possible references:
      - ML_17: McLaughlin et al. (2017) https://ui.adsabs.harvard.edu/abs/2017JPhB...50k4001M/abstract
      - SK_87: Shapiro & Kang (1987) https://ui.adsabs.harvard.edu/abs/1987ApJ...318...32S/abstract
      - J_88: John (1988) https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract
      - C_07: Chuzhoy et al. (2007) https://ui.adsabs.harvard.edu/abs/2007ApJ...665L..85C/abstract

    Returns:
      dictionary with:
        - photon energy  [eV]
        - photon frequency  [Hz]
        - photon wavelength  [Angstrom]
        - cross section  [cm^2]
    '''

    HM_detach_min_energy = 0.7543 * u.eV
    ioniz_energy_hydrogen = get_ioniz_energy_hydrogen()

    if reference == 'ML_17':

        input_file_path = os.path.join(os.path.dirname(__file__), 'inputdata', 'HM', 'sigma_McLaughlin+2017.txt')

        energy_array, cross_section_array = np.loadtxt(input_file_path, unpack=True)
        energy_array = energy_array * u.eV
        cross_section_array = cross_section_array * 1e-18 * u.cm**2
        mask = energy_array < ioniz_energy_hydrogen

        frequency_array = energy_array / const.h.to(u.eV / u.Hz)
        wavelength_array = nu2lambda(frequency_array)[::-1]

        data_dictionary = {
            'energy': energy_array[mask],
            'frequency': frequency_array[mask],
            'wavelength': wavelength_array[mask],
            'cross_section': cross_section_array[mask]
        }

    elif reference == 'SK_87':

        energy_array = np.linspace(HM_detach_min_energy.value, ioniz_energy_hydrogen.value, int(1e4)) * u.eV
        frequency_array = energy_array / const.h.to(u.eV / u.Hz)
        wavelength_array = nu2lambda(frequency_array)[::-1]

        freq_to_use = frequency_array.value
        cross_section_array = 7.928e5 * u.cm**2 * (freq_to_use - freq_to_use[0])**1.5 / freq_to_use**3

        data_dictionary = {
            'energy': energy_array,
            'frequency': frequency_array,
            'wavelength': wavelength_array,
            'cross_section': cross_section_array
        }

    elif reference == 'J_88':

        energy_array = np.linspace(HM_detach_min_energy.value, ioniz_energy_hydrogen.value, int(1e4)) * u.eV
        frequency_array = energy_array / const.h.to(u.eV / u.Hz)
        wavelength_array = nu2lambda(frequency_array)[::-1]

        coefficients = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])
        wl_to_use = (wavelength_array.to(u.micron)).value
        fit_value = [coefficients[i] * (1. / wl_to_use - 1. / wl_to_use[0])**(i / 2.) for i in range(len(coefficients))]
        fit_value = np.array(fit_value).sum(axis=0)

        cross_section_array = 1e-18 * u.cm**2 * fit_value * wl_to_use**3 * (1. / wl_to_use - 1. / wl_to_use[0])**1.5

        data_dictionary = {
            'energy': energy_array,
            'frequency': frequency_array,
            'wavelength': wavelength_array,
            'cross_section': cross_section_array
        }

    elif reference == 'C_07':

        energy_array = np.linspace(HM_detach_min_energy.value, ioniz_energy_hydrogen.value, int(1e4)) * u.eV
        frequency_array = energy_array / const.h.to(u.eV / u.Hz)
        wavelength_array = nu2lambda(frequency_array)[::-1]

        energy_to_use = energy_array.value
        cross_section_array = 2.1e-16 * u.cm**2 * (energy_to_use - energy_to_use[0])**1.5 / energy_to_use**3.11

        data_dictionary = {
            'energy': energy_array,
            'frequency': frequency_array,
            'wavelength': wavelength_array,
            'cross_section': cross_section_array
        }

    return data_dictionary






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