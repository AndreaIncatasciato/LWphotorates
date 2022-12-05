import os
import numpy as np
from astropy import constants as const
from astropy import units as u

from LWphotorates.utils import get_ioniz_energy_hydrogen, nu2lambda, lambda2nu, spec_lambda2nu, interpolate_array


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



def calculate_kHM(
    wavelength_array,
    spectra_wl,
    distance,
    cross_section_reference='ML_17'
):

    '''
    Compute the photodetachment rate of HM.
    Input:
        wavelength_array: wavelength array associated with the spectra in [A]
        spectra_wl: spectra, as monochromatic luminosity in [erg/A/s]
        distance: distance of the radiating source in [kpc]
        cross_section_reference: cross section to use, possible choices ['ML_17', 'SK_87', 'J_88', 'C_07']
    Output:
        detachment_rate: detachment rate in [1/s]
    '''

    cross_section = get_cross_section(reference=cross_section_reference)

    # check on dimension, so we can cycle over the first dimension
    if spectra_wl.ndim == 1:
        spectra_wl = np.atleast_2d(spectra_wl)
    number_of_spectra = spectra_wl.shape[0]

    units_monochromatic_luminosity_wl = u.erg / u.s / u.angstrom
    units_monochromatic_luminosity_freq = u.erg / u.s / u.Hz
    units_monochromatic_intensity_freq = units_monochromatic_luminosity_freq / u.cm**2 / u.sr

    frequency_array = lambda2nu(wavelength_array)
    spectra_freq = u.quantity.Quantity(value=np.empty_like(spectra_wl.value), unit=units_monochromatic_luminosity_freq)

    for i in range(number_of_spectra):
        spectra_freq[i] = spec_lambda2nu(wavelength_array, spectra_wl[i])

    solid_angle = 4. * np.pi * u.sr
    surface_area = 4. * np.pi * (distance.to(u.cm))**2
    intensity_freq = spectra_freq / solid_angle / surface_area

    new_intensity_freq = interpolate_array(
        old_x_axis=frequency_array,
        old_y_axis=intensity_freq,
        new_x_axis=cross_section['frequency']
    )

    integration_x_axis = cross_section['frequency']
    integration_y_axis = cross_section['cross_section'] * new_intensity_freq / cross_section['energy'].to(u.erg)
    dissociation_rate = solid_angle * np.trapz(integration_y_axis, integration_x_axis)

    return dissociation_rate