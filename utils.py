import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import InterpolatedUnivariateSpline


def get_ioniz_energy_hydrogen():

    '''
    Return the ionisation energy of an Hydrogen atom,
    that is very similar to the Rydberg constant for infinite nuclear mass.
    Ryd is originally in [1/m], so it is converted to [eV].
    '''

    ryd_hydrogen = const.Ryd * const.m_p / (const.m_p + const.m_e)
    return ryd_hydrogen * const.c * const.h.to(u.eV * u.s)


def nu2lambda(frequency_array):

    '''
    Convert an array of frequencies [Hz] to an array of wavelengths [A].
    Both arrays are in increasing order, so they are one the mirror of the other.
    '''

    if type(frequency_array) != u.Quantity:
        frequency_array = frequency_array * u.Hz

    speed_of_light = (const.c).to(u.angstrom * u.Hz)
    wavelength_array = speed_of_light / np.atleast_1d(frequency_array)[::-1]
    return wavelength_array


def lambda2nu(wavelength_array):

    '''
    Convert an array of wavelengths [A] to an array of frequencies [Hz].
    Both arrays are in increasing order, so they are one the mirror of the other.
    '''

    if type(wavelength_array) != u.Quantity:
        wavelength_array = wavelength_array * u.angstrom

    speed_of_light = (const.c).to(u.angstrom * u.Hz)
    frequency_array = speed_of_light / np.atleast_1d(wavelength_array)[::-1]
    return frequency_array
    

def spec_nu2lambda(frequency_array, spectrum_freq):

    '''
    Convert spectrum from [erg/s/Hz] to [erg/s/A].
    '''

    if type(frequency_array) != u.Quantity:
        frequency_array = frequency_array * u.Hz
    if type(spectrum_freq) != u.Quantity:
        spectrum_freq = spectrum_freq * u.erg / u.s / u.Hz

    speed_of_light = (const.c).to(u.angstrom * u.Hz)
    spectrum_wl = np.atleast_1d(spectrum_freq * frequency_array**2 / speed_of_light)
    return np.flip(spectrum_wl, axis=spectrum_wl.ndim - 1)


def spec_lambda2nu(wavelength_array, spectrum_wl):

    '''
    Convert spectrum from [erg/s/A] to [erg/s/Hz].
    '''

    if type(wavelength_array) != u.Quantity:
        wavelength_array = wavelength_array * u.angstrom
    if type(spectrum_wl) != u.Quantity:
        spectrum_wl = spectrum_wl * u.erg / u.s / u.angstrom

    speed_of_light = (const.c).to(u.angstrom * u.Hz)
    spectrum_freq = np.atleast_1d(spectrum_wl * wavelength_array**2 / speed_of_light)
    return np.flip(spectrum_freq, axis=spectrum_freq.ndim - 1)


def interpolate_array(old_x_axis, old_y_axis, new_x_axis):

    '''
    Change the resolution of an array.
    Normally used in this code to interpolate spectra to match the resolution of the cross sections.
    Input:
        old_x_axis: normally the x axis of the array, for spectra can be either wavelength or frequency
        old_y_axis: the original array
        new_x_axis: the target resolution
    Output:
        new_y_axis: the new array, with the new resolution
    '''

    if type(old_y_axis) == u.Quantity:
        units = old_y_axis.unit
    else:
        units = None
    
    if old_y_axis.ndim == 1:
        old_y_axis = np.atleast_2d(old_y_axis)
    number_of_spectra = old_y_axis.shape[0]
    new_y_axis = np.empty(shape=(number_of_spectra, len(new_x_axis)))

    for i in range(number_of_spectra):
        interp = InterpolatedUnivariateSpline(old_x_axis, old_y_axis[i].value, k=1, ext=1)
        new_y_axis[i] = interp(new_x_axis)

    if units is not None:
        new_y_axis *= units

    return new_y_axis


def convert_energy_cm2ev(energy_cm):

    '''
    Convert energies from [1/cm] to [eV].
    Input:
        energy_cm: energy, in [1/cm]
    Output:
        energy_ev: energy, in [eV]
    '''

    if type(energy_cm) != u.Quantity:
        energy_cm = energy_cm / u.cm
    energy_ev = energy_cm * (const.h).to(u.eV * u.s) * (const.c).to(u.cm / u.s)
    return energy_ev


def convert_energy_ev2cm(energy_ev):

    '''
    Convert energies from [eV] to [1/cm].
    Input:
        energy_ev: energy, in [eV]
    Output:
        energy_cm: energy, in [1/cm]
    '''

    if type(energy_ev) != u.Quantity:
        energy_ev = energy_ev * u.eV
    energy_cm  = energy_ev / ((const.h).to(u.eV * u.s) * (const.c).to(u.cm / u.s))
    return energy_cm


def convert_energy_cm2k(energy_cm):

    '''
    Convert energies from [1/cm] to equivalent temperature [K].
    Input:
        energy_cm: energy, in [1/cm]
    Output:
        energy_k: equivalent temperature, in [K]
    '''

    if type(energy_cm) != u.Quantity:
        energy_cm = energy_cm / u.cm
    energy_ev = convert_energy_cm2ev(energy_cm)
    energy_k = energy_ev / (const.k_B).to(u.eV / u.K)
    return energy_k


def convert_energy_k2cm(energy_k):

    '''
    Convert energies from equivalent temperature [K] to [1/cm].
    Input:
        energy_k: equivalent temperature, in [K]
    Output:
        energy_cm: energy, in [1/cm]
    '''

    if type(energy_k) != u.Quantity:
        energy_k = energy_k * u.K
    energy_ev = energy_k * (const.k_B).to(u.eV / u.K)
    energy_cm = convert_energy_ev2cm(energy_ev)
    return energy_cm


def generate_blackbody_spectrum(radiation_temperature, energy_array, normalise_spectrum=True, photon_energy_normalisation=None):

    '''
    Produce a black body spectrum with a given radiation temperature,
    normalised at a given energy (default: Lyman limit).
    Input:
        radiation_temperature: the temperature of the black body, in [K]
        energy_array: the array of photon energies, in [eV]
        normalise_spectrum: boolean, whether the spectrum needs to be normalised at a specific photon energy; default: True
        photon_energy_normalisation: photon energy at which the spectrum needs to be normalised, in [eV]
    Output:
        black-body spectrum, expressed as intensity, in [erg/s/Hz/cm^2/sr]
    '''

    if type(radiation_temperature) != u.Quantity:
        radiation_temperature = radiation_temperature * u.K
    if type(energy_array) != u.Quantity:
        energy_array = energy_array * u.eV
    if normalise_spectrum:
        if photon_energy_normalisation is None:
            photon_energy_normalisation = get_ioniz_energy_hydrogen()
        else:
            if type(photon_energy_normalisation) != u.Quantity:
                photon_energy_normalisation = photon_energy_normalisation * u.eV

    frequency_array = energy_array / const.h.to(u.eV / u.Hz)

    monocromatic_intensity_unit = u.erg / u.s / u.Hz / u.sr / u.cm**2

    speed_of_light = (const.c).to(u.cm * u.Hz)
    constant_factor = 2. / u.sr * (const.h).to(u.erg * u.s) / speed_of_light**2

    unnormed_blackbody = constant_factor * frequency_array**3 / (np.exp(energy_array / ((const.k_B).to(u.eV / u.K) * radiation_temperature)) - 1.)
    if not normalise_spectrum:
        return unnormed_blackbody.to(monocromatic_intensity_unit)
    else:
        photon_frequency_normalisation = photon_energy_normalisation / const.h.to(u.eV / u.Hz)
        normalisation = constant_factor * photon_frequency_normalisation**3 / (np.exp(photon_energy_normalisation / ((const.k_B).to(u.eV / u.K) * radiation_temperature)) - 1.)
        normed_blackbody = unnormed_blackbody / normalisation
        return normed_blackbody * monocromatic_intensity_unit



def generate_flat_spectrum(spectrum_length, normalisation_intensity=1):

    '''
    Produce a flat spectrum.
    Input:
        spectrum_length: number of the frequency sampling points
        normalisation_intensity: monochromatic intensity of the spectrum
    Output:
        flat_spectrum: the spectrum, units: [erg/s/Hz/sr/cm^2]
    '''

    monocromatic_intensity_unit = u.erg / u.s / u.Hz / u.sr / u.cm**2
    flat_spectrum = np.ones(shape=spectrum_length) * normalisation_intensity * monocromatic_intensity_unit
    return flat_spectrum