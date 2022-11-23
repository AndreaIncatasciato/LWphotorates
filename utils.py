import numpy as np
from astropy import constants as const
from astropy import units as u


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
    spectrum_wl = np.atleast_1d(spectrum_freq * frequency_array**2 / speed_of_light)[::-1]
    return spectrum_wl


def spec_lambda2nu(wavelength_array, spectrum_wl):

    '''
    Convert spectrum from [erg/s/A] to [erg/s/Hz].
    '''

    if type(wavelength_array) != u.Quantity:
        wavelength_array = wavelength_array * u.angstrom
    if type(spectrum_wl) != u.Quantity:
        spectrum_wl = spectrum_wl * u.erg / u.s / u.angstrom

    speed_of_light = (const.c).to(u.angstrom * u.Hz)
    spectrum_freq = np.atleast_1d(spectrum_wl * wavelength_array**2 / speed_of_light)[::-1]
    return spectrum_freq


def cm2eV(en_cm):

    '''
    Convert energies from [1/cm] to [eV].
    It works with variables both w/ and w/o units
    (in the latter case it assumes that the input units are [1/cm]).
    '''

    if type(en_cm)==u.Quantity:
        en_eV=en_cm*const.h.to(u.eV*u.s)*const.c.to(u.cm/u.s)
    else:
        en_eV=en_cm/u.cm*const.h.to(u.eV*u.s)*const.c.to(u.cm/u.s)
    return en_eV


def eV2cm(en_eV):

    '''
    Convert energies from [eV] to[1/cm].
    It works with variables both w/ and w/o units
    (in the latter case it assumes that the input units are [eV]).
    '''

    if type(en_eV)==u.Quantity:
        en_cm=en_eV/(const.h.to(u.eV*u.s)*const.c.to(u.cm/u.s))
    else:
        en_cm=en_eV*u.eV/(const.h.to(u.eV*u.s)*const.c.to(u.cm/u.s))
    return en_cm


def cm2K(en_cm):

    '''
    Convert energies from [1/cm] to equivalent temperature [K].
    It works with variables both w/ and w/o units
    (in the latter case it assumes that the input units are [1/cm]).
    '''

    en_eV=cm2eV(en_cm)
    en_K=en_eV/const.k_B.to(u.eV/u.K)
    return en_K


def K2cm(en_K):

    '''
    Convert energies from equivalent temperature [K] to [1/cm].
    It works with variables both w/ and w/o units
    (in the latter case it assumes that the input units are [K]).
    '''

    if type(en_K)==u.Quantity:
        en_eV=en_K*const.k_B.to(u.eV/u.K)
    else:
        en_eV=en_K*u.K*const.k_B.to(u.eV/u.K)
    en_cm=en_eV/(const.h.to(u.eV*u.s)*const.c.to(u.cm/u.s))
    return en_cm


def spectrum_BB_norm(Trad,energies,energy_norm=const.Ryd*const.m_p/(const.m_p+const.m_e)*const.h.to(u.eV/u.Hz)*const.c.to(u.m*u.Hz)):

    '''
    Produce a black body spectrum with a given radiation temperature,
    normalised at a given energy (default: Lyman limit).
    Input:
        radiation temperature            [K]
        array of photon energies         [eV]
        normalisation energy             [eV]
    Output:
        array of intensities             [erg/s/Hz/cm^2/sr]
    '''

    if type(Trad)!=u.Quantity:
        Trad=Trad*u.K
    if type(energies)!=u.Quantity:
        Trad=Trad*u.eV
    freq=energies/const.h.to(u.eV/u.Hz)
    bb=((2./u.sr*const.h.to(u.erg/u.Hz)*(freq**3)/(const.c.to(u.cm*u.Hz)**2))/(np.exp(energies/(const.k_B.to(u.eV/u.K)*Trad))-1.)).to(u.erg/u.cm/u.cm/u.Hz/u.s/u.sr)
    norm=1./((2./u.sr*const.h.to(u.erg/u.Hz)*((energy_norm/const.h.to(u.eV/u.Hz))**3)/(const.c.to(u.cm*u.Hz)**2))/(np.exp(energy_norm/(const.k_B.to(u.eV/u.K)*Trad))-1.)).to(u.erg/u.cm/u.cm/u.Hz/u.s/u.sr).value
    bb*=norm
    return bb


def spectrum_flat(energies):

    '''
    Produce a flat spectrum.
    Input:
        array of photon energies         [eV]
    Output:
        array of intensities             [erg/s/Hz/cm^2/sr]
    '''

    if type(energies)!=u.Quantity:
        Trad=Trad*u.eV
    return np.ones_like(energies.value)*u.erg/u.s/u.Hz/u.sr/u.cm**2