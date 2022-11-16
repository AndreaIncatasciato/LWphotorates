import numpy as np
from astropy import constants as const
from astropy import units as u


def nu2lambda(freq):

    '''
    Convert frequency [Hz] to wavelength [A].
    '''

    wl=(const.c.to(u.angstrom/u.s)/np.atleast_1d(freq)[::-1]).to(u.angstrom)
    return wl


def lambda2nu(wl):

    '''
    Convert wavelength [A] to frequency [Hz].
    '''

    freq=(const.c.to(u.angstrom/u.s)/np.atleast_1d(wl)[::-1]).to(u.Hz)
    return freq


def spec_nu2lambda(freq,spec_freq):

    '''
    Convert spectrum from [erg/Hz/s] to [erg/A/s].
    '''

    spec_wl=np.atleast_1d(spec_freq*freq**2/const.c.to(u.angstrom/u.s))[::-1].to(u.erg/u.s/u.angstrom)
    return spec_wl


def spec_lambda2nu(wl,spec_wl):

    '''
    Convert spectrum from [erg/A/s] to [erg/Hz/s].
    '''

    spec_freq=np.atleast_1d(spec_wl*wl**2/const.c.to(u.angstrom/u.s))[::-1].to(u.erg/u.s/u.Hz)
    return spec_freq


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