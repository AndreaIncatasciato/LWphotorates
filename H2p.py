import os
import numpy as np
import h5py
from astropy import constants as const
from astropy import units as u

from scipy.interpolate import InterpolatedUnivariateSpline

from LWphotorates.utils import nu2lambda, lambda2nu
from LWphotorates.utils import convert_energy_cm2ev, convert_energy_ev2cm, convert_energy_cm2k


def get_ground_states_data():

    '''
    Get the H2p rotovibrational levels UGAMOP data of the electronic ground state.
    Output:
        ground_states_data: dictionary with:
            'v': vibrational quantum number
            'J': rotational quantum number
            'eV': level energy in [eV]
            'cm': level energy in [1/cm]
            'K': equivalent temperature wrt the (0, 0) level (useful for the LTE partition function)
    '''

    energy_atomic_units = 2. * (const.Ryd) * (const.c).to(u.m * u.Hz) * (const.h).to(u.eV / u.Hz)

    ground_states_path = os.path.join(os.path.dirname(__file__), 'inputdata', 'H2p', 'Xen.txt')
    ground_states_file = np.loadtxt(ground_states_path, unpack=True)
    ground_states_data = {
        'v': np.array(ground_states_file[0], dtype=np.int16),
        'J': np.array(ground_states_file[1], dtype=np.int16),
        'eV': ground_states_file[2] * energy_atomic_units
    }
    ground_states_data['cm'] = convert_energy_ev2cm(ground_states_data['eV'])
    ground_states_data['K'] = convert_energy_cm2k(ground_states_data['cm'][0] - ground_states_data['cm'])
    
    return ground_states_data


def get_cross_section_zammit():

    '''
    Get the H2p (rotovibrational level resolved) cross sections from Zammit et al. (2017),
    together with the kinetic energy released in the gas.
    Reference: https://ui.adsabs.harvard.edu/abs/2017ApJ...851...64Z/abstract
    Output:
        ground_states_data: dictionary with:

            'v': vibrational quantum number
            'J': rotational quantum number
            'eV': level energy in [eV]
            'cm': level energy in [1/cm]
            'K': equivalent temperature wrt the (0, 0) level (useful for the LTE partition function)

            'photon_energy': photon energy in [eV]
            'photon_wl': photon wavelength in [A]
            'cross_section': photodissociation cross section in [cm^2]
            'heating_cross_section': heating cross section in [eV*cm^2]
    '''

    ground_states_data = get_ground_states_data()

    cross_section_path = os.path.join(os.path.dirname(__file__), 'inputdata', 'H2p', 'Zammit2017.hdf5')

    with h5py.File(name=cross_section_path, mode='r') as cross_section_file:
        ground_states_data['photon_wl'] = cross_section_file['phot_wl'][:] * u.Angstrom
        ground_states_data['photon_energy'] = cross_section_file['phot_energy'][:] * u.eV
        ground_states_data['cross_section'] = cross_section_file['sigma'][:] * u.cm**2
        ground_states_data['heating_cross_section'] = cross_section_file['sigma*en'][:] * u.eV * u.cm**2

    return ground_states_data


def get_cross_section_babb(wavelength_array):

    '''
    Get the H2p (rotovibrational level resolved) cross sections from Babb (2015),
    together with the kinetic energy released in the gas.
    Reference: https://ui.adsabs.harvard.edu/abs/2015ApJS..216...21B/abstract
    NB: they included only a subset of all the rovib levels of the electronic ground state,
    so here we are including only data relative to those levels.
    Output:
        ground_states_data: dictionary with:

            'v': vibrational quantum number
            'J': rotational quantum number
            'eV': level energy in [eV]
            'cm': level energy in [1/cm]
            'K': equivalent temperature wrt the (0, 0) level (useful for the LTE partition function)

            'photon_energy': photon energy in [eV]
            'photon_wl': photon wavelength in [A]
            'cross_section': photodissociation cross section in [cm^2]
            'heating_cross_section': heating cross section in [eV*cm^2]
    '''

    energy_atomic_units = 2. * (const.Ryd) * (const.c).to(u.m * u.Hz) * (const.h).to(u.eV / u.Hz)

    ground_states_data = get_ground_states_data()

    cross_section_path = os.path.join(os.path.dirname(__file__), 'inputdata', 'H2p', 'Babb2015.txt')
    cross_section_file = np.genfromtxt(cross_section_path, dtype={
            'names': ('v', 'J', 'kinetic_energy', 'rovib_level_energy', 'photon_wl', 'Msquared'),
            'formats':('i4', 'i4', 'f8', 'f8', 'f8', 'f8')}, unpack=False)

    ground_states_data['photon_wl'] = wavelength_array
    ground_states_data['photon_energy'] = lambda2nu(wavelength_array)[::-1] * (const.h).to(u.eV / u.Hz)
    ground_states_data['cross_section'] = np.zeros(shape=(0, len(wavelength_array)))
    ground_states_data['heating_cross_section'] = np.zeros(shape=(0, len(wavelength_array)))

    # Babb dataset contains data only for a subset of the rotovib states of the electronic ground state (337 out of 423)
    # the strategy here is to eliminate the states for which we don't have data
    index_levels_no_data = []
    for i in range(len(ground_states_data['eV'])):
        vibr_quantum_number = ground_states_data['v'][i]
        rot_quantum_number = ground_states_data['J'][i]
        mask = (cross_section_file['v'] == vibr_quantum_number) & (cross_section_file['J'] == rot_quantum_number)
        
        if mask.sum():
            original_wl = ((cross_section_file['photon_wl'][mask]) * u.nm).to(u.Angstrom)
            original_cs = 2.689e-18 * u.cm**2 * cross_section_file['Msquared'][mask] * 45.563 / cross_section_file['photon_wl'][mask]
            original_heating_cs = cross_section_file['kinetic_energy'][mask] * energy_atomic_units
            
            interp_cs = InterpolatedUnivariateSpline(x=original_wl, y=original_cs, k=1)
            new_cs = interp_cs(ground_states_data['photon_wl'], ext='zeros')
            interp_heating_cs = InterpolatedUnivariateSpline(x=original_wl, y=original_heating_cs, k=1)
            new_heating_cs = interp_heating_cs(ground_states_data['photon_wl'], ext='zeros') * new_cs
            
            ground_states_data['cross_section'] = np.concatenate(
                (ground_states_data['cross_section'], np.atleast_2d(new_cs)),
                axis=0)
            ground_states_data['heating_cross_section'] = np.concatenate(
                (ground_states_data['heating_cross_section'], np.atleast_2d(new_heating_cs)),
                axis=0)

        else:
            index_levels_no_data.append(i)

    for key in ['v', 'J', 'eV', 'cm', 'K']:
        ground_states_data[key] = np.delete(ground_states_data[key], index_levels_no_data)

    ground_states_data['cross_section'] *= u.cm**2
    ground_states_data['heating_cross_section'] *= (u.eV * u.cm**2)

    return ground_states_data


def calculate_partition_function(gas_temperature, ground_states_data=None, normalised=True):

    '''
    Calculate the partition function of the electronic ground state X rovib levels in the LTE limit.
    It depends on the gas temperature and
    by default it is normalised such that the sum over all the possible levels is 1.
    
    Input:
        gas_temperature: gas temperature in [K]
        ground_states_data: dictionary with the electronic ground state X rovib levels
            (if None, the default UGAMOP database will be read beforehand)
        normalised: boolean, if True the sum over all the possible levels is 1
    Output:
        partition_function: array with LTE Boltzmann coefficients that represent the partition function
    '''

    if type(gas_temperature) != u.Quantity:
        gas_temperature = gas_temperature * u.K

    if ground_states_data is None:
        ground_states_data = get_ground_states_data()

    partition_function = (2. - (-1.)**ground_states_data['J']) / 2. * (2. * ground_states_data['J'] + 1.) * np.exp(-ground_states_data['K'] / gas_temperature)

    if normalised:
        return partition_function / partition_function.sum()
    else:
        return partition_function


def calculate_critical_density(gas_temperature):

    '''
    Compute the critical density for the LTE limit as proposed in Glover 2015 appendix B1.2:
    https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.2082G/abstract.
    Assume a neutral gas with standard composition.
    Main colliders: H and free electrons (see Glover & Savin, 2009: https://ui.adsabs.harvard.edu/abs/2009MNRAS.393..911G/abstract).
    Input:
        gas_temperature: the gas temperature, in [K]
    Output:
        critical_density: the critical density for H2+, in [1/cm^3]
    '''

    if type(gas_temperature) != u.Quantity:
        gas_temperature = gas_temperature * u.K

# assume composition, all numbers normalised to the number density of H atoms (H + H2)
    He_number_fraction = 0.08
    H2_number_fraction = 1e-4
    H_number_fraction = 1. - 2. * H2_number_fraction
    em_number_fraction = 0.01

# to calculate the critical densities assume the main colliders are H and free electrons
# (see Glover & Savin 2009 and Glover 2015)
    critical_density_H = 400. * (gas_temperature / (1e4 * u.K))**(-1) * u.cm**-3
    critical_density_em = 50. * u.cm**-3
    critical_density = ((H_number_fraction + em_number_fraction) * (H_number_fraction / critical_density_H + em_number_fraction / critical_density_em)**(-1))
    return critical_density


def calc_sigma(
    ngas,
    Tgas,
    wl_new,
    Dunn,
    cross_section_reference='Z_17'
    ):

    '''
    Given gas density and temperature, calculate the effective cross section to compute the dissociation rate and the heating rate.
    Interpolate between GS and LTE limit.
    Input:
        ngas: gas number density                                           [1/cm^3]
        Tgas: gas temperature                                              [K]
        wavelength: array of wavelengths for the integration               [A]
        Dunn: boolean, True if you want to use the Franck-Condon distrubution of level population [v=0-18,J=1]
        Zammit: boolean, True to use the more updated cross section db from Zammit+2017, False to use Babb2015
    Output:
        sigma: effective cross section, same size as wavelength            [cm^2]
        heat: effective 'heating' cross section, same size as wavelength   [cm^2 eV]
    '''

    if cross_section_reference == 'Z_17':
        ground_states_data = get_cross_section_zammit()
    elif cross_section_reference == 'B_15':
        ground_states_data = get_cross_section_babb(wavelength_array=wl_new)

    if Dunn:
        FCpop=np.array([0.08964,0.16013,0.17616,0.15592,0.12281,0.09052,0.06423,0.04465,0.03074,0.02111,0.01451,0.01002,0.00694,0.00480,0.00329,0.00221,0.00139,0.00072,0.00018])
        sigma_FC=np.dot(np.atleast_2d(FCpop),sigma_pd['sigma'][Xen['J']==1][:-1])[0]
        heat_FC=np.dot(np.atleast_2d(FCpop),sigma_pd['heat_sigma'][Xen['J']==1][:-1])[0]
        return sigma_FC,heat_FC

    else:
        sigma_GS=sigma_pd['sigma'][0]
        heat_GS=sigma_pd['heat_sigma'][0]
        NX=Xpop(Xen=Xen,Tgas=Tgas)
        sigma_LTE=np.dot(np.atleast_2d(NX),sigma_pd['sigma'])[0]
        heat_LTE=np.dot(np.atleast_2d(NX),sigma_pd['heat_sigma'])[0]
        ncrit=calc_ncrit(Tgas=Tgas)
        alpha=(1+ngas/ncrit)**(-1)
        sigma=sigma_LTE*(sigma_GS/sigma_LTE)**alpha
        heat=heat_LTE*(heat_GS/heat_LTE)**alpha
        return sigma,heat


def calc_kH2p(
    lambda_array,
    spectra_lambda,
    distance,
    ngas,
    Tgas,
    Dunn=False,
    cross_section_reference='Z_17',
    return_sigma=False,
    return_heating=False,
    return_sigma_only=False,
    return_heating_only=False
    ):

    '''
    Calculate the photodissociation rate of H2p its associated heating rate.
    One rate for each spectrum. Critical density is assumed for a neutral gas with standard composition.
    Input:
        lambda_array: wavelength array associated with the spectra    [A]
        spectra_lambda: spectra                                       [erg/A/s]
        distance: distance of the radiating source                    [kpc]
        ngas: gas number density                                      [1/cm^3]
        Tgas: gas temperature                                         [K]
        Dunn: boolean, True if you want to use the Franck-Condon distrubution of level population [v=0-18,J=1]
        Zammit: boolean, True to use the more updated cross section db from Zammit+2017, False to use Babb2015
        return_sigma: the function will return diss rate, heating rate and the effective cross section
        return_heating: the function will return diss rate, heating rate and the monochromatic heating rate
        return_sigma_only: the function will return the effective cross section without calculating the diss rate
        return_heating_only: the function will return the monochromatic heating rate without calculating the diss rate
    Output:
        dissociation rate and heating rate, both interpolated between GS and LTE limits
        effective cross section
        monochromatic heating rate
        high-resolution wavelength array                              [A]
    '''

    Lyman_lim=(const.h*const.c*const.Ryd).to(u.eV)*const.m_p/(const.m_p+const.m_e)
    minEn=0.1*u.eV

    min_wl=nu2lambda(Lyman_lim/const.h.to(u.eV/u.Hz))[0]
    max_wl=nu2lambda(minEn/const.h.to(u.eV/u.Hz))[0]

    wl_new=(np.logspace(start=np.log10(min_wl.value),stop=np.log10(max_wl.value),num=int(1e5))*u.angstrom)
    en_new=lambda2nu(wl_new)*const.h.to(u.eV/u.Hz)

    sigma, heating = calc_sigma(
        ngas=ngas,
        Tgas=Tgas,
        wl_new=wl_new,
        Dunn=Dunn,
        cross_section_reference=cross_section_reference
        )

    if return_sigma_only:
        return sigma,wl_new
    if return_heating_only:
        return heating,wl_new

    if spectra_lambda.ndim==1:
        spectra_lambda=np.atleast_2d(spectra_lambda)

    intensity_lambda=spectra_lambda/(4*np.pi*u.sr*4*np.pi*(distance.to(u.cm))**2)
    units_intensity=intensity_lambda.unit
    intensity_lambda_HR=np.empty(shape=(len(intensity_lambda),len(wl_new)))
    for i in range(len(intensity_lambda)):
        interpspectrum=InterpolatedUnivariateSpline(lambda_array,intensity_lambda[i].value,k=1)
        intensity_lambda_HR[i]=interpspectrum(wl_new)
    intensity_lambda_HR=intensity_lambda_HR*units_intensity

    diss_rate=4.*np.pi*u.sr/(const.h.to(u.erg*u.s)*const.c.to(u.angstrom/u.s))*np.trapz(intensity_lambda_HR*wl_new*sigma,wl_new)
    heat_rate=4.*np.pi*u.sr/(const.h.to(u.erg*u.s)*const.c.to(u.angstrom/u.s))*np.trapz(intensity_lambda_HR*wl_new*heating,wl_new)

    if return_sigma&return_heating:
        return diss_rate,heat_rate,sigma,heating,wl_new
    elif return_sigma:
        return diss_rate,heat_rate,sigma,wl_new
    elif return_heating:
        return diss_rate,heat_rate,heating,wl_new
    else:
        return diss_rate,heat_rate