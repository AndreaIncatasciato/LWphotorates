import numpy as __np  # why this? so it doesn't show np when doing 'import ComputeRate'
from astropy import units as __u

from LWphotorates import H2
from LWphotorates import HM
from LWphotorates import H2p


# when doing from ComputeRate import * it will only these
__all__ = ['compute_H2_diss_rate', 'compute_HM_detach_rate', 'compute_H2p_diss_rate', 'utils', 'H2', 'HM', 'H2p']


units_monochromatic_luminosity_wl = __u.erg / __u.s / __u.angstrom
default_distance = 1. * __u.kpc
default_density = 1e2 * __u.cm**-3
default_temperature = 1e3 * __u.K



def compute_HM_detach_rate(
    wavelength_array,
    spectra_wl,
    distance=default_distance,
    cross_section_reference='ML_17'
):
    
    '''
    Calculate the photodetachment rate of HM for a given set of spectra.
    One rate for each spectrum. This is just a wrapper of the main function.
    Input:
        wavelength_array: wavelength array associated with the spectra in [A]
        spectra_wl: spectra, as monochromatic luminosity in [erg/A/s]
        distance: distance of the radiating source in [kpc]
        cross_section_reference: cross section to use, possible choices ['ML_17', 'SK_87', 'J_88', 'C_07']
    Output:
        detachment_rate: detachment rate in [1/s]
    '''
    
    # checks on units
    if type(wavelength_array) != __u.Quantity:
        wavelength_array = wavelength_array * __u.angstrom
    if type(spectra_wl) != __u.Quantity:
        spectra_wl = spectra_wl * units_monochromatic_luminosity_wl
    if type(distance) != __u.Quantity:
        distance = distance * __u.kpc

    if distance.value <= 0:
        print('Please provide a strictly positive distance.')
        return -1
    
    # if everything seems reasonable let's move on
    return HM.calculate_kHM(
        wavelength_array=wavelength_array,
        spectra_wl=spectra_wl,
        distance=distance,
        cross_section_reference=cross_section_reference
    )


def compute_H2p_diss_rate(
    wavelength_array,
    spectra_wl,
    distance=default_distance,
    gas_density=default_density,
    gas_temperature=default_temperature,
    cross_section_reference='Z_17'
):
    
    '''
    Calculate the photodissociation rate of H2p and the corresponding heating rate for a given set of spectra.
    The rate is interpolated between the low density limit (where only the rotovibrational ground level is populated)
    and the LTE limit, as in Glover (2015): https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.2082G/abstract.
    The critical density for LTE is assumed as for a neutral gas with standard composition.
    One rate for each spectrum. This is just a wrapper of the main function.
    Input:
        wavelength_array: wavelength array associated with the spectra in [A]
        spectra_wl: spectra, as monochromatic luminosity in [erg/A/s]
        distance: distance of the radiating source in [kpc]
        cross_section_reference: cross section to use, possible choices ['Z_17', 'B_15']
        gas_density: gas number density in [1/cm^3]
        gas_temperature: gas temperature in [K]
    Output:
        dissociation_rate: dissociation rate in [1/s]
        heating_rate: heating rate in [eV/s]
    '''
    
    # checks on units
    if type(wavelength_array) != __u.Quantity:
        wavelength_array = wavelength_array * __u.angstrom
    if type(spectra_wl) != __u.Quantity:
        spectra_wl = spectra_wl * units_monochromatic_luminosity_wl
    if type(distance) != __u.Quantity:
        distance = distance * __u.kpc
    if type(gas_density) != __u.Quantity:
        gas_density = gas_density * __u.cm**-3
    if type(gas_temperature) != __u.Quantity:
        gas_temperature = gas_temperature * __u.K

    if distance.value <= 0:
        print('Please provide a strictly positive distance.')
        return -1
    if gas_density.value <= 0:
        print('Please provide a strictly positive gas number density.')
        return -1
    if gas_temperature.value <= 0:
        print('Please provide a strictly positive gas temperature.')
        return -1
    
    # if everything seems reasonable let's move on
    return H2p.calculate_kH2p(
        wavelength_array=wavelength_array,
        spectra_wl=spectra_wl,
        distance=distance,
        gas_density=gas_density,
        gas_temperature=gas_temperature,
        cross_section_reference=cross_section_reference
    )


def compute_H2_diss_rate(
    wavelength_array,
    spectra_wl,
    distance=default_distance,
    gas_density=default_density,
    gas_temperature=default_temperature,
    excited_states_to_use='LW',
    lw_transitions_reference='U_19+S_15',
    line_profile_flag='V',
    min_partition_function=None,
    min_osc_strength_x_diss_fraction=None,
    min_osc_strength=None,
    min_diss_fraction=None
):

    '''
    Calculate the photodissociation rate of H2 and the corresponding heating rate.
    The rate is interpolated between the low density limit (where the partition function is determined with the Frigus code)
    and the LTE limit, as in Glover (2015): https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.2082G/abstract.
    The critical density for LTE is determined with a fitting function, valid between 1e2 K and 1e4 K.
    One rate for each spectrum. This is just a wrapper of the main function.
    Input:
        wavelength_array: wavelength array associated with the spectra in [A]
        spectra_wl: spectra, as monochromatic luminosity in [erg/A/s]
        distance: distance of the radiating source in [kpc]
        gas_density: gas number density in [1/cm^3]
        gas_temperature: gas temperature in [K]
        excited_states_to_use: which excited electronic states are taken into account;
            available choices are ['B', 'C', 'LW', 'additional', 'all']:
                'B': only B+ states (the 'Lyman' lines), available for all the datasets
                'C': only C+ and C- states (the 'Werner' lines), available for all the datasets
                'LW': B+/C+/C- states (all the 'Lyman-Werner' lines), available for all the datasets
                'additional': only B'+/D+/D- states (usually neglected), available only for Abgrall et al. (1994)
                'all': all six states (B+/C+/C-/B'+/D+/D-), available only for Abgrall et al. (1994)
        lw_transitions_reference: which database is employed;
            available choices are ['A_94', 'U_19', 'U_19+S_15']:
                'A_94': the most complete and widely used database
                    Abgrall et al. (1993a) https://ui.adsabs.harvard.edu/abs/1993A%26AS..101..273A/abstract
                    Abgrall et al. (1993b) https://ui.adsabs.harvard.edu/abs/1993A%26AS..101..323A/abstract
                    Abgrall et al. (1993c) https://ui.adsabs.harvard.edu/abs/1994CaJPh..72..856A/abstract
                    Abgrall et al. (1994) https://ui.adsabs.harvard.edu/abs/1993JMoSp.157..512A/abstract
                    Abgrall et al. (2000) https://ui.adsabs.harvard.edu/abs/2000A%26AS..141..297A/abstract
                'U_19': Ubachs et al. (2019) https://ui.adsabs.harvard.edu/abs/2019A%26A...622A.127U/abstract
                    B+/C+/C- states, transitions only from ground state levels with v=0, J=0-7,
                    recently updated and corrected; appropriate at low temperatures
                'U_19+S_15': join U_19 with Salumbides et al. (2015) https://ui.adsabs.harvard.edu/abs/2015MNRAS.450.1237S/abstract
                    B+/C+/C- states, complementary to U_19, recently updated and corrected
        line_profile_flag: profile of the LW lines:
            'L' for Lorentzian (only natural broadening) or 'V' for Voigt (natural + thermal broadening)
        min_partition_function: minimum level population to take into account a X rovib level,
            default value: None, suggested value: [1e-5 - 1e-3]
        min_osc_strength_x_diss_fraction: minimum value for 'f' * 'frac_diss'
            (oscillator strength and fraction of molecules that dissociate after the excitation)
            default value: None, suggested value: 1e-4
        min_osc_strength: minimum value for 'f' (oscillator strength)
            default value: None, suggested value: 1e-3
        min_diss_fraction: minimum value for 'frac_diss' (fraction of molecules that dissociate after the excitation)
            default value: None, suggested value: 1e-2
    Output:
        dissociation_rate: dissociation rate in [1/s]
        heating_rate: heating rate in [eV/s]
    '''

    # checks on units
    if type(wavelength_array) != __u.Quantity:
        wavelength_array = wavelength_array * __u.angstrom
    if type(spectra_wl) != __u.Quantity:
        spectra_wl = spectra_wl * units_monochromatic_luminosity_wl
    if type(distance) != __u.Quantity:
        distance = distance * __u.kpc
    if type(gas_density) != __u.Quantity:
        gas_density = gas_density * __u.cm**-3
    if type(gas_temperature) != __u.Quantity:
        gas_temperature = gas_temperature * __u.K

    if distance.value <= 0:
        print('Please provide a strictly positive distance.')
        return -1
    if gas_density.value <= 0:
        print('Please provide a strictly positive gas number density.')
        return -1
    if gas_temperature.value <= 0:
        print('Please provide a strictly positive gas temperature.')
        return -1
    
    # if everything seems reasonable let's move on
    return H2.calculate_kH2(
        wavelength_array=wavelength_array,
        spectra_wl=spectra_wl,
        distance=distance,
        gas_density=gas_density,
        gas_temperature=gas_temperature,
        excited_states_to_use=excited_states_to_use,
        lw_transitions_reference=lw_transitions_reference,
        line_profile_flag=line_profile_flag,
        min_partition_function=min_partition_function,
        min_osc_strength_x_diss_fraction=min_osc_strength_x_diss_fraction,
        min_osc_strength=min_osc_strength,
        min_diss_fraction=min_diss_fraction
    )
