import os
import numpy as np
import h5py
from astropy import constants as const
from astropy import units as u

from scipy.interpolate import InterpolatedUnivariateSpline

from LWphotorates.utils import nu2lambda, lambda2nu, spec_lambda2nu
from LWphotorates.utils import convert_energy_cm2ev, convert_energy_ev2cm, convert_energy_cm2k, get_ioniz_energy_hydrogen, interpolate_array

from frigus.readers.dataset import DataLoader
from frigus.readers.read_energy_levels import read_levels_lique
from frigus.population import population_density_at_steady_state
from astropy.modeling.models import Voigt1D, Lorentz1D


def get_ground_states_data():

    '''
    Get the H2 rotovibrational levels data of the electronic ground state
    from Komasa et al. (2011): https://ui.adsabs.harvard.edu/abs/2011JCTC....7.3105K/abstract
    Output:
        ground_states_data: dictionary with:
            'v': vibrational quantum number
            'J': rotational quantum number
            'eV': level energy in [eV]
            'cm': level energy in [1/cm]
            'K': equivalent temperature wrt the (0, 0) level (useful for the LTE partition function)
    '''

    ground_states_path = os.path.join(os.path.dirname(__file__), 'inputdata', 'H2', 'Xgroundstate', 'vibrotXenergy_Komasa+2011.txt')
    ground_states_file = np.loadtxt(ground_states_path, unpack=True)
    ground_states_data = {
        'v': np.array(ground_states_file[0], dtype=np.int16),
        'J': np.array(ground_states_file[1], dtype=np.int16),
        'cm': ground_states_file[2] / u.cm
    }
    ground_states_data['eV'] = convert_energy_cm2ev(ground_states_data['cm'])
    ground_states_data['K'] = convert_energy_cm2k(ground_states_data['cm'][0] - ground_states_data['cm'])

    return ground_states_data


def calculate_partition_function(gas_temperature, ground_states_data=None, normalised=True):

    '''
    Calculate the partition function of the electronic ground state X rovib levels in the LTE limit.
    It depends on the gas temperature and
    by default it is normalised such that the sum over all the possible levels is 1.
    
    Input:
        gas_temperature: gas temperature in [K]
        ground_states_data: dictionary with the electronic ground state X rovib levels
            (if None, the default Komasa database will be read beforehand)
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


def call_to_frigus(gas_density, gas_temperature, redshift_for_cmb=15.):

    '''
    In the low density regime LTE approximation is not valid anymore.
    Instead of considering only ground states of ortho and para-H2, we use the Frigus code:
    code repo: https://github.com/mherkazandjian/frigus
    installation: https://pypi.org/project/frigus/
    paper: https://ui.adsabs.harvard.edu/abs/2019RLSFN..30..707C/abstract
    The 58 most important rotovibrational levels are employed.
    The steady-state solution should be accurate enough.
    Input:
        gas_density: gas number density in [1/cm^3]
        gas_temperature: gas temperature in [K]
    Output:
        frigus_ground_states_data: dictionary with the rovib levels used in Frigus,
            same structure as the other dictionary
        frigus_partition_function: partition function determined by Frigus, in the low density regime
    '''

    frigus_ground_states_path = os.path.join(os.environ['FRIGUS_DATADIR_ROOT'], 'H2Xvjlevels_francois_mod.cs')
    frigus_ground_states_file = read_levels_lique(frigus_ground_states_path).data
    frigus_ground_states_data = {
        'v': np.array(frigus_ground_states_file['v'].tolist()),
        'J': np.array(frigus_ground_states_file['j'].tolist()),
        'eV': frigus_ground_states_file['E']
    }
    frigus_ground_states_data['cm'] = convert_energy_ev2cm(frigus_ground_states_data['eV'])
    frigus_ground_states_data['K'] = convert_energy_cm2k(frigus_ground_states_data['cm'] - frigus_ground_states_data['cm'][0])

    # CMB is only important at very low density when collisions are extremely rare
    redshift = redshift_for_cmb
    cmb_temperature = 2.72548 * u.K * (1. + redshift)
    # limit of validity of Frigus code: maximum temperature = 5e3K
    frigus_partition_function = population_density_at_steady_state(
        data_set=DataLoader().load('H2_lique'),
        t_kin=np.minimum(gas_temperature, 5e3 * u.K),
        t_rad=cmb_temperature,
        collider_density=gas_density.si
    )
    # add nuclear spin parity
    frigus_partition_function = frigus_partition_function.flatten() * (2. - (-1)**frigus_ground_states_data['J']) / 2.
    # normalise
    frigus_partition_function /= frigus_partition_function.sum()
    return frigus_ground_states_data, frigus_partition_function


def append_lw_transitions(lw_transitions_dictionary, new_file_path):

    '''
    Append a list of LW transitions to the general dictionary.
    Input:
        lw_transitions_dictionary: the dictionary of all LW transitions to consider
        new_file_path: path to the hdf5 file with the LW transitions to add
    '''

    with h5py.File(new_file_path, mode='r') as lw_transitions_to_add:
        for i in list(lw_transitions_dictionary.keys()):
            lw_transitions_dictionary[i] = np.concatenate((lw_transitions_dictionary[i], lw_transitions_to_add[i]))


def get_lw_transitions(excited_states_to_use, lw_transitions_reference):

    '''
    This function reads the LW transitions databases and saves everything in a dictionary.
    Input:
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
    Output:
        lw_transitions_dictionary: dictionary with the following data for each transition:
            VL: vibrational quantum number of the electronic ground state
            JL: rotational quantum number of the electronic ground state
            VU: vibrational quantum number of the excited state
            JU: rotational quantum number of the excited state
            wl: wavelength of the transition, in [A]
            freq: frequency of the transition, in [Hz]
            f: oscillator strength
            Gamma: natural broadening parameter in [1/s]
            frac_diss: fraction of excited molecules that will dissociate
            mean_Ekin: mean kinetic energy of the H atoms (heating of the gas per molecule), in [eV]
    '''

    # the structure of the dictionary cannot be changed, as it is the same as the hdf5 files
    lw_transitions_dictionary = {
        'VL': np.array([]),
        'JL': np.array([]),
        'VU': np.array([]),
        'JU': np.array([]),
        'wl': np.array([]),
        'f': np.array([]),
        'Gamma': np.array([]),
        'frac_diss': np.array([]),
        'mean_Ekin': np.array([])    
    }

    data_folder = os.path.join(os.path.dirname(__file__), 'inputdata', 'H2', 'transitions', 'cleaned')

    if (excited_states_to_use == 'B') | (excited_states_to_use == 'LW'):
        if (lw_transitions_reference == 'U_19') | (lw_transitions_reference == 'U_19+S_15'):
            data_path = os.path.join(data_folder, 'Bp_Ubachs+2019.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            if lw_transitions_reference == 'U_19+S_15':
                data_path = os.path.join(data_folder, 'Bp_Salumbides+2015.hdf5')
                append_lw_transitions(lw_transitions_dictionary, data_path)
        elif lw_transitions_reference == 'A_94':
            data_path = os.path.join(data_folder, 'Bp_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
    if (excited_states_to_use == 'C') | (excited_states_to_use == 'LW'):
        if (lw_transitions_reference == 'U_19') | (lw_transitions_reference == 'U_19+S_15'):
            data_path = os.path.join(data_folder, 'Cp_Ubachs+2019.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            data_path = os.path.join(data_folder, 'Cm_Ubachs+2019.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            if lw_transitions_reference == 'U_19+S_15':
                data_path = os.path.join(data_folder, 'Cp_Salumbides+2015.hdf5')
                append_lw_transitions(lw_transitions_dictionary, data_path)
                data_path = os.path.join(data_folder, 'Cm_Salumbides+2015.hdf5')
                append_lw_transitions(lw_transitions_dictionary, data_path)
        elif lw_transitions_reference == 'A_94':
            data_path = os.path.join(data_folder, 'Cp_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            data_path = os.path.join(data_folder, 'Cm_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
    if excited_states_to_use == 'additional':
        if (lw_transitions_reference == 'U_19') | (lw_transitions_reference == 'U_19+S_15'):
            print('Error! U_19 and/or S_15 databases include only B+, C+ and C- transitions. Try with A_94.')
        elif lw_transitions_reference == 'A_94':
            data_path = os.path.join(data_folder, 'Bprime_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            data_path = os.path.join(data_folder, 'Dp_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            data_path = os.path.join(data_folder, 'Dm_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
    if excited_states_to_use == 'all':
        if (lw_transitions_reference == 'U_19') | (lw_transitions_reference == 'U_19+S_15'):
            print('Error! U_19 and/or S_15 databases include only B+, C+ and C- transitions. Try with A_94.')
        elif lw_transitions_reference == 'A_94':
            data_path = os.path.join(data_folder, 'Bp_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            data_path = os.path.join(data_folder, 'Cp_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            data_path = os.path.join(data_folder, 'Cm_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            data_path = os.path.join(data_folder, 'Bprime_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            data_path = os.path.join(data_folder, 'Dp_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)
            data_path = os.path.join(data_folder, 'Dm_Abgrall+1994.hdf5')
            append_lw_transitions(lw_transitions_dictionary, data_path)

    lw_transitions_dictionary['wl'] *= u.angstrom
    lw_transitions_dictionary['mean_Ekin'] *= u.eV
    lw_transitions_dictionary['Gamma'] /= u.s
    lw_transitions_dictionary['freq'] = (const.c).to(u.angstrom * u.Hz) / (lw_transitions_dictionary['wl'])

    return lw_transitions_dictionary


def filter_lw_dataset(
    ground_states_data,
    partition_function,
    lw_transitions_dictionary,
    min_partition_function=None,
    min_osc_strength_x_diss_fraction=None,
    min_osc_strength=None,
    min_diss_fraction=None
    ):

    '''
    Filter out the least significant LW transitions or the least populated rotovibrational levels.
    This improves the speed of the code.
    Input:
        ground_states_data: dictionary with the electronic ground state X rovib levels
        partition_function: partition function of the molecules
        lw_transitions_dictionary: dictionary of LW transitions, with oscillator strengths and other data
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
        ground_states_data: (reduced) dictionary with the electronic ground state X rovib levels
        partition_function: (reduced) partition function of the molecules
        lw_transitions_dictionary: (reduced) dictionary of LW transitions, with oscillator strengths and other data
    '''

    if min_partition_function is not None:
        mask_rovib_levels = partition_function >= min_partition_function
        for key in ground_states_data:
            ground_states_data[key] = ground_states_data[key][mask_rovib_levels]
        partition_function = partition_function[mask_rovib_levels]

    mask_lw_transitions = np.full(lw_transitions_dictionary['f'].shape, True)
    if min_osc_strength_x_diss_fraction is not None:
        mask_product = (lw_transitions_dictionary['f'] * lw_transitions_dictionary['frac_diss']) >= min_osc_strength_x_diss_fraction
        mask_lw_transitions = mask_lw_transitions & mask_product
    if min_osc_strength is not None:
        mask_osc_strength = lw_transitions_dictionary['f'] >= min_osc_strength
        mask_lw_transitions = mask_lw_transitions & mask_osc_strength
    if min_diss_fraction is not None:
        mask_diss_fraction = lw_transitions_dictionary['frac_diss'] >= min_diss_fraction
        mask_lw_transitions = mask_lw_transitions & mask_diss_fraction

    for key in lw_transitions_dictionary.keys():
        lw_transitions_dictionary[key] = lw_transitions_dictionary[key][mask_lw_transitions]

    return ground_states_data, partition_function, lw_transitions_dictionary


def get_reaction_min_energy():

    '''
    Get the LW transition with the lowest energy, from the (0, 0) and (0, 1) levels
    of the electronic ground state of H2 (the ground states of para- and ortho-H2 respectively).
    This value corresponds to what is commonly assumed as the minimum energy for the dissociation rate.
    The real minimum energy depends on which rotovibrational levels are populated.
    Output:
        min_energy: in [eV]
    '''

    lw_transitions_dictionary = get_lw_transitions('LW', 'U_19+S_15')
    mask = (lw_transitions_dictionary['VL'] == 0) & (lw_transitions_dictionary['JL'] <= 1)
    min_energy = lw_transitions_dictionary['freq'][mask].min() * (const.h).to(u.eV / u.Hz)
    
    return min_energy


def generate_lorentzian_line_profile(peak_frequency, gamma, frequency_array):
    
    '''
    *** CURRENTLY NOT USED ***
    *** the corresponding astropy function is used, that gives exactly the same results ***
    Input:
        peak_frequency: frequency of the peak, in [Hz]
        gamma: full-width half-maximum of the Lorentzian profile (natural broadening of the line), in [Hz]
        frequency_array: frequency array, in [Hz]
    Output:
        line_profile: Lorentzian line profile
    '''

    # make sure gamma is in [Hz] (equivalent to [1/s]), so the final line profile will have the dimension of [1/Hz]
    if type(gamma) != u.Quantity:
        gamma = gamma * u.Hz
    else:
        gamma = gamma.to(u.Hz)

    line_profile = (gamma / 2.) / (np.pi * ((frequency_array - peak_frequency)**2 + (gamma / 2.)**2))
    return line_profile


def calculate_composite_cross_section(
    gas_density=1e2 * u.cm**-3,
    gas_temperature=1e3 * u.K,
    excited_states_to_use='LW',
    lw_transitions_reference='U_19+S_15',
    lte_limit=True,
    line_profile_flag='V',
    min_partition_function=None,
    min_osc_strength_x_diss_fraction=None,
    min_osc_strength=None,
    min_diss_fraction=None,
    custom_frequency_array=None
):

    '''
    Calculate the composite cross section for a given gas density and temperature,
    considering all the rotovibrational levels of the electronic ground state.
    Calculate the partition function in the LTE limit or with Frigus code.
    Input:
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
        lte_limit: boolean, whether the LTE limit is being applied to determine the partition function
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
        custom_frequency_array: array of frequencies for the integration of the cross section;
            default is None, in which case a default array between 6.5 and 13.6 eV will be used
    Output:
        cross_section: composite cross section in [cm^2]
        heating_cross_section: composite 'heating' cross section in [eV cm^2]
    '''

    if lte_limit:
        ground_states_data = get_ground_states_data()
        partition_function = calculate_partition_function(gas_temperature, ground_states_data)
    else:
        ground_states_data, partition_function = call_to_frigus(gas_density, gas_temperature)
    lw_transitions_dictionary = get_lw_transitions(excited_states_to_use=excited_states_to_use, lw_transitions_reference=lw_transitions_reference)

    ground_states_data, partition_function, lw_transitions_dictionary = filter_lw_dataset(
        ground_states_data,
        partition_function,
        lw_transitions_dictionary,
        min_partition_function=min_partition_function,
        min_osc_strength_x_diss_fraction=min_osc_strength_x_diss_fraction,
        min_osc_strength=min_osc_strength,
        min_diss_fraction=min_diss_fraction
    )

    mass_H2 = 2.016 * const.u
    constant_factor = ((np.pi * const.e.si**2 / (4. * np.pi * const.m_e * const.c * const.eps0)).si.cgs).to(u.cm**2 * u.Hz)

    if custom_frequency_array is not None:
        if type(custom_frequency_array) != u.Quantity:
            custom_frequency_array = custom_frequency_array * u.Hz
        frequency_array = custom_frequency_array
    else:
        # minimum energy for a transition (in 'A_94' database): 6.7215773 eV
        # minimum energy for a transition (in 'U_19+S_15' database): 8.5509522 eV
        H2_diss_min_energy = 6.5 * u.eV
        ioniz_energy_hydrogen = get_ioniz_energy_hydrogen()
        energy_array = np.linspace(H2_diss_min_energy, ioniz_energy_hydrogen, int(1e5))
        frequency_array = energy_array / (const.h).to(u.eV / u.Hz)
    cross_section = np.zeros_like(frequency_array.value) * u.cm**2
    heating_cross_section = np.zeros_like(frequency_array.value) * u.eV * u.cm**2

    for index_gs in range(len(ground_states_data['v'])):
        gs_level_population = partition_function[index_gs]
        gs_vibr_quantum_number = ground_states_data['v'][index_gs]
        gs_rot_quantum_number = ground_states_data['J'][index_gs]
        mask_transitions = (lw_transitions_dictionary['VL'] == gs_vibr_quantum_number) & (lw_transitions_dictionary['JL'] == gs_rot_quantum_number)
        for index_trans in range(len(lw_transitions_dictionary['freq'][mask_transitions])):
            gamma = lw_transitions_dictionary['Gamma'][mask_transitions][index_trans]
            osc_strength = lw_transitions_dictionary['f'][mask_transitions][index_trans]
            diss_fraction = lw_transitions_dictionary['frac_diss'][mask_transitions][index_trans]
            mean_kin_energy = lw_transitions_dictionary['mean_Ekin'][mask_transitions][index_trans]
            peak_frequency = lw_transitions_dictionary['freq'][mask_transitions][index_trans]

            if line_profile_flag == 'V':
                gaussian_fwhm = peak_frequency * np.sqrt(8. * np.log(2.) * const.k_B * gas_temperature / (mass_H2 * const.c**2).to(u.J))
                profile_function = Voigt1D(
                    x_0=peak_frequency,
                    amplitude_L=2. / (np.pi * gamma),
                    fwhm_L=gamma,
                    fwhm_G=gaussian_fwhm
                )
                line_profile = profile_function(frequency_array).to(u.Hz**-1)
            elif line_profile_flag == 'L':
                # line_profile = generate_lorentzian_line_profile(peak_frequency, gamma, frequency_array).to(u.Hz**-1)
                profile_function = Lorentz1D(
                    x_0=peak_frequency,
                    amplitude=2. / (np.pi * gamma),
                    fwhm=gamma,
                )
                line_profile = profile_function(frequency_array).to(u.Hz**-1)
            cross_section += line_profile * constant_factor * gs_level_population * osc_strength * diss_fraction
            heating_cross_section += line_profile * constant_factor * gs_level_population * osc_strength * diss_fraction * mean_kin_energy

    return cross_section, heating_cross_section


def calculate_critical_density(gas_temperature):

    # fit that describes how the critical density changes with the gas temperature [range 1e2-1e4 K]
    # coefficients run from the highest power to the constant term
    critical_density_fit_coefficients = np.array([
        0.34039599533860915, -3.354413404320089,
        8.794202214931317, -0.21462262237517066
    ])
    log_gas_temperature = np.log10(gas_temperature.value)
    critical_density = np.power(10., np.polyval(critical_density_fit_coefficients, log_gas_temperature)) * u.cm**-3

    return critical_density

def calculate_critical_density_old(gas_temperature):

    critical_density_fit_coefficients = np.array([
        0.34039642148109583, -0.29084941325456615, 
        -2.141586661424785, 5.168955239880383
        ])
    log_gas_temperature = np.log10(gas_temperature.value)
    critical_density = np.power(10., np.polyval(critical_density_fit_coefficients, log_gas_temperature - 3.)) * u.cm**-3

    return critical_density


def calculate_kH2_low_resolution(
    wavelength_array,
    spectra_wl,
    distance,
    gas_density=1e2 * u.cm**-3,
    gas_temperature=1e3 * u.K,
    excited_states_to_use='LW',
    lw_transitions_reference='U_19+S_15',
    line_profile_flag='V',
    min_partition_function=None,
    min_osc_strength_x_diss_fraction=None,
    min_osc_strength=None,
    min_diss_fraction=None
):

    '''
    *** CURRENTLY NOT USED ***
    Calculate the photodissociation rate of H2 and the corresponding heating rate.
    The rate is interpolated between the low density limit (where the partition function is determined with the Frigus code)
    and the LTE limit, as in Glover (2015): https://ui.adsabs.harvard.edu/abs/2015MNRAS.451.2082G/abstract.
    The critical density for LTE is determined with a fitting function, valid between 1e2 K and 1e4 K.
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

    frequency_array = lambda2nu(wavelength_array)
    energy_array = (const.h).to(u.eV / u.Hz) * frequency_array
    # check on dimension, so we can cycle over the first dimension
    if spectra_wl.ndim == 1:
        spectra_wl = np.atleast_2d(spectra_wl)
    number_of_spectra = spectra_wl.shape[0]

    units_monochromatic_luminosity_wl = u.erg / u.s / u.angstrom
    units_monochromatic_luminosity_freq = u.erg / u.s / u.Hz
    units_monochromatic_intensity_freq = units_monochromatic_luminosity_freq / u.cm**2 / u.sr

    spectra_freq = u.quantity.Quantity(value=np.empty_like(spectra_wl.value), unit=units_monochromatic_luminosity_freq)
    for i in range(number_of_spectra):
        spectra_freq[i] = spec_lambda2nu(wavelength_array, spectra_wl[i])

    # determine the cross sections in the two limits
    low_density_cross_section, low_density_heating_cross_section = calculate_composite_cross_section(
        gas_density, gas_temperature,
        excited_states_to_use, lw_transitions_reference,
        False, line_profile_flag,
        min_partition_function, min_osc_strength_x_diss_fraction, min_osc_strength, min_diss_fraction,
        frequency_array
    )
    lte_cross_section, lte_heating_cross_section = calculate_composite_cross_section(
        gas_density, gas_temperature,
        excited_states_to_use, lw_transitions_reference,
        True, line_profile_flag,
        min_partition_function, min_osc_strength_x_diss_fraction, min_osc_strength, min_diss_fraction,
        frequency_array
    )

    # perform the integrations
    solid_angle = 4. * np.pi * u.sr
    surface_area = 4. * np.pi * (distance.to(u.cm))**2
    intensity_freq = spectra_freq / solid_angle / surface_area

    integration_x_axis = frequency_array

    integration_y_axis = low_density_cross_section * intensity_freq / energy_array.to(u.erg)
    low_density_rate = solid_angle * np.trapz(integration_y_axis, integration_x_axis)
    integration_y_axis = lte_cross_section * intensity_freq / energy_array.to(u.erg)
    lte_rate = solid_angle * np.trapz(integration_y_axis, integration_x_axis)

    integration_y_axis = low_density_heating_cross_section * intensity_freq / energy_array.to(u.erg)
    low_density_heating_rate = solid_angle * np.trapz(integration_y_axis, integration_x_axis)
    integration_y_axis = lte_heating_cross_section * intensity_freq / energy_array.to(u.erg)
    lte_heating_rate = solid_angle * np.trapz(integration_y_axis, integration_x_axis)

    critical_density = calculate_critical_density(gas_temperature)
    alpha_exponent = (1. + gas_density / critical_density)**-1

    dissociation_rate = lte_rate * (low_density_rate / lte_rate)**alpha_exponent
    heating_rate = lte_heating_rate * (low_density_heating_rate / lte_heating_rate)**alpha_exponent

    return dissociation_rate, heating_rate


def calculate_kH2(
    wavelength_array,
    spectra_wl,
    distance,
    gas_density=1e2 * u.cm**-3,
    gas_temperature=1e3 * u.K,
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

    frequency_array = lambda2nu(wavelength_array)
    energy_array = (const.h).to(u.eV / u.Hz) * frequency_array
    # check on dimension, so we can cycle over the first dimension
    if spectra_wl.ndim == 1:
        spectra_wl = np.atleast_2d(spectra_wl)
    number_of_spectra = spectra_wl.shape[0]

    units_monochromatic_luminosity_wl = u.erg / u.s / u.angstrom
    units_monochromatic_luminosity_freq = u.erg / u.s / u.Hz
    units_monochromatic_intensity_freq = units_monochromatic_luminosity_freq / u.cm**2 / u.sr

    spectra_freq = u.quantity.Quantity(value=np.empty_like(spectra_wl.value), unit=units_monochromatic_luminosity_freq)
    for i in range(number_of_spectra):
        spectra_freq[i] = spec_lambda2nu(wavelength_array, spectra_wl[i])
    
    solid_angle = 4. * np.pi * u.sr
    surface_area = 4. * np.pi * (distance.to(u.cm))**2
    intensity_freq = spectra_freq / solid_angle / surface_area
    
    # minimum energy for a transition (in 'A_94' database): 6.7215773 eV
    # minimum energy for a transition (in 'U_19+S_15' database): 8.5509522 eV
    H2_diss_min_energy = 6.5 * u.eV
    ioniz_energy_hydrogen = get_ioniz_energy_hydrogen()
    # interpolate spectra to enhance the resolution, 1e5 points are enough if using the Voigt line profile;
    # am I allowed to do this? the alternative is not viable, the result would depend too much on the initial resolution
    custom_energy_array = np.linspace(H2_diss_min_energy, ioniz_energy_hydrogen, int(1e5))
    custom_frequency_array = custom_energy_array / (const.h).to(u.eV / u.Hz)
    new_intensity_freq = interpolate_array(
        old_x_axis=frequency_array,
        old_y_axis=intensity_freq,
        new_x_axis=custom_frequency_array
    )

    # determine the cross sections in the two limits
    low_density_cross_section, low_density_heating_cross_section = calculate_composite_cross_section(
        gas_density, gas_temperature,
        excited_states_to_use, lw_transitions_reference,
        False, line_profile_flag,
        min_partition_function, min_osc_strength_x_diss_fraction, min_osc_strength, min_diss_fraction,
        custom_frequency_array
    )
    lte_cross_section, lte_heating_cross_section = calculate_composite_cross_section(
        gas_density, gas_temperature,
        excited_states_to_use, lw_transitions_reference,
        True, line_profile_flag,
        min_partition_function, min_osc_strength_x_diss_fraction, min_osc_strength, min_diss_fraction,
        custom_frequency_array
    )

    # perform the integrations
    integration_x_axis = custom_frequency_array

    integration_y_axis = low_density_cross_section * new_intensity_freq / custom_energy_array.to(u.erg)
    low_density_rate = solid_angle * np.trapz(integration_y_axis, integration_x_axis)
    integration_y_axis = lte_cross_section * new_intensity_freq / custom_energy_array.to(u.erg)
    lte_rate = solid_angle * np.trapz(integration_y_axis, integration_x_axis)

    integration_y_axis = low_density_heating_cross_section * new_intensity_freq / custom_energy_array.to(u.erg)
    low_density_heating_rate = solid_angle * np.trapz(integration_y_axis, integration_x_axis)
    integration_y_axis = lte_heating_cross_section * new_intensity_freq / custom_energy_array.to(u.erg)
    lte_heating_rate = solid_angle * np.trapz(integration_y_axis, integration_x_axis)

    critical_density = calculate_critical_density(gas_temperature)
    alpha_exponent = (1. + gas_density / critical_density)**-1

    dissociation_rate = lte_rate * (low_density_rate / lte_rate)**alpha_exponent
    heating_rate = lte_heating_rate * (low_density_heating_rate / lte_heating_rate)**alpha_exponent

    return dissociation_rate, heating_rate
