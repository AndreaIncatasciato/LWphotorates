import os
import numpy as np
import h5py
from astropy import constants as const
from astropy import units as u

from scipy.interpolate import InterpolatedUnivariateSpline

from LWphotorates.utils import nu2lambda, lambda2nu, spec_lambda2nu
from LWphotorates.utils import convert_energy_cm2ev, convert_energy_ev2cm, convert_energy_cm2k

import frigus
from frigus.readers.dataset import DataLoader
from frigus.readers.read_energy_levels import read_levels_lique
from frigus.population import population_density_at_steady_state
from astropy.modeling.models import Voigt1D


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


def read_Xstates():
    
    '''
    This function reads the file from Komasa et al. (2011)
    with energies of the rotovibrational levels of the electronic ground state X.

    Structure of the input file (3 cols):
        v
        J
        binding energy        [1/cm]
    Output:
        dictionary with:
            v
            J
            E(0,0)-E(v,J)     [1/cm]
            E(0,0)-E(v,J)     [eV]
            equivalent T      [K]        
    '''

    Xen_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/Xgroundstate/vibrotXenergy_Komasa+2011.txt'
    Xen_db=np.loadtxt(Xen_db_f,unpack=True)
    Xen={
        'v':Xen_db[0],
        'J':Xen_db[1],
        'cm':Xen_db[2,0]-Xen_db[2]
    }
    Xen['eV']=convert_energy_cm2ev(Xen['cm'])
    Xen['K']=convert_energy_cm2k(Xen['cm'])
    Xen['cm']/=u.cm

    return Xen


def join_dbs(path,all_transitions):

    '''
    Append a list of LW transitions to the general dictionary.
    Input:
        path is the path to the file with the transitions (its structure is fixed)
        all_transitions is the dictionary with the same structure as the input files
    Output: nothing
    '''

    trans_db=h5py.File(name=path,mode='r')
    for i in list(all_transitions.keys()):
        all_transitions[i]=np.concatenate((all_transitions[i],trans_db[i]))
    trans_db.close()


def read_transitions(db_touse,exstates_touse):

    '''
    This function reads the transitions database.
    Input:
        db_touse: which database is employed; available choices are 'A94','U19','U19+S15'
            'A94' is the classical db from Abgrall+1993a,b,c,Abgrall+1994, the most complete
            'U19' is the db from Ubachs+2019, with B+/C+/C- states; it has transitions only from X states with v=0,J=0-7 so good at low temperatures
            'U19+S15' joins Ubachs+2019 with Salumbides+2015, with B+/C+/C- states; it has transitions a good number of LW transitions, good compromise
        exstates_touse: which excited electronic states are taken into account; available choices are 'B','C','LW','additional','all'
            'B' means only B+ states (the 'Lyman' lines), available for all the datasets provided
            'C' means only C+ and C- states (the 'Werner' lines), available for all the datasets provided
            'LW' means B+/C+/C- states (all the 'Lyman-Werner' lines), available for all the datasets provided
            'additional' means only B'+/D+/D- states (that are usually neglected), available only for the Abgrall+1994 db
            'all' means all six states (B+/C+/C-/B'+/D+/D-), available only for the Abgrall+1994 db
    Output:
        a dictionary with the following data for each transition:
            VL: vibrational quantum number of the X state
            JL: rotational quantum number of the X state
            VU: vibrational quantum number of the excited state
            JU: rotational quantum number of the excited state
            wl: wavelength of the transition                                                   [A]
            f: oscillator strength
            Gamma: natural broadening parameter                                                [1/s]
            frac_diss: fraction of excited molecules that will dissociate
            mean_Ekin: mean kinetic energy of the H atoms (heating of the gas per molecule)    [eV]
    '''

    all_transitions={
        'VL':np.array([]),
        'JL':np.array([]),
        'VU':np.array([]),
        'JU':np.array([]),
        'wl':np.array([]),
        'f':np.array([]),
        'Gamma':np.array([]),
        'frac_diss':np.array([]),
        'mean_Ekin':np.array([])    
    }

    if (exstates_touse=='B')|(exstates_touse=='LW'):
        if (db_touse=='U19')|(db_touse=='U19+S15'):
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Bp_Ubachs+2019.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            if db_touse=='U19+S15':
                trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Bp_Salumbides+2015.hdf5'
                join_dbs(path=trans_db_f,all_transitions=all_transitions)
        elif db_touse=='A94':
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Bp_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
    if (exstates_touse=='C')|(exstates_touse=='LW'):
        if (db_touse=='U19')|(db_touse=='U19+S15'):
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Cp_Ubachs+2019.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Cm_Ubachs+2019.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            if db_touse=='U19+S15':
                trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Cp_Salumbides+2015.hdf5'
                join_dbs(path=trans_db_f,all_transitions=all_transitions)
                trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Cm_Salumbides+2015.hdf5'
                join_dbs(path=trans_db_f,all_transitions=all_transitions)
        elif db_touse=='A94':
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Cp_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Cm_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
    if exstates_touse=='additional':
        if (db_touse=='U19')|(db_touse=='U19+S15'):
            print('Error! Ubachs and/or Salumbides dbs have only B+, C+ and C- transitions')
        elif db_touse=='A94':
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Bprime_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Dp_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Dm_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
    if exstates_touse=='all':
        if (db_touse=='U19')|(db_touse=='U19+S15'):
            print('Error! Ubachs and/or Salumbides dbs have only B+, C+ and C- transitions')
        elif db_touse=='A94':
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Bp_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Cp_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Cm_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Bprime_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Dp_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)
            trans_db_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2/transitions/cleaned/Dm_Abgrall+1994.hdf5'
            join_dbs(path=trans_db_f,all_transitions=all_transitions)

    all_transitions['freq']=const.c.to(u.angstrom*u.Hz)/(all_transitions['wl']*u.angstrom)

    return all_transitions


def Xpop(Xen,Tgas):

    '''
    Define the population of the electronic ground state X rovib levels in the LTE limit.
    The population level is just a number between 0 and 1, the array is normalised to 1.
    Input:
        Tgas: gas temperature      [K]
        Xen: dictionary with the electronic ground state X rovib levels
    Output:
        NX: array with LTE Boltzmann coefficients
    '''

    if type(Tgas)==u.Quantity:
        Tgas=Tgas.value

    
    NX=(2-(-1)**Xen['J'])*(2*Xen['J']+1)*np.exp(-Xen['K'].value/Tgas)
    NX/=NX.sum()
    return NX


def lorentzian(Gamma,nu0,nu):
    
    '''
    Input:
        Gamma: FWHM of the Lorentzian profile (natural broadening of the line)
        nu0: central frequency
        nu: frequency array
    Output:
        lorentzian line profile, array with the same size as the frequency array
    '''
    
    return (Gamma/2.)/(np.pi*((nu-nu0)**2+(Gamma/2.)**2))


def Xpop_frigus(ngas,Tgas):

    '''
    In the low density regime LTE approximation is not valid anymore.
    Instead of considering only ground states of ortho and para-H2 use Frigus code.
    This uses 58 states, steady-state solution should be fine.
    Input:
        ngas: gas number density                                      [1/cm^3]
        Tgas: gas temperature                                         [K]
    Output:
        Xen: dictionary with only the two rovib ground states (para and ortho) of X states
        NX: population of these two levels
    '''

    '''
    mask_Xlevel=(Xen['v']==0)&(Xen['J']<2)
    Xen_OLD=Xen
    for i in list(Xen_OLD.keys()):
        Xen[i]=Xen_OLD[i][mask_Xlevel]
    NX_OLD=NX
    NX=NX_OLD[mask_Xlevel]
    NX/=NX.sum()
    return Xen,NX
    '''

    data=read_levels_lique(os.environ['FRIGUS_DATADIR_ROOT']+'/H2Xvjlevels_francois_mod.cs').data
    Xen_Frigus={
        'v':np.array(data['v'].tolist()),
        'J':np.array(data['j'].tolist()),
        'eV':data['E']-data['E'][0]
    }
    Xen_Frigus['cm']=convert_energy_ev2cm(Xen_Frigus['eV'])
    Xen_Frigus['K']=convert_energy_cm2k(Xen_Frigus['cm'])

    redshift=15.
    cmb_temp=2.72548*(1.+redshift)*u.K    # CMB is only important at very low density when collisions are extremely rare
    ngas=ngas.si
    Tgas=np.minimum(Tgas,5e3*u.K)   # limit of validity of Frigus code

    NX_Frigus=population_density_at_steady_state(
        data_set=DataLoader().load('H2_lique'),
        t_kin=Tgas,
        t_rad=cmb_temp,
        collider_density=ngas
    )
    NX_Frigus=NX_Frigus.flatten()

    for i in range(len(NX_Frigus)):
        if Xen_Frigus['J'][i]%2:
            NX_Frigus[i]*=3   # add nuclear spin parity, don't know why they are missing it
    NX_Frigus/=NX_Frigus.sum()
    return Xen_Frigus,NX_Frigus


def filter_speedup(Xen,NX,all_transitions,thresh_Xlevel,thresh_oscxfdiss,thresh_osc,thresh_fdiss):

    '''
    To speed up the computation, in case lots of rovib levels or transitions are available, just consider the most significant.
    Input:
        Xen: dictionary with the energies of all the X levels
        NX: population of all the X levels
        all_transitions: dictionary with all the LW transitions
        thresh_Xlevel: minimum level population to take into account a X rovib level, suggested values [1e-5-1e-3]
        thresh_oscxfdiss: minimum level population to take into account a X rovib level, suggested value 1e-4
        thresh_osc: minimum level population to take into account a X rovib level, suggested value 1e-3
        thresh_fdiss: minimum level population to take into account a X rovib level, suggested value 1e-2
    Output:
        Xen: dictionary with the energies of all the X levels with population above a certain threshold, if defined
        NX: population of all the X levels with population above a certain threshold, if defined
        all_transitions: dictionary with all the LW transitions, with osc, fdiss or osc*fdiss above a certain threshold, if defined
    '''

    if thresh_Xlevel is not None:
        mask_Xlevel=NX>=thresh_Xlevel
        Xen_OLD=Xen
        for i in list(Xen_OLD.keys()):
            Xen[i]=Xen_OLD[i][mask_Xlevel]
        NX_OLD=NX
        NX=NX_OLD[mask_Xlevel]

    mask_transitions=None
    if thresh_osc is not None:
        mask_transitions=all_transitions['f']>=thresh_osc
    elif thresh_fdiss is not None:
        mask_transitions=all_transitions['frac_diss']>=thresh_fdiss
    elif thresh_oscxfdiss is not None:
        mask_transitions=(all_transitions['f']*all_transitions['frac_diss'])>=thresh_oscxfdiss
    if mask_transitions is not None:
        all_transitions_OLD=all_transitions
        for i in list(all_transitions_OLD.keys()):
            all_transitions[i]=all_transitions_OLD[i][mask_transitions]

    return Xen,NX,all_transitions


def calc_sigma(freq_array,ngas,Tgas,LTE,db_touse='U19+S15',exstates_touse='LW',lineprofile_touse='V',thresh_Xlevel=None,thresh_oscxfdiss=None,thresh_osc=None,thresh_fdiss=None):

    '''
    Given Tgas and the parameters of the run, calculate the effective cross section to compute the dissociation rate and the heating rate.
    Input:
        freq_array: array of frequencies for the integration        [Hz]
        ngas: gas number density                                      [1/cm^3]
        Tgas: gas temperature                                         [K]
        LTE: boolean, whether the LTE limit is being applied
        db_touse: which database is employed; available choices are 'A94','U19','U19+S15'
            'A94' is the classical db from Abgrall+1993a,b,c,Abgrall+1994, the most complete
            'U19' is the db from Ubachs+2019, with B+/C+/C- states; it has transitions only from X states with v=0,J=0-7 so good at low temperatures
            'U19+S15' joins Ubachs+2019 with Salumbides+2015, with B+/C+/C- states; it has transitions a good number of LW transitions, good compromise
        exstates_touse: which excited electronic states are taken into account; available choices are 'B','C','LW','additional','all'
            'B' means only B+ states (the 'Lyman' lines), available for all the datasets provided
            'C' means only C+ and C- states (the 'Werner' lines), available for all the datasets provided
            'LW' means B+/C+/C- states (all the 'Lyman-Werner' lines), available for all the datasets provided
            'additional' means only B'+/D+/D- states (that are usually neglected), available only for the Abgrall+1994 db
            'all' means all six states (B+/C+/C-/B'+/D+/D-), available only for the Abgrall+1994 db
        lineprofile_touse: profile of the LW lines, 'L' for Lorentzian (only natural broadening) or 'V' for Voigt (natural + thermal broadening)
        thresh_Xlevel: minimum level population to take into account a X rovib level, suggested values [1e-5-1e-3]
        thresh_oscxfdiss: minimum level population to take into account a X rovib level, suggested value 1e-4
        thresh_osc: minimum level population to take into account a X rovib level, suggested value 1e-3
        thresh_fdiss: minimum level population to take into account a X rovib level, suggested value 1e-2
    Output:
        totsigma: effective cross section, same size as freq_array            [cm^2]
        heating: effective 'heating' cross section, same size as freq_array   [cm^2 eV]
    '''

    if LTE:
        Xen=read_Xstates()
        NX=Xpop(Xen=Xen,Tgas=Tgas)    
    else:
        Xen,NX=Xpop_frigus(ngas=ngas,Tgas=Tgas)
    all_transitions=read_transitions(db_touse=db_touse,exstates_touse=exstates_touse)
    Xen,NX,all_transitions=filter_speedup(Xen=Xen,NX=NX,all_transitions=all_transitions,
        thresh_Xlevel=thresh_Xlevel,thresh_oscxfdiss=thresh_oscxfdiss,thresh_osc=thresh_osc,thresh_fdiss=thresh_fdiss)

    mass_H2=2.016*const.u
    coeff=(np.pi*const.e.si**2/(4.*np.pi*const.m_e*const.c*const.eps0)).si.cgs.value

    totsigma=np.zeros_like(freq_array.value)
    heating=np.zeros_like(freq_array.value)

    for index in range(len(Xen['v'])):
        VL=Xen['v'][index]
        JL=Xen['J'][index]
        mask_v=all_transitions['VL']==VL
        mask_j=all_transitions['JL'][mask_v]==JL
        PvJ=NX[index]
        for i in range(len(all_transitions['freq'][mask_v][mask_j])):
            gamma=all_transitions['Gamma'][mask_v][mask_j][i]/u.s
            osc=all_transitions['f'][mask_v][mask_j][i]
            fdiss=all_transitions['frac_diss'][mask_v][mask_j][i]
            ekin=all_transitions['mean_Ekin'][mask_v][mask_j][i]
            freq=all_transitions['freq'][mask_v][mask_j][i]
            if lineprofile_touse=='V':
                fwhm_G=np.sqrt(8.*np.log(2.)*const.k_B*Tgas/(mass_H2*const.c**2).to(u.J))*freq
                v1=Voigt1D(x_0=freq.value,amplitude_L=2./(np.pi*gamma.value),fwhm_L=gamma.value,fwhm_G=fwhm_G.value)
                lprof=v1(freq_array.value)*osc*fdiss*PvJ*coeff
            elif lineprofile_touse=='L':
                lprof=lorentzian(Gamma=gamma.value,nu0=freq.value,nu=freq_array.value)*osc*fdiss*PvJ*coeff
            totsigma+=lprof
            heating+=lprof*ekin

    return totsigma,heating


def calc_kH2(lambda_array,spectra_lambda,distance,ngas,Tgas,
    return_sigma=False,return_heating=False,return_sigma_only=False,return_heating_only=False,
    db_touse='U19+S15',exstates_touse='LW',lineprofile_touse='V',
    thresh_Xlevel=None,thresh_oscxfdiss=None,thresh_osc=None,thresh_fdiss=None):
    
    '''
    Calculate the photodissociation rate of H2 due to LW photons and its associated heating rate.
    One rate for each spectrum.
    Input:
        lambda_array: wavelength array associated with the spectra    [A]
        spectra_lambda: spectra                                       [erg/A/s]
        distance: distance of the radiating source                    [kpc]
        ngas: gas number density                                      [1/cm^3]
        Tgas: gas temperature                                         [K]
        return_sigma: the function will return diss rate, heating rate and the effective cross section
        return_heating: the function will return diss rate, heating rate and the monochromatic heating rate
        return_sigma_only: the function will return the effective cross section without calculating the diss rate
        return_heating_only: the function will return the monochromatic heating rate without calculating the diss rate
        db_touse: which database is employed; available choices are 'A94','U19','U19+S15'
            'A94' is the classical db from Abgrall+1993a,b,c,Abgrall+1994, the most complete
            'U19' is the db from Ubachs+2019, with B+/C+/C- states; it has transitions only from X states with v=0,J=0-7 so good at low temperatures
            'U19+S15' joins Ubachs+2019 with Salumbides+2015, with B+/C+/C- states; it has transitions a good number of LW transitions, good compromise
        exstates_touse: which excited electronic states are taken into account; available choices are 'B','C','LW','additional','all'
            'B' means only B+ states (the 'Lyman' lines), available for all the datasets provided
            'C' means only C+ and C- states (the 'Werner' lines), available for all the datasets provided
            'LW' means B+/C+/C- states (all the 'Lyman-Werner' lines), available for all the datasets provided
            'additional' means only B'+/D+/D- states (that are usually neglected), available only for the Abgrall+1994 db
            'all' means all six states (B+/C+/C-/B'+/D+/D-), available only for the Abgrall+1994 db
        lineprofile_touse: profile of the LW lines, 'L' for Lorentzian (only natural broadening) or 'V' for Voigt (natural + thermal broadening)
        thresh_Xlevel: minimum level population to take into account a X rovib level, suggested values [1e-5-1e-3]
        thresh_oscxfdiss: minimum level population to take into account a X rovib level, suggested value 1e-4
        thresh_osc: minimum level population to take into account a X rovib level, suggested value 1e-3
        thresh_fdiss: minimum level population to take into account a X rovib level, suggested value 1e-2
    Output:
        dissociation rate and heating rate, both interpolated between GS and LTE limits
        effective cross section
        monochromatic heating rate
        high-resolution frequency array                               [Hz]
    '''

    Lyman_lim=(const.h*const.c*const.Ryd).to(u.eV)*const.m_p/(const.m_p+const.m_e)
    minEn=6.*u.eV    # minimum energy for a transition (in the LW db of A94): 6.7215772 eV
    maxEn=Lyman_lim
    minFreq=minEn/const.h.to(u.eV/u.Hz)
    maxFreq=maxEn/const.h.to(u.eV/u.Hz)
    minWl=nu2lambda(maxFreq)
    maxWl=nu2lambda(minFreq)
    en_array_HR=np.linspace(minEn.value,maxEn.value,100000)*u.eV
    nu_array_HR=en_array_HR/const.h.to(u.eV/u.Hz)

    sigma_noLTE,heating_noLTE=calc_sigma(
        freq_array=nu_array_HR,ngas=ngas,Tgas=Tgas,LTE=False,
        db_touse=db_touse,exstates_touse=exstates_touse,lineprofile_touse=lineprofile_touse,
        thresh_Xlevel=thresh_Xlevel,thresh_oscxfdiss=thresh_oscxfdiss,thresh_osc=thresh_osc,thresh_fdiss=thresh_fdiss)
    sigma_LTE,heating_LTE=calc_sigma(
        freq_array=nu_array_HR,ngas=ngas,Tgas=Tgas,LTE=True,
        db_touse=db_touse,exstates_touse=exstates_touse,lineprofile_touse=lineprofile_touse,
        thresh_Xlevel=thresh_Xlevel,thresh_oscxfdiss=thresh_oscxfdiss,thresh_osc=thresh_osc,thresh_fdiss=thresh_fdiss)

    if return_sigma_only:
        return sigma_noLTE,sigma_LTE,nu_array_HR
    if return_heating_only:
        return heating_noLTE,heating_LTE,nu_array_HR

    if spectra_lambda.ndim==1:
        spectra_lambda=np.atleast_2d(spectra_lambda)

    nu_array=lambda2nu(lambda_array)
    spectra_nu=u.quantity.Quantity(value=np.empty_like(spectra_lambda.value),unit=u.erg/u.s/u.Hz)
    for i in range(len(spectra_lambda)):
        spectra_nu[i]=spec_lambda2nu(lambda_array, spectra_lambda[i])

    intensity_nu=spectra_nu/(4*np.pi*u.sr*4*np.pi*(distance.to(u.cm))**2)
    units_intensity=intensity_nu.unit
    intensity_nu_HR=np.empty(shape=(len(intensity_nu),len(nu_array_HR)))
    for i in range(len(intensity_nu)):
        interpspectrum=InterpolatedUnivariateSpline(nu_array,intensity_nu[i].value,k=1)
        intensity_nu_HR[i]=interpspectrum(nu_array_HR)
    intensity_nu_HR=intensity_nu_HR*units_intensity

    diss_rate_noLTE=4.*np.pi*u.sr*np.trapz(sigma_noLTE*u.cm**2*intensity_nu_HR/en_array_HR.to(u.erg),nu_array_HR)
    heat_rate_noLTE=4.*np.pi*u.sr*np.trapz(heating_noLTE*u.cm**2*intensity_nu_HR/en_array_HR.to(u.erg),nu_array_HR)

    diss_rate_LTE=4.*np.pi*u.sr*np.trapz(sigma_LTE*u.cm**2*intensity_nu_HR/en_array_HR.to(u.erg),nu_array_HR)
    heat_rate_LTE=4.*np.pi*u.sr*np.trapz(heating_LTE*u.cm**2*intensity_nu_HR/en_array_HR.to(u.erg),nu_array_HR)

    fit_ncrit_coef=[5.168955239880383,-2.141586661424785,-0.29084941325456615,0.34039642148109583]   # this is my fit that describes how ncrit changes with Tgas [range 1e2-1e4 K]
    log_Tgas=np.log10(Tgas.value)
    ncrit=np.power(10.,fit_ncrit_coef[0]+fit_ncrit_coef[1]*(-3+log_Tgas)+fit_ncrit_coef[2]*(-3+log_Tgas)**2+fit_ncrit_coef[3]*(-3+log_Tgas)**3)*u.cm**-3
    alpha=(1+ngas/ncrit)**(-1)

    diss_rate=diss_rate_LTE*(diss_rate_noLTE/diss_rate_LTE)**alpha
    heat_rate=heat_rate_LTE*(heat_rate_noLTE/heat_rate_LTE)**alpha

    if return_sigma&return_heating:
        return diss_rate,heat_rate,sigma_noLTE,heating_noLTE,sigma_LTE,heating_LTE,nu_array_HR
    elif return_sigma:
        return diss_rate,heat_rate,sigma_noLTE,sigma_LTE,nu_array_HR
    elif return_heating:
        return diss_rate,heat_rate,heating_noLTE,heating_LTE,nu_array_HR
    else:
        return diss_rate,heat_rate