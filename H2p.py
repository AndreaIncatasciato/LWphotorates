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


def read_Babb(wl_new):
    
    '''
    Read the H2p (rotovibrational level resolved) cross section from Babb (2015), together with the relative energies of the rovib levels
    and the kinetic energy released in the gas.

    Input:
        high-resolution wavelength to resample the cross sections         [A]
    Output:
        Dictionary Xen:
            rovib quantum numbers
            energies of the levels                                        [1/cm]/[eV]/[K]
        Dictionary sigma_pd:
            cross sections                                                [cm^2]
            kinetic energy released                                       [eV]
    '''

    Babb_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2p/Babb2015.txt'
    Babb=np.loadtxt(Babb_f,unpack=True)

    Babb_v=[]
    Babb_j=[]
    Babb_en=[]
    Babb_sigma=[]
    Babb_Ekin=[]

    uniqv=np.unique(Babb[0])
    for v in uniqv:
        maskv=Babb[0]==v
        uniqj=np.unique(Babb[1][maskv])
        for j in uniqj:
            maskj=Babb[1][maskv]==j
            Babb_v.append(int(v))
            Babb_j.append(int(j))
            Babb_en.append(Babb[3][maskv][maskj][0])
            Ekin=Babb[2][maskv][maskj]*u.rydberg.to(u.eV)*2*u.eV
            wl=(Babb[4][maskv][maskj]*u.nm).to(u.angstrom)
            sigma=2.689e-18*Babb[5][maskv][maskj]*45.563/Babb[4][maskv][maskj]*u.cm**2
            interp=InterpolatedUnivariateSpline(x=wl,y=sigma,k=1)
            sigma_new=interp(wl_new,ext='zeros')
            Babb_sigma.append(sigma_new)
            interp=InterpolatedUnivariateSpline(x=wl,y=Ekin,k=1)
            Ekin_new=interp(wl_new,ext='zeros')
            Babb_Ekin.append(Ekin_new)

    Xen={
        'v':np.array(Babb_v),
        'J':np.array(Babb_j),
        'cm':Babb_en/u.cm
    }
    Xen['eV']=convert_energy_cm2ev(Xen['cm'])
    Xen['K']=convert_energy_cm2k(Xen['cm'][0]-Xen['cm'])

    sigma_pd={
        'sigma':np.array(Babb_sigma)*u.cm**2,
        'heat_sigma':np.array(Babb_Ekin)*np.array(Babb_sigma)*u.cm**2*u.eV
    }

    return Xen,sigma_pd



def read_Zammit(wl_new):
    
    '''
    Read the UGAMOP data for the rovib states.
    Read the H2p (rotovibrational level resolved) cross sections from Zammit et al. (2017), together with the kinetic energy released in the gas.

    Input:
        high-resolution wavelength to resample the cross sections         [A]
    Output:
        Dictionary Xen:
            rovib quantum numbers
            energies of the levels                                        [1/cm]/[eV]/[K]
        Dictionary sigma_pd:
            cross sections                                                [cm^2]
            heating cross section                                         [eV]
    '''

    Xen=np.loadtxt(os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2p/Xen.txt',unpack=True)
    Xen={
        'v':np.array(Xen[0],dtype=np.int16),
        'J':np.array(Xen[1],dtype=np.int16),
        'eV':Xen[2]*u.rydberg.to(u.eV)*2*u.eV
    }
    Xen['cm']=convert_energy_ev2cm(Xen['eV'])
    Xen['K']=convert_energy_cm2k(Xen['cm'][0]-Xen['cm'])

    Zammit_f=os.path.dirname(os.path.abspath(__file__))+'/inputdata/H2p/Zammit2017.hdf5'
    f=h5py.File(name=Zammit_f,mode='r')

    sigma=np.empty(shape=(len(f['sigma']),len(wl_new)))
    heat_sigma=np.empty(shape=(len(f['sigma*en']),len(wl_new)))
    for i in range(len(f['sigma'])):
        interp=InterpolatedUnivariateSpline(x=f['phot_wl'],y=f['sigma'][i],k=1)
        sigma[i]=interp(wl_new,ext='zeros')
        interp=InterpolatedUnivariateSpline(x=f['phot_wl'],y=f['sigma*en'][i],k=1)
        heat_sigma[i]=interp(wl_new,ext='zeros')

    sigma_pd={
        'sigma':sigma*u.cm**2,
        'heat_sigma':heat_sigma*u.eV*u.cm**2
    }

    return Xen,sigma_pd


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


def calc_ncrit(Tgas):

    '''
    Compute the critical density for the LTE limit as proposed in Glover 2015.
    Assume a neutral gas with standard composition.
    Main colliders: H and e.
    Input:
        Tgas       [K]
    Output:
        ncrit      [1/cm^3]
    '''

    if type(Tgas)==u.Quantity:
        Tgas=Tgas.value

# assume composition
    XHe=0.08
    XH2=1e-4
    XH=1.-2*XH2
    Xe=0.01

# critical densities, assume main colliders are H and e
# (see Glover & Savin 2009 and Glover 2015)
    ncrH=400.*(Tgas/1e4)**(-1)*u.cm**-3
    ncre=50.*u.cm**-3
    return ((XH+Xe)*(XH/ncrH+Xe/ncre)**(-1))


def calc_sigma(ngas,Tgas,wl_new,Dunn,Zammit):

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

    if Zammit:
        Xen,sigma_pd=read_Zammit(wl_new=wl_new)
    else:
        Xen,sigma_pd=read_Babb(wl_new=wl_new)

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


def calc_kH2p(lambda_array,spectra_lambda,distance,ngas,Tgas,Dunn=False,Zammit=True,
    return_sigma=False,return_heating=False,
    return_sigma_only=False,return_heating_only=False):

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

    sigma,heating=calc_sigma(ngas=ngas,Tgas=Tgas,wl_new=wl_new,Dunn=Dunn,Zammit=Zammit)

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