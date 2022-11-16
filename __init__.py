import numpy as __np  # why this? so it doesn't show np when doing 'import ComputeRate'
from astropy import units as __u

import photorates.H2 as H2
import photorates.H2p as H2p
import photorates.HM as HM


# when doing from ComputeRate import * it will only these
__all__=['compute_kH2','compute_kHM','compute_kH2p','utils','H2','H2p','HM']



def compute_kHM(lambda_array,spectra_lambda,distance=1.*__u.kpc,
    return_sigma=False,return_sigma_only=False):
    
    '''
    Calculate the photodetachment rate of HM.
    One rate for each spectrum.
    Input:
        lambda_array: wavelength array associated with the spectra    [A]
        spectra_lambda: spectra                                       [erg/A/s]
        distance: distance of the radiating source                    [kpc]
        return_sigma: the function will return the rate and the cross section
        return_sigma_only: the function will return the cross section without calculating the rate
    Output:
        detachment rate                                               [1/s]
        cross section dictionary
    '''
    
    if type(distance)!=__u.Quantity:
        distance=distance*__u.kpc

    if distance.value<=0:
        print('wrong distance dumbass!')
        return -1
    
# if everything seems reasonable let's move on
    return HM.calc_kHM(lambda_array=lambda_array,spectra_lambda=spectra_lambda,distance=distance,
        return_sigma=return_sigma,return_sigma_only=return_sigma_only)





def compute_kH2p(lambda_array,spectra_lambda,distance=1.*__u.kpc,ngas=1e2*__u.cm**-3,Tgas=1e3*__u.K,
    Dunn=False,Zammit=True,
    return_sigma=False,return_heating=False,
    return_sigma_only=False,return_heating_only=False):
    
    '''
    Calculate the photodissociation rate of H2p and its associated heating rate.
    One rate for each spectrum.
    Input:
        lambda_array: wavelength array associated with the spectra    [A]
        spectra_lambda: spectra                                       [erg/A/s]
        distance: distance of the radiating source                    [kpc]
        ngas: gas number density                                      [1/cm^3]
        Tgas: gas temperature                                         [K]
        Dunn: boolean, True if you want to use the Frank-Condon distrubution of level population [v=0-18,J=1]
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
    
    if type(distance)!=__u.Quantity:
        distance=distance*__u.kpc
    if type(ngas)!=__u.Quantity:
        ngas=ngas*__u.cm**-3
    if type(Tgas)!=__u.Quantity:
        Tgas=Tgas*__u.K

    if distance.value<=0:
        print('wrong distance dumbass!')
        return -1
    if ngas.value<=0:
        print('wrong gas density dumbass!')
        return -1
    if Tgas.value<=0:
        print('wrong gas temperature dumbass!')
        return -1
    
# if everything seems reasonable let's move on
    return H2p.calc_kH2p(lambda_array=lambda_array,spectra_lambda=spectra_lambda,distance=distance,ngas=ngas,Tgas=Tgas,
        Dunn=Dunn,Zammit=Zammit,
        return_sigma=return_sigma,return_heating=return_heating,
        return_sigma_only=return_sigma_only,return_heating_only=return_heating_only)




def compute_kH2(lambda_array,spectra_lambda,distance=1.*__u.kpc,ngas=1e2*__u.cm**-3,Tgas=1e3*__u.K,
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
    
    if type(distance)!=__u.Quantity:
        distance=distance*__u.kpc
    if type(ngas)!=__u.Quantity:
        ngas=ngas*__u.cm**-3
    if type(Tgas)!=__u.Quantity:
        Tgas=Tgas*__u.K

    if distance.value<=0:
        print('wrong distance dumbass!')
        return -1
    if ngas.value<=0:
        print('wrong gas density dumbass!')
        return -1
    if Tgas.value<=0:
        print('wrong gas temperature dumbass!')
        return -1
    
# if everything seems reasonable let's move on
    return H2.calc_kH2(lambda_array=lambda_array,spectra_lambda=spectra_lambda,distance=distance,ngas=ngas,Tgas=Tgas,
        return_sigma=return_sigma,return_heating=return_heating,return_sigma_only=return_sigma_only,return_heating_only=return_heating_only,
        db_touse=db_touse,exstates_touse=exstates_touse,lineprofile_touse=lineprofile_touse,
        thresh_Xlevel=thresh_Xlevel,thresh_oscxfdiss=thresh_oscxfdiss,thresh_osc=thresh_osc,thresh_fdiss=thresh_fdiss)