from typing import Union
from pathlib import Path
import xarray as xr
import numpy as np
from astropy import units as au
from astropy import constants as ac
from astropy.units import Quantity
from hoki import load as bpass_load
import gzip
import shutil
from LWphotorates.utils import generate_blackbody_spectrum, get_ioniz_energy_hydrogen
from LWphotorates.H2 import get_reaction_min_energy as get_min_lw_energy

DATA_DIR = Path("/cephfs/andrea/stellar_spectra")
OUT_STELLAR_MASS = 1e6 * au.Msun
BPASS_LUMINOSITY_CONVERSION_CONSTANT = ac.L_sun.to(au.erg/au.s) / au.AA
MIN_ENERGY_FOR_BLACKBODY = 0.1 * au.eV


def convert_yggdrasil_spec_to_xr(
        in_file_path: Path, ds_description: str, out_file_path: Union[Path, None] = None) -> xr.Dataset:
    """
    Convert a Yggdrasil series of spectra from plain txt to an xarray.Dataset.

    Yggdrasil spectra are currently obtained from https://www.astro.uu.se/~ez/yggdrasil/yggdrasil.html) as plain txt files.
    This function reads the file, loads into a xarray.Dataset and, if indicated, saves it to a netCDF file.

    Note that originally the spectra are for a stellar population with initial mass that depends on IMF: normally for PopIII stars they use approx 1 Msun.
    The function also multiplies the spectra to return the luminosity of a stellar population of total mass 1e6 Msun,
    which is the standard in SSP codes.

    Parameters
    ----------
    in_file_path : Path
        The path to the input file.
    ds_description : str
        The description to add to the xarray.Dataset as attribute.
    out_file_path : Union[Path, None], optional
        If not None, the xarray.Dataset is saved at this location.

    Returns
    -------
    xr.Dataset
        The Yggdrasil spectra, saved as an xarray.Dataset with a single variable and two coordinates (age and wavelength).
    """
    rows = []
    current_age = None

    with open(in_file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Mass available"):
                stellar_mass = float(line.split(":")[1]) * au.Msun
                continue

            if line.startswith("Age"):
                current_age = float(line.split(":")[1])
                continue

            parts = line.split()

            if len(parts) == 2:
                try:
                    wl = float(parts[0])
                    spec = float(parts[1])
                    rows.append([current_age, wl, spec])
                except ValueError:
                    pass

    age_array, wl_array, spec_array = np.array(rows).T
    age_array = np.unique(age_array)
    wl_array = np.unique(wl_array)
    spec_array = np.reshape(spec_array, newshape=(len(age_array), len(wl_array)))

    ds = xr.Dataset(
        data_vars=dict(
            luminosity=(
                ["age", "wavelength"], spec_array * OUT_STELLAR_MASS / stellar_mass,
                {"units": (au.erg / au.s / au.AA).to_string(), "stellar_mass": OUT_STELLAR_MASS.value}),
        ),
        coords=dict(
            age=(["age"], age_array, {"units": au.Myr.to_string()}),
            wavelength=(["wavelength"], wl_array, {"units": au.AA.to_string()}),
        ),
        attrs=dict(description=ds_description)
    )

    if out_file_path is not None:
        ds.to_netcdf(out_file_path)

    return ds


def convert_bpass_spec_to_xr(imf: str, metallicity: str, multiplicity: str) -> xr.Dataset:
    """
    Convert a BPASS series of spectra to an xarray.Dataset.

    BPASS spectra files can be currently read with the hoki library, that returns a pandas.DataFrame.
    This function manipulates the DataFrame, loads into a xarray.Dataset and saves it to a netCDF file.

    Parameters
    ----------
    imf : str
        The stellar IMF of interest. For the available IMFs see Table 1 in Stanway & Eldridge (2018).
    metallicity : str
        The metallicity of the stellar population of interest. For the available metallicities see Table 1 in the BPASS v2.2 manual (p.8).
    multiplicity : str
        The multiplicity of interest. Choose between "sin" (single stars) and "bin" (binaries).

    Returns
    -------
    xr.Dataset
        The BPASS spectra, saved as an xarray.Dataset with a single variable and two coordinates (age and wavelength).
    """
    file_name = f"spectra-{multiplicity}-{imf}.{metallicity}"
    in_file_path = DATA_DIR / "bpass" / (file_name + ".dat.gz")
    uncompressed_file_path = DATA_DIR / "bpass" / (file_name + ".dat")
    out_file_path = DATA_DIR / "bpass" / (file_name + ".nc")

    # make sure the uncompressed file is available
    if not uncompressed_file_path.is_file():
        with gzip.open(in_file_path, "rb") as f_in:
            with open(uncompressed_file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    # load the file and and manipulate the dataframe
    df = bpass_load.model_output((uncompressed_file_path).as_posix())
    wl_array = df["WL"].to_numpy()
    df = df.drop(columns="WL")
    age_array = [np.power(10, float(age_column) - 6) for age_column in df.columns]
    spectra_array = df.to_numpy().T * BPASS_LUMINOSITY_CONVERSION_CONSTANT

    # save into an xarray.Dataset
    ds_description = (
        f"BPASS v2.2.1 spectra, IMF: {imf}, metallicity: {metallicity}, multiplicity: {multiplicity}. "
        "For more info look here: https://warwick.ac.uk/fac/sci/physics/research/astro/research/catalogues/bpass/v2p2/."
    )
    ds = xr.Dataset(
        data_vars=dict(
            luminosity=(
                ["age", "wavelength"], spectra_array,
                {"units": (au.erg / au.s / au.AA).to_string(), "stellar_mass": OUT_STELLAR_MASS.value}),
        ),
        coords=dict(
            age=(["age"], age_array, {"units": au.Myr.to_string()}),
            wavelength=(["wavelength"], wl_array, {"units": au.AA.to_string()}),
        ),
        attrs=dict(description=ds_description)
    )

    # save the xarray.Dataset to a netCDF file and delete the uncompressed file
    ds.to_netcdf(out_file_path)
    uncompressed_file_path.unlink()

    return ds


def normalise_blackbody_spectrum(
    blackbody_temperature: Union[float, Quantity],
    lw_photon_count_per_baryon: float,
    max_stellar_age: Union[float, Quantity] = 5 * au.Myr,
    stellar_mass: Union[float, Quantity] = OUT_STELLAR_MASS,
) -> tuple:
    """
    Normalise a blackbody spectrum such that the number of LW photons emitted per baryon equals to the input value.

    This is a common assumption in the literature, in order to find a sensible normalisation to a blackbody spectrum
    that is intended to mimic a more complex stellar spectrum.

    For example, Greif and Bromm (2006) assign PopIII stars with 1e5 K blackbody spectra, normalised such that they emit
    2e4 LW photons per baryon in their lifetime (assumed to be 5 Myr).
    Similarly, for PopII stars they use 1e4 K blackbody spectra that emit 4e3 LW photons per baryon in their lifetime (again 5 Myr).

    Parameters
    ----------
    blackbody_temperature : Union[float, Quantity]
        The temperature of the blackbody radiation, that sets the spectral shape. If a float is passed, the code assumes that it is in Kelvin.
    lw_photon_count_per_baryon : float
        The LW photons emitted per baryon throughout the lifetime of the stellar population.
    max_stellar_age : Union[float, Quantity], optional
        The maximum stellar age considered. Before it the stellar population is assumed to have a constant LW emission.
        Past this age the population does not emit any LW photon. If a float is passed, the code assumes that it is in Myr. By default 5 Myr.
    stellar_mass : Union[float, Quantity], optional
        The total mass of the stellar population. If a float is passed, the code assumes that it is solar masses. By default OUT_STELLAR_MASS (currently 1e6 Msun).

    Returns
    -------
    tuple[Quantity]
        The energy array and the normalised spectrum (in units of monochromatic luminosity).
    """
    if isinstance(blackbody_temperature, float):
        blackbody_temperature *= au.K
    if isinstance(max_stellar_age, float):
        stellar_mass *= au.Myr
    if isinstance(stellar_mass, float):
        stellar_mass *= au.M_sun
    
    # transform lw_photon_count_per_baryon to a photon rate (# of photons per baryon per second),
    # assuming that the LW photon rate is constant throughtout the stellar lifetime
    target_photon_rate = lw_photon_count_per_baryon * stellar_mass.to(au.kg) / ac.m_p / max_stellar_age.to(au.s)

    # generate the BB spectrum
    # (generate_blackbody_spectrum returns an intensity by default, here the units can be ignored and set manually
    # to a luminosity, given that we are interested in the spectral shape and not in the absolute values)
    energy_array = np.linspace(
        MIN_ENERGY_FOR_BLACKBODY, get_ioniz_energy_hydrogen(), 10000)
    frequency_array = energy_array / ac.h.to(au.eV / au.Hz)
    spectrum_array = generate_blackbody_spectrum(blackbody_temperature, energy_array).value * au.erg / au.Hz / au.s

    # integrate the spectrum only in the LW energy range
    min_lw_energy = get_min_lw_energy()
    max_lw_energy = get_ioniz_energy_hydrogen()
    lw_mask = (energy_array >= min_lw_energy) & (energy_array <= max_lw_energy)
    current_photon_rate = np.trapz(
        spectrum_array[lw_mask] / energy_array[lw_mask],
        frequency_array[lw_mask]).to(1 / au.s)
    spectrum_normalisation = target_photon_rate / current_photon_rate
    spectrum_array *= spectrum_normalisation

    return energy_array, spectrum_array
