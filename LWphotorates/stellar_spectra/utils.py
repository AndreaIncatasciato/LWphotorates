from typing import Union
from pathlib import Path
import xarray as xr
import numpy as np
from astropy import units as au
from astropy import constants as ac
from hoki import load as bpass_load
import gzip
import shutil

DATA_DIR = Path("/cephfs/andrea/stellar_spectra")
OUT_STELLAR_MASS = 1e6 * au.Msun
BPASS_LUMINOSITY_CONVERSION_CONSTANT = ac.L_sun.to(au.erg/au.s) / au.AA


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