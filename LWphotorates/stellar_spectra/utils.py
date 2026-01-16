from typing import Union
from pathlib import Path
import xarray as xr
import numpy as np
from astropy import units as au

DATA_DIR = Path("/cephfs/andrea/stellar_spectra")


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
        The Yggdrasil spectra, saved as an xarray.Dataset with a single variable.
    """
    OUT_STELLAR_MASS = 1e6 * au.Msun

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