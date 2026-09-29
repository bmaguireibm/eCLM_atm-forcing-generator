#!/usr/bin/env python3
"""Create monthly ERA5 input from the ECMWF ARCO Zarr store.

ERA5 single-level data is read directly from the ECMWF ARCO geochunked-chunked
Zarr store.  The CDS API fallback is retained for the seasonal datasets used
by the SEAS5 workflow.
"""
import argparse
import calendar
import os
import tempfile

import numpy as np
import xarray as xr


ERA5_ZARR_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr"
)

ERA5_VARIABLES = {
    "10m_u_component_of_wind": ("u10", "u10"),
    "10m_v_component_of_wind": ("v10", "v10"),
    "2m_dewpoint_temperature": ("d2m", "d2m"),
    "2m_temperature": ("t2m", "t2m"),
    "surface_pressure": ("sp", "sp"),
    "mean_surface_downward_long_wave_radiation_flux": ("strd", "avg_sdlwrf"),
    "mean_surface_downward_short_wave_radiation_flux": ("ssrd", "avg_sdswrf"),
    "mean_total_precipitation_rate": ("tp", "avg_tprate"),
}
CDS_DEFAULT_VARIABLES = [
    "surface_pressure",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_total_precipitation_rate",
]


def default_request(year, monthstr, days, variables=None):
    """Return the default ERA5 request in CDS variable names."""
    return {
        "product_type": ["reanalysis"],
        "variable": list(variables or ERA5_VARIABLES),
        "year": [str(year)],
        "month": [monthstr],
        "day": days,
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [74, -42, 20, 69],
    }


def generate_days(year, month):
    """Get the number of days in a given month and year.

    Args:
        year (int): Year
        month (int): Month (1-12)

    Returns:
        list: List of day numbers for the month
    """
    # Get the number of days in the given month
    num_days = calendar.monthrange(year, month)[1]

    # Generate the list of days as integers
    days = [day for day in range(1, num_days + 1)]

    return days


def get_arco_api_key():
    """Get the ARCO bearer token from the environment or CDS config."""
    api_key = os.environ.get("ARCO_API_KEY") or os.environ.get("CDSAPI_KEY")
    if api_key:
        return api_key

    config_path = os.environ.get("CDSAPI_CONFIG") or os.path.expanduser("~/.cdsapirc")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            for line in config_file:
                name, separator, value = line.partition(":")
                if separator and name.strip() == "key":
                    api_key = value.strip().strip("'\"")
                    if api_key:
                        return api_key
    except FileNotFoundError:
        pass

    raise RuntimeError(
        "An API key is required for the ECMWF ARCO store; set ARCO_API_KEY "
        "or CDSAPI_KEY, or configure the key in ~/.cdsapirc."
    )


def detect_file_type(filepath):
    """Detect if downloaded file is NetCDF, GRIB or ZIP format.

    Args:
        filepath (str): Path to the downloaded file

    Returns:
        str: File extension ('.nc' for NetCDF, '.zip' for ZIP, '.grib' for GRIB)

    Raises:
        ValueError: If file format is not recognized as NetCDF, GRIB, or ZIP
    """
    # Read file magic bytes
    with open(filepath, 'rb') as f:
        magic = f.read(8)

    # ZIP files start with 'PK' (0x504B)
    if magic[:2] == b'PK':
        return '.zip'

    # NetCDF files start with 'CDF' (0x43444601 or 0x43444602) or HDF5 signature
    if magic[:3] == b'CDF' or magic[:4] == b'\x89HDF':
        return '.nc'

    # GRIB files start with 'GRIB'
    if magic[:4] == b'GRIB':
        return '.grib'

    # If we reach here, the file format is not recognized
    magic_hex = magic.hex()
    raise ValueError(
        f"Unrecognized file format for '{filepath}'. "
        f"Magic bytes: {magic_hex}. "
        f"Expected NetCDF (CDF/HDF5), GRIB, or ZIP (PK) format."
    )


def generate_datarequest(year, monthstr, days,
                         dataset="reanalysis-era5-single-levels",
                         request=None,
                         target=None):
    """Generate and execute ERA5 data download request.

    "ERA5 hourly data on single levels from 1940 to present":
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview

    Args:
        year (int): Year to download
        monthstr (str): Month as zero-padded string (e.g., '07')
        days (list): List of days in the month
        dataset (str, optional): CDS dataset name. Defaults to 'reanalysis-era5-single-levels'.
        request (dict, optional): Custom CDS request dictionary. If None, uses default request.
        target (str, optional): Output filename. If None, detects filetype and sets extension.

    Returns:
        str: Path to downloaded file
    """

    import cdsapi

    # Active download client for the seasonal datasets that do not exist in
    # the ERA5 ARCO store.
    client = cdsapi.Client()

    # Default request if not provided
    if request is None:
        request = default_request(year, monthstr, days, CDS_DEFAULT_VARIABLES)
    else:
        request = request.copy()
        # Adapt year, month and day to input values
        request["year"] = [str(year)]
        request["month"] = [monthstr]
        if dataset == "seasonal-original-single-levels":
            # First day of month specified for SEAS5
            request["day"] = ["01"]
        else:
            request["day"] = days

    # Temporary filename w/o extension if not provided
    auto_detect_extension = target is None
    if auto_detect_extension:
        # Create a temporary file for download
        temp_fd, target = tempfile.mkstemp(
            prefix=f'download_era5_{year}_{monthstr}',
            dir='.')
        os.close(temp_fd)  # Close the file descriptor

    # Get the data from cds
    client.retrieve(dataset, request, target)

    # If target was not provided, detect the file type after download
    if auto_detect_extension:
        # Detect the actual file type
        extension = detect_file_type(target)

        # Rename to clean predictable filename with correct extension
        final_target = f'download_era5_{year}_{monthstr}{extension}'
        os.rename(target, final_target)
        target = final_target

    return target


def generate_arco_datarequest(year, monthstr, days, request=None):
    """Read one month of ERA5 from ARCO and write the legacy CDS archive.

    The archive contains the same instant and average NetCDF members expected
    by ``prepare_ERA5_input.sh``.  ARCO's ``ssrd``, ``strd`` and ``tp`` are
    hourly accumulations, whereas the CDS request uses hourly mean fluxes or
    rates, so those three fields are converted here.
    """
    if request is None:
        request = default_request(year, monthstr, days)
    else:
        request = request.copy()

    api_key = get_arco_api_key()

    requested_variables = request.get("variable", list(ERA5_VARIABLES))
    unsupported = [name for name in requested_variables if name not in ERA5_VARIABLES]
    if unsupported:
        raise ValueError(f"Variables are not available in the ERA5 ARCO store: {unsupported}")

    # The CDS area is [north, west, south, east].  ARCO contains the full
    # grid, so preserve the old single-grid-point request by selecting its
    # centre and the nearest ERA5 grid point.
    area = request.get("area")
    if area is None or len(area) != 4:
        raise ValueError("An ERA5 request area [north, west, south, east] is required")
    latitude = (float(area[0]) + float(area[2])) / 2
    longitude = (float(area[1]) + float(area[3])) / 2

    dataset = xr.open_zarr(
        ERA5_ZARR_URL,
        consolidated=True,
        storage_options={"headers": {"Authorization": f"Bearer {api_key}"}},
        chunks={},
    )

    days = [int(day) for day in days]
    last_day = calendar.monthrange(year, int(monthstr))[1]
    start = f"{year}-{monthstr}-01T00:00"
    end = f"{year}-{monthstr}-{last_day:02d}T23:00"
    dataset = dataset.sel(time=slice(start, end))
    dataset = dataset.where(dataset.time.dt.day.isin(days), drop=True)

    requested_times = request.get("time")
    if requested_times:
        requested_hours = [
            int(str(value).split(":", 1)[0]) for value in requested_times
        ]
        dataset = dataset.where(dataset.time.dt.hour.isin(requested_hours), drop=True)

    selected = dataset.sel(latitude=latitude, longitude=longitude, method="nearest")
    selected_latitude = float(selected.latitude)
    selected_longitude = float(selected.longitude)
    selected = selected[[ERA5_VARIABLES[name][0] for name in requested_variables]]

    # Materialize the small point selection before closing the remote store;
    # this keeps all remote reads within this function and produces ordinary
    # NumPy-backed NetCDF output.
    selected = selected.load()
    selected = selected.expand_dims(
        latitude=[selected_latitude], longitude=[selected_longitude]
    ).transpose("time", "latitude", "longitude")
    selected = selected.rename({"time": "valid_time"})
    selected["number"] = xr.DataArray(np.int32(0))
    selected["expver"] = xr.DataArray(
        np.full(selected.sizes["valid_time"], "0001"), dims="valid_time"
    )

    instant = {}
    average = {}
    for source_name in requested_variables:
        source, target = ERA5_VARIABLES[source_name]
        variable = selected[source]
        if target in {"avg_sdlwrf", "avg_sdswrf", "avg_tprate"}:
            # ARCO stores hourly accumulations: radiation in J m-2 and
            # precipitation in metres of water.  The CDS precipitation rate
            # is kg m-2 s-1, hence the water-density conversion for tp.
            variable = variable * (1000.0 if target == "avg_tprate" else 1.0) / 3600.0
            variable.attrs = dict(variable.attrs)
            variable.attrs["units"] = (
                "kg m**-2 s**-1" if target == "avg_tprate" else "W m**-2"
            )
            average[target] = variable
        else:
            instant[target] = variable

    output_base = f"download_era5_{year}_{monthstr}"
    archive_path = f"{output_base}.zip"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.abspath(temp_dir)
        instant_path = os.path.join(temp_path, "data_stream-oper_stepType-instant.nc")
        average_path = os.path.join(temp_path, "data_stream-oper_stepType-avg.nc")

        for variables, path in ((instant, instant_path), (average, average_path)):
            xr.Dataset(variables, coords={
                "valid_time": selected.valid_time,
                "latitude": selected.latitude,
                "longitude": selected.longitude,
                "number": selected.number,
                "expver": selected.expver,
            }).to_netcdf(path, engine="netcdf4")

        import zipfile
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(instant_path, os.path.basename(instant_path))
            archive.write(average_path, os.path.basename(average_path))

    return archive_path


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Download ERA5 reanalysis data from Copernicus Climate Data Store (CDS)."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=False,
        default=None,
        help="Year to download (e.g., 2017). Required for default request, optional for custom request (uses year from custom request if not provided).",
    )
    parser.add_argument(
        "--month",
        type=int,
        required=False,
        default=None,
        help="Month to download (1-12). Required for default request, optional for custom request (uses month from custom request if not provided)."
    )
    parser.add_argument(
        "--day",
        type=str,
        required=False,
        default=None,
        help="Day(s) to download as comma-separated values (e.g., '15' or '1,15,30'). If not provided, all days in the month are downloaded."
    )
    parser.add_argument(
        "--dirout",
        type=str,
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--request",
        type=str,
        required=False,
        help="Path to Python file defining custom 'request' variable"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="reanalysis-era5-single-levels",
        help="CDS dataset name (default: reanalysis-era5-single-levels)"
    )

    # Parse command-line arguments
    args = parser.parse_args()
    year = args.year
    month = args.month
    dirout = args.dirout

    # Ensure the output directory exists, if not, create it
    if not os.path.exists(dirout):
        os.makedirs(dirout)

    # Resolve request path before changing directory
    request_path = os.path.abspath(args.request) if args.request else None

    # change to output directory
    os.chdir(dirout)

    # Load custom request if provided
    custom_request = None
    custom_dataset = args.dataset
    if args.request:
        if not os.path.isfile(args.request):
            raise FileNotFoundError(f"Custom request file not found: {args.request}")

        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_request_module", request_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        if hasattr(custom_module, 'request'):
            custom_request = custom_module.request
            print(f"Loaded custom request from: {request_path}")
        else:
            print(f"Warning: No 'request' variable found in {request_path}, using default")
        if hasattr(custom_module, 'dataset'):
            custom_dataset = custom_module.dataset
            print(f"Loaded custom dataset from: {request_path}")

    # Handle year: extract from custom request if not provided
    if year is None:
        if custom_request and "year" in custom_request:
            year_from_request = custom_request["year"]
            if isinstance(year_from_request, list):
                year = int(year_from_request[0])
            else:
                year = int(year_from_request)
            print(f"Using year from custom request: {year}")
        else:
            raise ValueError(
                "Year is required. Provide it either as --year argument "
                "or in the custom request file."
            )

    # Handle month: extract from custom request if not provided
    if month is None:
        if custom_request and "month" in custom_request:
            month_from_request = custom_request["month"]
            if isinstance(month_from_request, list):
                month = int(month_from_request[0])
            else:
                month = int(month_from_request)
            print(f"Using month from custom request: {month}")
        else:
            raise ValueError(
                "Month is required. Provide it either as --month argument "
                "or in the custom request file."
            )

    # Format the month with a leading zero if needed
    monthstr = f"{month:02d}"

    # Handle day: parse from argument, extract from custom request, or compute all days
    if args.day is not None:
        # Parse comma-separated day values
        days = [int(d.strip()) for d in args.day.split(',')]
        print(f"Using days from argument: {days}")
    elif custom_request and "day" in custom_request:
        # Extract from custom request
        day_from_request = custom_request["day"]
        if isinstance(day_from_request, list):
            days = [int(d) for d in day_from_request]
        else:
            days = [int(day_from_request)]
        print(f"Using days from custom request: {days}")
    else:
        # Compute all days in the month
        days = generate_days(year, month)
        print(f"Using all days in month: {len(days)} days")

    print(f"Preparing ERA5 data for {year}-{monthstr}")
    print(f"Dataset: {custom_dataset}")
    print(f"Output directory: {os.getcwd()}")

    if custom_dataset == "reanalysis-era5-single-levels":
        target = generate_arco_datarequest(
            year, monthstr, days, request=custom_request
        )
        print(f"ARCO read complete: {os.path.abspath(target)}")
    else:
        target = generate_datarequest(
            year, monthstr, days, dataset=custom_dataset, request=custom_request
        )
        print(f"Download complete: {os.path.abspath(target)}")
