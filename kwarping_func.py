import numpy as np
import xarray as xr
from ipywidgets import interact
import ipywidgets as widgets
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def diracangle_expected(E_kin_dirac):
    k_dirac = 1.7 # Position of the Dirac point in momentum space in 1/Aa
    theta_expected = np.arcsin(k_dirac / 0.5123 / np.sqrt(E_kin_dirac))
    return theta_expected

def theta_to_k(measured_angle, theta_0, measured_energy):
    '''
    Convert measured angle (in degrees) to momentum (k-space)

    measured_angle : array of measured angles in degrees
    theta_0 : angular offset
    measured_energy : array of measured kinetic energies in eV
    '''
    k = 0.5123 * np.sin(np.radians(measured_angle + theta_0)) * np.sqrt(measured_energy)
    return k

def k_to_angle(k, theta_0, measured_energy):
    '''
    Convert momentum (k-space) to angle (in degrees)

    k : momentum value i 1/Å
    theta_0 : angular offset
    measured_energy : measured kinetic energy in eV
    '''
    angle = np.degrees(np.arcsin(k / (0.5123 * np.sqrt(measured_energy)))) - theta_0
    return angle

def kwarping_scan(scan, theta_0):
    '''
    Convert the angle coordinates of a scan to k-space

    scan : xarray DataArray object
    theta_0 : angular offset

    Returns:
    data_out : xarray DataArray object, The scan data interpolated in k-space
    '''

    # Get coordinate axes of the scan
    theta_par = scan.coords['theta_par'].values
    KE = scan.coords['eV'].values

    # Convert angle coordinates to k-space for every angle-energy coordinate point
    kx= theta_to_k(theta_par[:, np.newaxis], theta_0, KE[np.newaxis, :])

    # Grid of k-values for interpolation
    k_values = np.linspace(kx.min(), kx.max(), len(theta_par))

    # Convert k-values back to angles for interpolation
    theta_out = k_to_angle(k_values[:, np.newaxis], theta_0, KE[np.newaxis, :])

    # Create xarray DataArray for energy coordinates (eV)
    eV_xarray = xr.DataArray(
            KE,
            dims=['eV'],
            coords={'eV': KE})

    # Create xarray DataArray for converted angles (from k-space)
    theta_from_k_xarray = xr.DataArray(
            theta_out*1 ,
            dims=['k_par', 'eV'],
            coords={'k_par': k_values, 'eV': scan.eV.data})

    # Interpolate the original scan data based on k-space and energy
    data_out = scan.interp(eV=eV_xarray, theta_par=theta_from_k_xarray)

    return data_out

def fermishift_scan(scan, fermienergy):
    '''
    Shift the energy coordinates of a scan by subtracting the Fermi energy
    scan : xarray DataArray object
    fermienergy : Fermi energy in eV
    '''

    scan_shifted = scan.copy()
    scan_shifted.coords['eV'] = scan_shifted.coords['eV'] - fermienergy
    return scan_shifted

def kwarping_scansx(scansx, theta_0, fermienergy):
    '''
    Convert the angle coordinates of a list of scans to k-space
    and shift the energy coordinates by subtracting the Fermi energy

    scansx : list of xarray DataArray objects
    theta_0 : angular offset

    Returns:
    scansx: list of xarray DataArray objects
    '''
    # Create a copy of the input list
    scansx_kwarped = scansx.copy()

    # Iterate over the list of scans
    for i, scan in enumerate(scansx):
        # Perform kwarping on the current scan
        kwarped_scan = kwarping_scan(scan, theta_0)
        scansx_kwarped[i] = fermishift_scan(kwarped_scan, fermienergy)

    return scansx_kwarped