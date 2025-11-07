##from google.colab import drive
##from google.colab import files

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.ndimage
from ipywidgets import interact
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import ListedColormap 
import ipywidgets as widgets
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime

linestyles = ['-', '--', ':', '-.', '.']


color_blind_friendly_colors = [
    "#004488",  # dark blue
    "#DDAA33",  # mustard (yellow-brown)
    "#BB5566",  # soft red
    "#6688CC",  # light blue
    "#117733",  # dark green
    "#EE8866",  # peach
    "#949494",  # gray   
]

# function for clustering color map using colour blind friendly colours index map:
def create_custom_cmap(n_clusters):
    cluster_array = np.arange(0, n_clusters, 1 , dtype=int)
    index_map = {}
    for i  in cluster_array: 
        index_map[i]=i
    reordered_colors = [color_blind_friendly_colors[index_map[i]] for i in range(n_clusters)]
    return ListedColormap(reordered_colors)


# gaussian filter 1d for non-uniform spacing
def gaussian_weighted_average(x, y, sigma, density_range=1):
    """
    Apply a Gaussian-weighted average to irregularly spaced (x, y) data.

    Parameters:
    - x: array-like, the x-values of the data points (irregularly spaced).
    - y: array-like, the corresponding y-values of the data points.
    - sigma: float, the standard deviation of the Gaussian filter in unit of the x-values
    - density_range

    Returns:
    - y_filtered: array-like, the Gaussian-weighted average values at the original x points.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # pad the dataset beyond the boundary (a a a a | a b c d | d d d d)
    n_pad_start = int(np.ceil (4 * sigma / (x[1] - x[0])))
    n_pad_end = int(np.ceil (4 * sigma / (x[-1] - x[-2])))

    # Create padded x array with constant spacing
    x_padded_start = [x[0] - i * (x[1] - x[0]) for i in range(n_pad_start, 0, -1)]
    x_padded_end = [x[-1] + i * (x[-1] - x[-2]) for i in range(1, n_pad_end + 1)]
    x_padded = np.concatenate((x_padded_start, x, x_padded_end))
    # Create padded y array by replicating the boundary values
    y_padded = np.concatenate(([y[0]] * n_pad_start, y, [y[-1]] * n_pad_end))

    y_filtered = np.zeros_like(x_padded, dtype=np.float64)
    weights_j = np.zeros_like(x_padded, dtype=np.float64)

    for j, xj in enumerate(x_padded):
        # Calculate Gaussian weights for all points relative to xj
        for i, xi in enumerate(x_padded):
            weights_j[i] = np.exp(-((xi - xj) / sigma)**2)

            # adjust weights to account for varying density of x-points (=numberOfXPOints per x-axisUnit)
            if i < density_range:
                x_density = 2.0 / (x_padded[i+density_range] - x_padded[i])
                weights_j[i] /= x_density    
            elif i >= len(x) - density_range:
                x_density = 2.0 / (x_padded[i] - x_padded[i-density_range])
                weights_j[i] /= x_density
            else:
                x_density = 1.0 / (x_padded[i+density_range] - x_padded[i-density_range]) if density_range > 0 else 1.0
                weights_j[i] /= x_density
    
        y_filtered[j] = np.sum(weights_j * y_padded) / np.sum(weights_j)

    # remove padding
    y_filtered = y_filtered[n_pad_start:-n_pad_end]

    return y_filtered

# Smoothing and normalization function 
def smooth_and_normalize_gwa(trace, t_axis, sigma=140):
    smoothed_trace = gaussian_weighted_average(t_axis, trace, sigma)
    
    max_val = np.max(smoothed_trace)
    start_intensity = np.mean(smoothed_trace[:5])

    if max_val == 0:
        normalized_trace = np.zeros_like(trace)
    else:
        # normalize to a maximum of 1
        if (max_val - start_intensity) != 0:
          normalized_trace = (smoothed_trace - start_intensity) / (max_val - start_intensity)
        else:
          normalized_trace = smoothed_trace - start_intensity

    return normalized_trace

def process_dataset_gwa(scansx, roi_kx, roi_ky, t_axis):
    # Ensure scansx is a list of xarray DataArrays
    if not all(isinstance(scan, xr.DataArray) for scan in scansx):
        raise ValueError("All items in scansx should be xarray DataArrays")

    dimt = len(scansx)  # Number of time points
    dimx, dimy = scansx[0].shape  # Assuming all DataArrays have the same shape

    data = np.array([scan.values for scan in scansx])  # Convert to numpy array
    data = np.nan_to_num(data)  # Remove NaNs

    # Smooth data
    smoothed_data = np.empty_like(data)
    for i in range(data.shape[0]):
        smoothed_data[i] = scipy.ndimage.gaussian_filter(data[i], sigma=(2,6))
    data = smoothed_data

    data = data[:, roi_kx[0]:roi_kx[1], roi_ky[0]:roi_ky[1]]  # Truncate data

    # Prepare for ROI calculation
    num_rois_kx = (roi_kx[1] - roi_kx[0]) // roi_kx[2]
    num_rois_ky = (roi_ky[1] - roi_ky[0]) // roi_ky[2]
    n = roi_kx[2]
    m = roi_ky[2]

    # Calculate integrated intensities
    intensities = np.zeros((num_rois_kx * num_rois_ky, data.shape[0]))
    for idx, image in enumerate(data):
        roi_count = 0
        for roi_x in range(num_rois_kx):
            for roi_y in range(num_rois_ky):
                roi = image[(roi_x)*n:(roi_x+1)*n, (roi_y)*m:(roi_y+1)*m]
                intensities[roi_count, idx] = np.sum(roi)
                roi_count += 1

    # Normalize the intensities
    max_intensity = np.max(intensities)
    intensities = (intensities / max_intensity) * 100

    normalized_intensities = np.array([
        smooth_and_normalize_gwa(trace, t_axis, sigma=29.73) for trace in intensities
    ])

    # time-smoothing of intensities
    t_smoothed_intensities = np.array([
        gaussian_weighted_average(t_axis, trace, sigma=29.73) for trace in intensities
    ])
    t_smoothed_intensities = (t_smoothed_intensities / np.max(t_smoothed_intensities)) * 100

    return t_smoothed_intensities, normalized_intensities

def run_kmeans_clustering_with_threshold_gwa(n_clusters, suffixes, roi_dict, scansx, threshold, t_axis):
    """
    Run KMeans clustering on intensities and normalized intensities with a threshold for multiple suffixes
    (where kmeans has been applied to filtered_intensities)
    After applying kmeans, the non_exceeding datapoints are inserted again with a common cluster index

    Args:
        n_clusters (int): Number of clusters for KMeans.
        suffixes (list of str): List of suffixes to iterate over.
        roi_dict (dict): Dictionary containing region of interest (ROI) definitions.
        scansx (list of xarray.DataArray): List of xarray.DataArrays representing dispersion at different time points.
        threshold (float): Intensity threshold for selecting curves.

    Returns:
        dict: Results dictionary containing intensities, normalized intensities, and KMeans models for each suffix.
    """
    results = {}  # This dictionary will store all results keyed by suffix

    for suffix in suffixes:
        # Access roi_e and roi_k from roi_dict using the suffix
        roi_e = roi_dict[f'roi_e{suffix}']
        roi_k = roi_dict['roi_k']

        # Process the dataset and extract intensities
        intensities, norm_intensities = process_dataset_gwa(scansx, roi_k, roi_e, t_axis)

        # Identify curves that exceed the threshold
        max_intensities = np.max(np.abs(intensities), axis=1)
        exceeds_threshold = max_intensities > threshold
        valid_indices = np.where(exceeds_threshold)[0]
        
        # Handle cases where no curves exceed the threshold
        if len(valid_indices) == 0:
            print(f"No curves exceed the threshold for suffix {suffix}")
            results[f'results{suffix}'] = {
                'intensities': np.array([]),
                'normalized_intensities': np.array([]),
                'kmeans_all': None,
                'kmeans_norm': None,
                'clustered_curves_indices': []
            }
            continue

        # Filter intensities and normalized_intensities for clustering
        filtered_intensities = intensities[valid_indices]
        filtered_norm_intensities = norm_intensities[valid_indices]

        # Ensure dimensions are consistent with clustering requirements
        num_clusters = min(n_clusters, len(valid_indices))  # Avoid too many clusters

        # Perform K-means clustering on filtered data
        kmeans_intensities = KMeans(n_clusters=num_clusters, n_init=400, random_state=0).fit(filtered_intensities)
        kmeans_norm_intensities = KMeans(n_clusters=num_clusters, n_init=400, random_state=0).fit(filtered_norm_intensities)
        
        # Store the results
        results[f'results{suffix}'] = {
            'intensities': intensities,
            'normalized_intensities': norm_intensities,
            'kmeans_all': kmeans_intensities,
            'kmeans_norm': kmeans_norm_intensities,
            'clustered_curves_indices': valid_indices
        }

        #insert cluster indices for the non_exceeding curves
        #- Retrieve the ROI for energy and k values
        roi_e = roi_dict[f'roi_e{suffix}']
        roi_k = roi_dict['roi_k']
        num_rois_e = (roi_e[1] - roi_e[0]) // roi_e[2]
        num_rois_k = (roi_k[1] - roi_k[0]) // roi_k[2]

        #- Retrieve data
        data = results[f'results{suffix}']

        #- Extract energy and k values from the scansx
        e_values = scansx[2].coords['eV'].values
        k_values = scansx[2].coords['k_par'].values

        cluster_labels_complete = np.zeros(num_rois_k * num_rois_e)

        #- Insert the non-exceeding ROIs into the results of k-means on filtered intensity data by assigning the non-exceeding ROIs to one cluster
        #-- for kmeans_norm
        complete_kmeans_filtered_norm = np.full(num_rois_k * num_rois_e, n_clusters, dtype=data['kmeans_norm'].labels_.dtype)
        complete_kmeans_filtered_norm[data['clustered_curves_indices']] = data['kmeans_norm'].labels_   
        data['kmeans_norm'].labels_  = complete_kmeans_filtered_norm
        #-- for kmeans_all
        complete_kmeans_filtered_all = np.full(num_rois_k * num_rois_e, n_clusters, dtype=data['kmeans_all'].labels_.dtype)
        complete_kmeans_filtered_all[data['clustered_curves_indices']] = data['kmeans_all'].labels_
        data['kmeans_all'].labels_ = complete_kmeans_filtered_all

    
    return results



#--- sort clusters my mean energy (and insert filtered intensities in case of "with_threshold")
def sort_kmeans_clusters_by_mean_energy(kmeans, roi_Eb_values_dict, roi_dict, suffix):
    labels = kmeans.labels_
    n_clusters = kmeans.n_clusters
    roi_Eb_values = roi_Eb_values_dict[suffix]

    cluster_mean_energies = np.zeros(n_clusters)

    # Calculate the mean energy of each cluster
    for cluster_idx in range(n_clusters):
        cluster_indices = np.where(labels == cluster_idx)[0]
        cluster_energies = roi_Eb_values[cluster_indices % len(roi_Eb_values)]
        cluster_mean_energies[cluster_idx] = np.mean(cluster_energies)

    # Get the sorting order of the clusters by their mean energy
    sorted_cluster_indices = np.argsort(cluster_mean_energies)

    # sort labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_cluster_indices)}
    label_mapping[n_clusters] = n_clusters # necessary in case of "with threshold"
    kmeans.labels_ = np.array([label_mapping[label] for label in labels])

    # sort cluster centers
    sorted_cluster_centers = np.empty_like(kmeans.cluster_centers_)
    for new_label, old_label in enumerate(sorted_cluster_indices):
        sorted_cluster_centers[new_label] = kmeans.cluster_centers_[old_label]

    sorted_cluster_mean_energies = np.sort(cluster_mean_energies)
    ##print(f'Cluster mean energies: {cluster_mean_energies}, sorted cluster indices: {sorted_cluster_indices}')

    return kmeans, sorted_cluster_centers, sorted_cluster_mean_energies


#---
def plot_clusters_and_centers_norm(results, custom_cmap, interested_suffix, scansx, roi_dict, time):

    fig, axes = plt.subplots(1, 2, figsize=(3.5*2,3.5))

    ax = axes[0]
    # Retrieve the ROI for energy and k values
    roi_e = roi_dict[f'roi_e{interested_suffix}']
    roi_k = roi_dict['roi_k']
    
    num_rois_e = (roi_e[1] - roi_e[0]) // roi_e[2]
    num_rois_k = (roi_k[1] - roi_k[0]) // roi_k[2]

    # Retrieve data
    data = results[f'results{interested_suffix}']

    # Extract energy and k values from the scansx
    e_values = scansx[2].coords['eV'].values
    k_values = scansx[2].coords['k_par'].values

    
    cluster_labels_complete = results[f'results{interested_suffix}']['kmeans_norm'].labels_
    n_clusters = results[f'results{interested_suffix}']['kmeans_norm'].n_clusters

    # Generate the cluster images using the remapped labels
    cluster_image_all = np.zeros((num_rois_k, num_rois_e))
    k = 0
    roi_y_values = np.zeros_like(cluster_labels_complete, dtype=float)

    for roi_x in range(num_rois_k):
        for roi_y in range(num_rois_e):
            cluster_image_all[roi_x, roi_y] = cluster_labels_complete[k]
            roi_y_values[k] = roi_y  # Store roi_y value corresponding to each label
            k += 1

    # Calculate average and standard deviation of roi_y for each cluster
    average_roi_y = np.zeros(n_clusters)
    std_dev_roi_y = np.zeros(n_clusters)
    average_e = np.zeros(n_clusters)
    std_dev_e = np.zeros(n_clusters)
    
    for i in range(n_clusters):
        indices = (cluster_labels_complete == i)
        average_roi_y[i] = np.mean(roi_y_values[indices])
        average_e[i] = e_values[roi_e[0]] + (e_values[roi_e[1] - 1] - e_values[roi_e[0]]) * (0.5 + np.mean(roi_y_values[indices])) / num_rois_e
        std_dev_roi_y[i] = np.std(roi_y_values[indices])
        std_dev_e[i] = (e_values[roi_e[1] - 1] - e_values[roi_e[0]]) * np.std(roi_y_values[indices]) / num_rois_e
        #print(f"Cluster {i}: Average energy = {average_e[i]:.2f}, Std Dev = {std_dev_e[i]:.2f}")

    # Plotting "All Intensities"
    extent=[k_values[roi_k[0]], k_values[roi_k[1] - 1], e_values[roi_e[1] - 1], e_values[roi_e[0]]]
    im_all = ax.imshow(cluster_image_all.T, cmap=custom_cmap, extent=[k_values[roi_k[0]], k_values[roi_k[1] - 1], e_values[roi_e[0]+num_rois_e*roi_e[2]], e_values[roi_e[0]]], aspect='auto')
    
    #adjust colorbar
    plt.colorbar(im_all, ax=ax, fraction=0.046, pad=0.04, ticks = range(n_clusters))
    ax.invert_yaxis()
    ax.set_xlabel('k (Å-1)')
    ax.set_ylabel("E - EF (eV)")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    ax = axes[1]
    # Retrieve specific results for the given suffix
    specific_results = results[f'results{interested_suffix}']
    
    # Retrieve KMeans model and cluster centers
    kmeans_norm_model = specific_results['kmeans_norm']
    cluster_centers_norm = specific_results['kmeans_norm_cluster_centers']
    labels = kmeans_norm_model.labels_
    norm_intensities = specific_results['normalized_intensities']
    
    # Calculate the standard deviation of data points in each cluster
    cluster_stds = []
    for i in range(n_clusters):
        cluster_points = norm_intensities[labels == i]
        # Calculate std deviation across all points in the cluster along the time axis
        cluster_std = np.std(cluster_points, axis=0)
        cluster_stds.append(cluster_std)

    colors = [custom_cmap.colors[i] for i in range(n_clusters)]

    # Plot cluster centers for normalized intensities
    for i in range(n_clusters):
        ax.plot(time, cluster_centers_norm[i], label=f'{i}', color=colors[i]) #marker ='o'# marker='s', ms=3
        # Optionally add shaded area for variance
        '''ax.fill_between(time, 
                        cluster_centers_norm[i] - cluster_stds[i], 
                        cluster_centers_norm[i] + cluster_stds[i], 
                        color=colors[i], alpha=0.3)'''

    ax.set_xlabel("t - t0 (fs)")
    ax.set_ylabel("Normalized Intensity")
    ax.margins(x=0)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(500))
    ax.set_xlim(-500,3000)

    ax.legend(title='cluster #:', frameon=False)
    fig.tight_layout()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
   # plt.savefig(f"plot_{current_time}.svg", format="svg", dpi=300, transparent=True)
   # plt.close()
   # files.download(f"plot_{current_time}.svg")


#--- Save Eb values
def save_roi_Eb_values(roi_dict, suffixes, Eb_values):
    roiEb_values_dict = {}
    for suffix in suffixes:
        # truncate the energy values according to the interested suffix
        roi_e = roi_dict[f'roi_e{suffix}']
        num_rois_e = (roi_e[1] - roi_e[0]) // roi_e[2]
        Eb_values_trunc = Eb_values[roi_e[0]:roi_e[1]]
        roi_Eb_values = np.zeros(num_rois_e)

        # average energy values in each roi
        for roi_y in range(num_rois_e):
            roi_Eb_values[roi_y] = np.mean(Eb_values_trunc[roi_y * roi_e[2]:(roi_y+1) * roi_e[2]])
        roiEb_values_dict[suffix] = roi_Eb_values
    return roiEb_values_dict

#--- Save k values
def save_roi_k_values(roi_dict, k_values):
    # truncate the k values according to the interested k range
    roi_k = roi_dict[f'roi_k']
    num_rois_k = (roi_k[1] - roi_k[0]) // roi_k[2]
    k_values_trunc = k_values[roi_k[0]:roi_k[1]]
    roi_k_values = np.zeros(num_rois_k)

    # average energy values in each roi
    for roi_x in range(num_rois_k):
        roi_k_values[roi_x] = np.mean(k_values_trunc[roi_x * roi_k[2]:(roi_x+1) * roi_k[2]])
    return roi_k_values

