import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Used to create smooth curves between data points 
from scipy import interpolate
#Used here to calculate chi-squared and p-values for statistical analysis.
from scipy import stats


def plot_filter_profiles(csv_file):

    print("Script started")


    df_raw = pd.read_csv(csv_file, header=None)
 
  

    print("First 5 rows of raw data:")
    print(df_raw.head())
    



    data_start_row = 2

    experimental_x = df_raw.iloc[data_start_row:, 0].values.astype(float)
    experimental_y = df_raw.iloc[data_start_row:, 7].values.astype(float)
    experimental_yerr = df_raw.iloc[data_start_row:, 8].values.astype(float)

    original_x = df_raw.iloc[data_start_row:, 11].values.astype(float)
    original_y = df_raw.iloc[data_start_row:, 12].values.astype(float)


    print(f"Experimental data shapes - X: {experimental_x.shape}, Y:{experimental_y.shape}")
    print(f"Original data shapes - X: {original_x.shape}, Y: {original_y.shape}")

    exp_valid_mask = ~np.isnan(experimental_x) & ~np.isnan(experimental_y) & ~np.isnan(experimental_yerr)
    orig_valid_mask = ~np.isnan(original_x) & ~np.isnan(original_y)

    experimental_x = experimental_x[exp_valid_mask]
    experimental_y = experimental_y[exp_valid_mask]
    experimental_yerr = experimental_yerr[exp_valid_mask]

    original_x = original_x[orig_valid_mask]
    original_y = original_y[orig_valid_mask]

    print(f"After cleaning - Experimental X: {experimental_x.shape}, Original X: {original_x.shape}")

    interp_func = interpolate.interp1d(
        original_x, 
        original_y, 
        kind='cubic',  # Use cubic spline interpolation for smooth curves
        bounds_error=False,  # Don't raise error for points outside the original x range
        fill_value=0.0  # Fill with zeros for points outside the range
    )
    
    interpolated_original_y = interp_func(experimental_x)
    min_y = np.min(experimental_y)
    max_y = np.max(experimental_y)

    print(f"Min experimental Y: {min_y}, Max experimental Y: {max_y}")

    normalized_y = (experimental_y - min_y) / (max_y - min_y)

    best_multiplier = 1.0
    best_chi2 = float('inf')
    chi2_results = []
    
    multipliers = np.linspace(0.8, 1.2, 41)  # Test from 0.8 to 1.2 in steps of 0.01
    
    for mult in multipliers:
        # Apply the multiplier
        test_y = normalized_y * mult

    # Calculate chi-squared between adjusted experimental and interpolated original
        # For points with valid error measurements
        valid_err_mask = experimental_yerr > 0
        if np.sum(valid_err_mask) > 0:
            # Scale errors appropriately 
            scaled_err = experimental_yerr[valid_err_mask] * (mult/(max_y-min_y))
            
            # Calculate chi-squared
            chi2 = np.sum(
                ((test_y[valid_err_mask] - interpolated_original_y[valid_err_mask]) ** 2) / 
                (scaled_err ** 2)
            )
            
            # Normalize by degrees of freedom (N - 1 parameters)
            chi2_reduced = chi2 / (np.sum(valid_err_mask) - 1)
            
            chi2_results.append((mult, chi2_reduced))
            
            # Save the best multiplier
            if chi2_reduced < best_chi2:
                best_chi2 = chi2_reduced
                best_multiplier = mult
    
    print(f"Best multiplier: {best_multiplier}, Chi-squared: {best_chi2}")
    
    # Apply the best multiplier
    adjusted_experimental_y = normalized_y * best_multiplier
    
    # Calculate the p-value for the chi-squared test
    # Degrees of freedom = number of data points with valid errors - 1 parameter (multiplier)
    degrees_of_freedom = np.sum(experimental_yerr > 0) - 1
    p_value = 1 - stats.chi2.cdf(best_chi2 * degrees_of_freedom, degrees_of_freedom)
    
    print(f"Chi-squared p-value: {p_value}")
    
    # Create the figure with 3 rows, 2 columns
    fig = plt.figure(figsize=(14, 15))
    
    # Plot the chi-squared results
    plt.subplot(3, 2, 1)
    plt.plot(
        [m for m, _ in chi2_results], 
        [c for _, c in chi2_results], 
        'g-', 
        linewidth=1.5
    )
    plt.axvline(x=best_multiplier, color='r', linestyle='--', label=f'Best multiplier: {best_multiplier:.3f}')
    plt.xlabel('Multiplier', fontsize=12)
    plt.ylabel('Reduced Chi-squared', fontsize=12)
    plt.title('Chi-squared vs. Multiplier', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # EXP PROFILE - ADJUSTED
    plt.subplot(3, 2, 2)
    plt.errorbar(
        experimental_x, 
        adjusted_experimental_y, 
        yerr=experimental_yerr * (best_multiplier/(max_y-min_y)), 
        fmt='bo-', 
        linewidth=1.5, 
        markersize=4, 
        capsize=3, 
        label='Adjusted Experimental Profile'
    )
    plt.plot(
        experimental_x, 
        interpolated_original_y, 
        'r--', 
        linewidth=1.5, 
        label='Interpolated Original Profile'
    )
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Transmission', fontsize=12)
    plt.title(f'Optimized Experimental Filter Profile\nChi² = {best_chi2:.3f}, p = {p_value:.3e}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # O.G PROFILE
    plt.subplot(3, 2, 3)
    plt.plot(original_x, original_y, 'ro-', linewidth=1.5, markersize=2, label='Original Filter Profile')
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Transmission', fontsize=12)
    plt.title('Original Filter Profile', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Residuals plot (difference between adjusted experimental and interpolated original)
    plt.subplot(3, 2, 4)
    residuals = adjusted_experimental_y - interpolated_original_y
    
    plt.errorbar(
        experimental_x, 
        residuals,
        yerr=experimental_yerr * (best_multiplier/(max_y-min_y)),
        fmt='go', 
        linewidth=1.5, 
        markersize=4, 
        capsize=3,
        label='Residuals'
    )
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Line')
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Residuals (Adjusted - Interpolated)', fontsize=12)
    plt.title('Residuals Plot', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # BOTH PROFILE COMBINATION - Now spans across bottom of figure
    plt.subplot(3, 1, 3)
    plt.errorbar(
        experimental_x, 
        adjusted_experimental_y, 
        yerr=experimental_yerr * (best_multiplier/(max_y-min_y)), 
        fmt='bo-', 
        linewidth=1.5, 
        markersize=4, 
        capsize=3, 
        label='Adjusted Experimental Profile'
    )
    plt.plot(original_x, original_y, 'ro-', linewidth=1.5, markersize=2, label='Original Filter Profile')
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Transmission', fontsize=12)
    plt.title('Combined Filter Profiles (Optimized Fit)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Common x-range for comparison plot
    min_wavelength = min(min(experimental_x), min(original_x))
    max_wavelength = max(max(experimental_x), max(original_x))
    plt.xlim(min_wavelength, max_wavelength)
    
    plt.tight_layout()
    plt.savefig('filter_profiles_with_chi2.png', dpi=300)
    
    plt.show()

#if __name__ == "SCALES Code":
    # Path 
    #csv_file = "data.csv"
    
    #plot_filter_profiles(csv_file)
    if __name__ == "__main__":
        # Ensure this matches your actual filename in the folder
        csv_file = "data.csv" 

        import os
        if os.path.exists(csv_file):
            plot_filter_profiles(csv_file)
        else:
            print(f"Error: {csv_file} not found. Check the file name and path.")

    




# problem is that there is no plot i can see. 