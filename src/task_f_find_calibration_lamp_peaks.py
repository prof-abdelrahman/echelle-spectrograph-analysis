import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
import os

class Task_f_find_calibration_lamp_peaks:
    @staticmethod
    def find_calibration_lamp_peaks(calib_spectrum, calib_wavelength):
        """
        Find peaks in the calibration lamp spectrum and fit Gaussians to them.
        
        Parameters:
        - calib_spectrum: 1D array with the intensity values of the calibration lamp spectrum.
        - calib_wavelength: 1D array with the corresponding wavelength values.
        
        Returns:
        - peak_fits: 2D array with the fitted peak parameters (center, amplitude, sigma) for each peak.
        """
        # Find peaks in calibration lamp spectrum
        prominence = np.max(calib_spectrum) * 0.05  # Adjust prominence threshold as needed
        peaks, properties = signal.find_peaks(calib_spectrum, prominence=prominence, width=3)
        
        # Plot the spectrum with identified peaks
        plt.figure(figsize=(14, 6))
        plt.plot(calib_wavelength, calib_spectrum, 'b-', label='Calibration Spectrum')
        plt.plot(calib_wavelength[peaks], calib_spectrum[peaks], 'ro', label='Identified Peaks')
        
        # Commented for now as they are not showing up properly
        # # Get current y-axis limits
        # ymin, ymax = plt.ylim()
        # # Set new limits with more space at top
        # plt.ylim(ymin, ymax*1.2)  # Add 20% more space at top
        
        # # Add peak annotations for some peaks (skip some for clarity)
        # for i, peak in enumerate(peaks):
        #     if i % 5 == 0:  # Only label every 5th peak for clarity
        #         plt.text(calib_wavelength[peak], calib_spectrum[peak]*1.1, 
        #                 f"{calib_wavelength[peak]:.1f}", ha='center', fontsize=8)
        
        plt.title('Calibration Lamp Spectrum with Identified Peaks')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.grid(True, alpha=0.3)
        plt.legend()  # Add legend for blue and red colors
        plt.tight_layout()  # Ensure proper spacing
        plt.savefig(os.path.join("results","calibration_lamp_peaks.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Fit Gaussian to each peak for more precise position
        peak_fits = []
        
        def gaussian(x, amplitude, center, sigma):
            return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
        
        for peak in peaks:
            # Extract region around peak
            window = 10  # Points on each side
            start_idx = max(0, peak - window)
            end_idx = min(len(calib_wavelength), peak + window + 1)
            
            x_data = calib_wavelength[start_idx:end_idx]
            y_data = calib_spectrum[start_idx:end_idx]
            
            try:
                # Initial guess parameters
                p0 = [calib_spectrum[peak], calib_wavelength[peak], 1.0]
                
                # Fit Gaussian
                popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0)
                peak_fits.append((popt[1], popt[0], popt[2]))  # center, amplitude, sigma
            except:
                # If fit fails, use the original peak position
                peak_fits.append((calib_wavelength[peak], calib_spectrum[peak], np.nan))
        
        # Convert to numpy array for easier handling
        peak_fits = np.array(peak_fits)
        
        # Print statistics about the peaks
        print(f"Number of peaks found: {len(peak_fits)}")
        print(f"Average peak width (sigma): {np.nanmean(peak_fits[:, 2]):.3f} nm")
        
        # Print the wavelengths of all identified peaks in a tabular format
        print("\nIdentified peak wavelengths (nm):")
        print("---------------------------------")
        for i, (center, amplitude, sigma) in enumerate(peak_fits):
            print(f"Peak {i+1:3d}: {center:.2f} nm (amplitude: {amplitude:.0f}, width: {sigma:.3f} nm)")
                
        return peak_fits