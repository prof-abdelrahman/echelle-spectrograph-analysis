import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

class Task_g_develop_wavelength_solution:
    @staticmethod
    def calibration_frame_image(calib_frame):
        plt.figure(figsize=(12, 6))
        plt.imshow(calib_frame, cmap='viridis', vmin=np.percentile(calib_frame, 5), vmax=np.percentile(calib_frame, 95))
        plt.title('Calibration Frame')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.colorbar(label='Intensity')
        plt.savefig(os.path.join("results","calibration_frame_image.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def find_calibration_frame_peaks(calib_frame, order_centers, mean_dark):
        # # Remove bias from calibration frame (Not needed as per professor's guidance)
        # calib_corrected = calib_frame - mean_dark[:calib_frame.shape[0], :calib_frame.shape[1]]
        
        # As per professor's guidance, no need to correct for bias
        calib_corrected = calib_frame  # Using directly without bias correction
        
        all_peaks = []
        
        # For each order, find the spectral lines
        for i, center in enumerate(order_centers):
            if center >= calib_corrected.shape[0]:
                continue
                
            # Extract the order profile
            half_width = 5  # Adjust based on order width
            order_min = max(0, center - half_width)
            order_max = min(calib_corrected.shape[0], center + half_width + 1)
            
            # Take maximum along the order width to enhance signal
            order_profile = np.max(calib_corrected[order_min:order_max, :], axis=0)
            
            # Find peaks in this order
            prominence = np.max(order_profile) * 0.1  # Adjust as needed
            x_positions, properties = signal.find_peaks(order_profile, prominence=prominence, distance=5)
            
            # For each peak, fit a Gaussian for more precise position
            def gaussian(x, amplitude, center, sigma):
                return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
            
            peak_fits = []
            
            for peak_x in x_positions:
                # Extract region around peak
                window = 5  # Points on each side
                start_idx = max(0, peak_x - window)
                end_idx = min(len(order_profile), peak_x + window + 1)
                
                if end_idx - start_idx < 5:  # Need at least 5 points for a good fit
                    continue
                    
                x_data = np.arange(start_idx, end_idx)
                y_data = order_profile[start_idx:end_idx]
                
                try:
                    # Initial guess parameters
                    p0 = [order_profile[peak_x], peak_x, 2.0]
                    
                    # Fit Gaussian
                    popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0)
                    
                    # Store order number, x position, and fit parameters
                    peak_fits.append((i, center, popt[1], popt[0], popt[2]))  # order_idx, y_position, x_position, amplitude, sigma
                except:
                    # If fit fails, use the original peak position
                    peak_fits.append((i, center, peak_x, order_profile[peak_x], np.nan))
            
            all_peaks.extend(peak_fits)
        
        # Convert to numpy array
        all_peaks = np.array(all_peaks)
        
        # Visualize some of the detected peaks on the calibration frame
        plt.figure(figsize=(12, 8))
        plt.imshow(calib_corrected, cmap='viridis', aspect='auto', 
                vmin=np.percentile(calib_corrected, 10), 
                vmax=np.percentile(calib_corrected, 99))
        
        # Plot peak positions (just a subset for clarity)
        subset = np.random.choice(len(all_peaks), min(100, len(all_peaks)), replace=False)
        plt.scatter(all_peaks[subset, 2], all_peaks[subset, 1], color='r', s=10, alpha=0.7)
        
        plt.title('Calibration Frame with Detected Peaks')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.colorbar(label='Intensity')
        plt.savefig(os.path.join("results", "calibration_frame_peaks.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return all_peaks

    @staticmethod
    def develop_wavelength_solution(calib_lamp_peaks, frame_peaks):
        # I need to associate the known wavelengths from the calibration lamp
        # with the detected peak positions on the detector
        
        # For each order, I'll create a polynomial fit that maps
        # detector x-position to wavelength
        
        # Group peaks by order
        orders = np.unique(frame_peaks[:, 0]).astype(int)
        wavelength_solutions = []
        
        for order_idx in orders:
            # Get peaks for this order
            order_peaks = frame_peaks[frame_peaks[:, 0] == order_idx]
            
            # For demonstration, I'll use a simple approach:
            # Assume the peaks are in the same order as in the calibration lamp
            # (This is not generally true and would need a more sophisticated matching algorithm)
            
            # Number of peaks to use (minimum of what we have in both datasets)
            n_peaks = min(len(order_peaks), len(calib_lamp_peaks))
            
            if n_peaks < 5:  # Need at least 5 points for a reasonable polynomial fit
                continue
                
            # Get x positions on detector and corresponding wavelengths
            detector_positions = order_peaks[:n_peaks, 2]
            wavelengths = calib_lamp_peaks[:n_peaks, 0]  # Using wavelength centers from fits
            
            # Fit a polynomial (3rd order for demonstration)
            coeffs = np.polyfit(detector_positions, wavelengths, 3)
            
            # Calculate residuals
            fit_wavelengths = np.polyval(coeffs, detector_positions)
            residuals = wavelengths - fit_wavelengths
            rms_error = np.sqrt(np.mean(residuals**2))
            
            # Store solution for this order
            wavelength_solutions.append((order_idx, coeffs, rms_error))
            
            # Plot the fit and residuals
            if len(wavelength_solutions) <= 5:  # Only plot first 5 orders for clarity
                fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Plot data and fit
                x_dense = np.linspace(np.min(detector_positions), np.max(detector_positions), 100)
                y_fit = np.polyval(coeffs, x_dense)
                
                axes[0].scatter(detector_positions, wavelengths, c='b', label='Peaks')
                axes[0].plot(x_dense, y_fit, 'r-', label='Polynomial Fit')
                axes[0].set_ylabel('Wavelength (nm)')
                axes[0].set_title(f'Order {order_idx+1} Wavelength Solution')
                axes[0].legend()
                
                # Plot residuals
                axes[1].scatter(detector_positions, residuals)
                axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.5)
                axes[1].set_xlabel('Detector X Position')
                axes[1].set_ylabel('Residuals (nm)')
                axes[1].set_title(f'RMS Error: {rms_error:.3f} nm')
                
                plt.tight_layout()
                plt.savefig(os.path.join("results", f'wavelength_solution_order_{order_idx+1}.png'), dpi=300, bbox_inches='tight')
                plt.show()
        
        # Summarize the wavelength solutions
        print("\nWavelength Solution Summary:")
        for order_idx, coeffs, rms_error in wavelength_solutions:
            print(f"Order {order_idx+1}: RMS Error = {rms_error:.4f} nm")
        
        return wavelength_solutions
    
    @staticmethod
    def plot_all_wavelength_solutions(wavelength_solutions, frame_peaks):
        """Plot wavelength solutions for all orders and create a comprehensive visualization"""
        
        # Create a unified figure for all wavelength solutions
        plt.figure(figsize=(12, 8))
        
        # Track the overall wavelength range covered
        min_wavelength = float('inf')
        max_wavelength = float('-inf')
        
        # Plot each order's solution with a different color
        colors = plt.cm.viridis(np.linspace(0, 1, len(wavelength_solutions)))
        
        # Table to store the wavelength ranges
        wavelength_ranges = []
        
        # Plot each solution
        for i, (order_idx, coeffs, rms_error) in enumerate(wavelength_solutions):
            # Get peaks for this order
            order_peaks = frame_peaks[frame_peaks[:, 0] == order_idx]
            
            if len(order_peaks) == 0:
                continue
                
            # Get x positions on detector
            detector_positions = order_peaks[:, 2]
            
            # Create dense x values for smooth curve
            x_dense = np.linspace(np.min(detector_positions), np.max(detector_positions), 200)
            
            # Calculate wavelengths from polynomial coefficients
            wavelengths = np.polyval(coeffs, x_dense)
            
            # Update global min/max wavelength
            min_wavelength = min(min_wavelength, np.min(wavelengths))
            max_wavelength = max(max_wavelength, np.max(wavelengths))
            
            # Store the wavelength range for this order
            wavelength_ranges.append((order_idx, np.min(wavelengths), np.max(wavelengths), rms_error))
            
            # Plot the wavelength solution
            plt.plot(x_dense, wavelengths, '-', color=colors[i], 
                    label=f'Order {order_idx+1} (RMS: {rms_error:.3f} nm)')
        
        plt.title('Wavelength Solutions for All Orders')
        plt.xlabel('Detector X Position (pixel)')
        plt.ylabel('Wavelength (nm)')
        plt.grid(True, alpha=0.3)
        
        # Create a custom legend with fewer entries if there are many orders
        if len(wavelength_solutions) > 15:
            # Select a subset of orders for the legend
            step = len(wavelength_solutions) // 10 + 1
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[i] for i in range(0, len(handles), step)],
                    [labels[i] for i in range(0, len(labels), step)],
                    loc='best', fontsize='small')
        else:
            plt.legend(loc='best', fontsize='small')
        
        plt.tight_layout()
        plt.savefig(os.path.join("results", "all_wavelength_solutions.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create a full spectral coverage visualization
        plt.figure(figsize=(14, 6))
        
        # Sort by order
        wavelength_ranges.sort(key=lambda x: x[0])
        
        # Plot each order's coverage as a horizontal line
        for i, (order_idx, min_wl, max_wl, rms_error) in enumerate(wavelength_ranges):
            plt.plot([min_wl, max_wl], [i, i], '-', linewidth=3, 
                    color=plt.cm.viridis(i/len(wavelength_ranges)))
            plt.text(min_wl - 2, i, f"{order_idx+1}", va='center', ha='right', fontsize=8)
            plt.text(max_wl + 2, i, f"{min_wl:.1f}-{max_wl:.1f} nm", va='center', ha='left', fontsize=8)
        
        plt.title('Spectral Coverage of All Orders')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Order Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join("results", "spectral_coverage.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print table of wavelength ranges
        print("\nWavelength Solution Summary for All Orders:")
        print("============================================")
        print(f"{'Order':>5} | {'Min λ (nm)':>10} | {'Max λ (nm)':>10} | {'Range (nm)':>10} | {'RMS Error (nm)':>15}")
        print("-" * 60)
        
        for order_idx, min_wl, max_wl, rms_error in wavelength_ranges:
            print(f"{order_idx+1:5d} | {min_wl:10.2f} | {max_wl:10.2f} | {max_wl-min_wl:10.2f} | {rms_error:15.4f}")
        
        print("\nTotal Spectral Coverage:")
        print(f"Minimum wavelength: {min_wavelength:.2f} nm")
        print(f"Maximum wavelength: {max_wavelength:.2f} nm")
        print(f"Total range: {max_wavelength - min_wavelength:.2f} nm")
        
        # Calculate any gaps in coverage
        gaps = []
        for i in range(1, len(wavelength_ranges)):
            prev_max = wavelength_ranges[i-1][2]  # Max wavelength of previous order
            curr_min = wavelength_ranges[i][1]    # Min wavelength of current order
            
            if curr_min > prev_max + 0.5:  # Gap threshold of 0.5 nm
                gaps.append((wavelength_ranges[i-1][0], wavelength_ranges[i][0], prev_max, curr_min))
        
        if gaps:
            print("\nGaps in spectral coverage:")
            for prev_order, next_order, prev_max, next_min in gaps:
                print(f"Gap between orders {prev_order} and {next_order}: {prev_max:.2f} - {next_min:.2f} nm ({next_min - prev_max:.2f} nm)")
        else:
            print("\nNo significant gaps in spectral coverage detected.")
        
        return wavelength_ranges