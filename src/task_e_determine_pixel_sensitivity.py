import numpy as np
import matplotlib.pyplot as plt
import os
# (flat, white_spectrum, white_wavelength, order_centers, mean_dark) to remember

class Task_e_determine_pixel_sensitivity:
    @staticmethod
    # Pixel sensitivity without wavelength solution
    def determine_pixel_sensitivity(flat, white_spectrum, white_wavelength, order_centers, mean_dark):
        # Remove bias from flat frame
        flat_corrected = flat - mean_dark
        
        # Create a sensitivity map initialized with zeros
        sensitivity_map = np.zeros_like(flat_corrected)
        
        # For each identified order, estimate its wavelength range and map sensitivity
        for i, center in enumerate(order_centers):
            # Extract range around the order center
            half_width = 20  # Can be adjust based on order width
            order_min = max(0, center - half_width)
            order_max = min(flat_corrected.shape[0], center + half_width + 1)
            order_region = flat_corrected[order_min:order_max, :]
            
            # For now, I'll use a simplified model:
            # Assume linear relationship between detector position and wavelength
            detector_positions = np.arange(flat_corrected.shape[1])
            
            # Interpolate white lamp spectrum to our detector coordinates
            interp_spectrum = np.interp(
                detector_positions/flat_corrected.shape[1]*(white_wavelength.max()-white_wavelength.min())+white_wavelength.min(),
                white_wavelength,
                white_spectrum
            )
            
            # Process each row in the order region
            for j in range(order_min, order_max):
                row_data = flat_corrected[j, :]
                
                # Calculate sensitivity for this row
                # Avoid division by zero by adding small value
                row_sensitivity = row_data / (interp_spectrum + 1e-10)
                
                # Apply Gaussian weighting based on distance from order center
                weight = np.exp(-((j - center)**2) / (2*(half_width/2)**2))
                
                # Store in sensitivity map
                sensitivity_map[j, :] = row_sensitivity * weight
        
        # Normalize the entire sensitivity map
        if np.max(sensitivity_map) > 0:
            sensitivity_map = sensitivity_map / np.max(sensitivity_map)
        
        # Calculate sensitivity statistics
        # Overall statistics
        sensitivity_mean = np.mean(sensitivity_map[sensitivity_map > 0.01])
        sensitivity_median = np.median(sensitivity_map[sensitivity_map > 0.01])
        sensitivity_std = np.std(sensitivity_map[sensitivity_map > 0.01])
        
        print("\nSensitivity Statistics (Overall):")
        print(f"Mean sensitivity: {sensitivity_mean:.4f}")
        print(f"Median sensitivity: {sensitivity_median:.4f}")
        print(f"Standard deviation: {sensitivity_std:.4f}")
        print(f"Maximum sensitivity: 1.0000 (normalized)")
        print(f"Minimum non-zero sensitivity: {np.min(sensitivity_map[sensitivity_map > 0]):.4f}")
        
        # Per-order statistics
        print("\nSensitivity Statistics (Per Order):")
        order_stats = []
        for i, center in enumerate(order_centers):
            half_width = 5  # Narrower range for statistics
            order_min = max(0, center - half_width)
            order_max = min(sensitivity_map.shape[0], center + half_width + 1)
            order_sensitivity = sensitivity_map[order_min:order_max, :]
            
            # Get statistics for this order
            order_mean = np.mean(order_sensitivity[order_sensitivity > 0.01])
            order_max = np.max(order_sensitivity)
            order_peak_col = np.unravel_index(np.argmax(order_sensitivity), order_sensitivity.shape)[1]
            
            # Calculate wavelength variation along the order
            left_third = np.mean(order_sensitivity[:, :order_sensitivity.shape[1]//3][order_sensitivity[:, :order_sensitivity.shape[1]//3] > 0.01])
            middle_third = np.mean(order_sensitivity[:, order_sensitivity.shape[1]//3:2*order_sensitivity.shape[1]//3][order_sensitivity[:, order_sensitivity.shape[1]//3:2*order_sensitivity.shape[1]//3] > 0.01])
            right_third = np.mean(order_sensitivity[:, 2*order_sensitivity.shape[1]//3:][order_sensitivity[:, 2*order_sensitivity.shape[1]//3:] > 0.01])
            
            wavelength_variation = max(left_third, middle_third, right_third) / min(left_third, middle_third, right_third) if min(left_third, middle_third, right_third) > 0 else 1.0
            
            order_stats.append((i+1, order_mean, order_max, order_peak_col, wavelength_variation))
            
            print(f"Order {i+1} (row {center}): Mean={order_mean:.4f}, Max={order_max:.4f}, Peak Column={order_peak_col}, Wavelength Variation Factor={wavelength_variation:.2f}")
        
        # Find orders with highest and lowest sensitivity
        most_sensitive_order = max(order_stats, key=lambda x: x[2])[0]
        least_sensitive_order = min(order_stats, key=lambda x: x[2])[0]
        
        print(f"\nMost sensitive order: Order {most_sensitive_order}")
        print(f"Least sensitive order: Order {least_sensitive_order}")
        
        # Calculate average wavelength sensitivity variation
        avg_wavelength_variation = np.mean([x[4] for x in order_stats])
        print(f"Average wavelength variation factor across orders: {avg_wavelength_variation:.2f}")
        
        # Visualize the sensitivity map
        plt.figure(figsize=(12, 8))
        plt.imshow(sensitivity_map, cmap='viridis', aspect='auto')
        plt.colorbar(label='Relative Sensitivity')
        plt.title('Detector Pixel Sensitivity Map')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig(os.path.join("results","sensitivity_map.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot sensitivity profiles for selected orders
        plt.figure(figsize=(12, 6))
        for i in range(0, len(order_centers), len(order_centers)//5):  # Plot ~5 orders
            center = order_centers[i]
            profile = sensitivity_map[center, :]
            plt.plot(profile, label=f'Order {i+1} (row {center})')
        
        plt.title('Sensitivity Profiles Along Selected Orders')
        plt.xlabel('Pixels')
        plt.ylabel('Relative Sensitivity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join("results","sensitivity_profiles.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return sensitivity_map
    
    @staticmethod
    def determine_pixel_sensitivity_with_wavelength(flat, order_centers, wavelength_solutions, white_wavelength, white_spectrum):
        """
        Determine pixel sensitivity using the wavelength solution.
        """
        # Print debugging information
        print(f"Number of wavelength solutions: {len(wavelength_solutions)}")
        print(f"Number of order centers: {len(order_centers)}")
        
        # Initialize sensitivity map
        sensitivity_map = np.zeros_like(flat)
        
        # Create an interpolation function for the white lamp spectrum
        from scipy.interpolate import interp1d
        white_interp = interp1d(white_wavelength, white_spectrum, 
                                bounds_error=False, fill_value="extrapolate")
        
        # For each order with a wavelength solution
        for i, (order_idx, coeffs, _) in enumerate(wavelength_solutions):
            # I only have wavelength solutions for the first 10 orders
            if i < len(order_centers):
                center = order_centers[i]
                
                # Define the width around the order to process
                half_width = 10  # Hint: adjust based on order width
                
                # Process each row in this order
                for y in range(max(0, center - half_width), min(flat.shape[0], center + half_width + 1)):
                    # Weighting factor based on distance from center (Gaussian)
                    weight = np.exp(-(y - center)**2 / (2 * (half_width/2)**2))
                    
                    # For each pixel in this row
                    for x in range(flat.shape[1]):
                        # Calculate wavelength for this pixel using the wavelength solution
                        wavelength = np.polyval(coeffs, x)
                        
                        # Get expected intensity from white lamp at this wavelength
                        expected_intensity = white_interp(wavelength)
                        
                        # Calculate sensitivity (avoid division by zero)
                        if expected_intensity > 0:
                            # Sensitivity = measured / expected
                            sensitivity = flat[y, x] / expected_intensity
                            sensitivity_map[y, x] = sensitivity * weight
        
        # Normalize the sensitivity map
        if np.max(sensitivity_map) > 0:
            sensitivity_map = sensitivity_map / np.max(sensitivity_map)
        
        # Calculate sensitivity statistics - modify this part to only consider orders with solutions
        # Overall statistics
        sensitivity_mean = np.mean(sensitivity_map[sensitivity_map > 0.01])
        sensitivity_median = np.median(sensitivity_map[sensitivity_map > 0.01])
        sensitivity_std = np.std(sensitivity_map[sensitivity_map > 0.01])
        
        print("\nSensitivity Statistics (Overall):")
        print(f"Mean sensitivity: {sensitivity_mean:.4f}")
        print(f"Median sensitivity: {sensitivity_median:.4f}")
        print(f"Standard deviation: {sensitivity_std:.4f}")
        print(f"Maximum sensitivity: 1.0000 (normalized)")
        print(f"Minimum non-zero sensitivity: {np.min(sensitivity_map[sensitivity_map > 0]):.4f}")
        
        # Per-order statistics
        print("\nSensitivity Statistics (Per Order):")
        order_stats = []
        
        # Only analyze orders that have wavelength solutions (first 10)
        for i, center in enumerate(order_centers):
            half_width = 5  # Narrower range for statistics
            order_min = max(0, center - half_width)
            order_max = min(sensitivity_map.shape[0], center + half_width + 1)
            order_sensitivity = sensitivity_map[order_min:order_max, :]
            
            # Check if this order has sensitivity values (i.e., has a wavelength solution)
            if i < len(wavelength_solutions) and np.max(order_sensitivity) > 0:
                # Get statistics for this order
                order_mean = np.mean(order_sensitivity[order_sensitivity > 0.01])
                order_max = np.max(order_sensitivity)
                if order_max > 0:
                    order_peak_col = np.unravel_index(np.argmax(order_sensitivity), order_sensitivity.shape)[1]
                else:
                    order_peak_col = 0
                
                # Calculate wavelength variation along the order
                left_third = np.mean(order_sensitivity[:, :order_sensitivity.shape[1]//3][order_sensitivity[:, :order_sensitivity.shape[1]//3] > 0.01]) if np.any(order_sensitivity[:, :order_sensitivity.shape[1]//3] > 0.01) else 0
                middle_third = np.mean(order_sensitivity[:, order_sensitivity.shape[1]//3:2*order_sensitivity.shape[1]//3][order_sensitivity[:, order_sensitivity.shape[1]//3:2*order_sensitivity.shape[1]//3] > 0.01]) if np.any(order_sensitivity[:, order_sensitivity.shape[1]//3:2*order_sensitivity.shape[1]//3] > 0.01) else 0
                right_third = np.mean(order_sensitivity[:, 2*order_sensitivity.shape[1]//3:][order_sensitivity[:, 2*order_sensitivity.shape[1]//3:] > 0.01]) if np.any(order_sensitivity[:, 2*order_sensitivity.shape[1]//3:] > 0.01) else 0
                
                if min(left_third, middle_third, right_third) > 0:
                    wavelength_variation = max(left_third, middle_third, right_third) / min(left_third, middle_third, right_third)
                else:
                    wavelength_variation = 1.0
            else:
                # No wavelength solution for this order
                order_mean = float('nan')
                order_max = 0.0
                order_peak_col = 0
                wavelength_variation = 1.0
            
            order_stats.append((i+1, order_mean, order_max, order_peak_col, wavelength_variation))
            
            print(f"Order {i+1} (row {center}): Mean={order_mean:.4f}, Max={order_max:.4f}, Peak Column={order_peak_col}, Wavelength Variation Factor={wavelength_variation:.2f}")
        
        # Find orders with highest and lowest sensitivity (only consider those with data)
        valid_order_stats = [os for os in order_stats if not np.isnan(os[1]) and os[2] > 0]
        
        if valid_order_stats:
            most_sensitive_order = max(valid_order_stats, key=lambda x: x[2])[0]
            least_sensitive_order = min(valid_order_stats, key=lambda x: x[2])[0]
            
            print(f"\nMost sensitive order: Order {most_sensitive_order}")
            print(f"Least sensitive order: Order {least_sensitive_order}")
            
            # Calculate average wavelength sensitivity variation
            avg_wavelength_variation = np.mean([x[4] for x in valid_order_stats])
            print(f"Average wavelength variation factor across orders: {avg_wavelength_variation:.2f}")
        else:
            print("\nNo valid sensitivity data found for any order.")
        
        # Visualize the sensitivity map
        plt.figure(figsize=(12, 8))
        plt.imshow(sensitivity_map, cmap='viridis', aspect='auto')
        plt.colorbar(label='Relative Sensitivity')
        plt.title('Detector Pixel Sensitivity Map (Based on Wavelength Solution)')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig(os.path.join("results","sensitivity_map_with_wavelength.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot the sensitivity as a function of wavelength
        wavelengths = []
        sensitivities = []
        
        # Sample points from the sensitivity map (only from orders with wavelength solutions)
        for i, (order_idx, coeffs, _) in enumerate(wavelength_solutions):
            if i < len(order_centers):
                center = order_centers[i]
                
                # Sample points along this order
                x_samples = np.linspace(0, flat.shape[1]-1, 50).astype(int)
                
                for x in x_samples:
                    wavelength = np.polyval(coeffs, x)
                    sensitivity = sensitivity_map[center, x]
                    
                    if sensitivity > 0:
                        wavelengths.append(wavelength)
                        sensitivities.append(sensitivity)
        
        # Plot sensitivity profiles for selected orders
        plt.figure(figsize=(12, 6))
        for i in range(0, len(order_centers), len(order_centers)//5):  # Plot ~5 orders
            center = order_centers[i]
            profile = sensitivity_map[center, :]
            plt.plot(profile, label=f'Order {i+1} (row {center})')
        plt.title('Sensitivity Profiles Along Selected Orders (Wavelength Solution)')
        plt.xlabel('Pixels')
        plt.ylabel('Relative Sensitivity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join("results","sensitivity_profiles_wavelength_solution.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot sensitivity vs wavelength
        plt.figure(figsize=(12, 6))
        plt.scatter(wavelengths, sensitivities, alpha=0.5, s=5)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Relative Sensitivity')
        plt.title('Detector Sensitivity as a Function of Wavelength')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join("results", "sensitivity_vs_wavelength.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return sensitivity_map