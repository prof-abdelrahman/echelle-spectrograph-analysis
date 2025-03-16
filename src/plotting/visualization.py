import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage import filters
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import os

class Visualization:
    @staticmethod
    def visualize_mean_dark_frame(dark_mean):
        """Visualize the mean dark frame."""
        plt.figure(figsize=(10, 10))
        plt.imshow(dark_mean, cmap="gray", origin="upper")
        plt.colorbar(label='Intensity')
        plt.title('Mean Dark Frame')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.savefig(os.path.join("results", "mean_dark_frame.png"), dpi=300)
        plt.show()
        
    @staticmethod
    def histogram_of_bias_values(dark_mean, mean_dark_summary):
        """Plot histogram of bias values."""
        plt.figure(figsize=(10, 6))
        plt.hist(dark_mean.flatten(), bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Histogram of Dark Frame Bias Values')
        plt.xlabel('Bias Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.axvline(mean_dark_summary, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dark_summary:.2f}')
        plt.legend()
        plt.savefig(os.path.join("results", "histogram_bias_values.png"), dpi=300)
        plt.show()
        
    @staticmethod
    def visualize_std_dark_frame(dark_std):
        """Visualize the standard deviation dark frame."""
        plt.figure(figsize=(10, 10))
        plt.imshow(dark_std, cmap='viridis', origin='upper')
        plt.colorbar(label='Standard Deviation')
        plt.title('Dark Frame Noise (Standard Deviation)')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.savefig(os.path.join("results", "std_dark_frame.png"), dpi=300)
        plt.show()
        
    @staticmethod
    def histogram_of_noise_values(dark_std, mean_noise_summary):
        """Plot histogram of noise values."""
        plt.figure(figsize=(10, 6))
        plt.hist(dark_std.flatten(), bins=100, color='orange', edgecolor='black', alpha=0.7)
        plt.title('Histogram of Dark Frame Noise Values')
        plt.xlabel('Noise Value (Standard Deviation)')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.axvline(mean_noise_summary, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_noise_summary:.2f}')
        plt.legend()
        plt.savefig(os.path.join("results", "histogram_noise_values.png"), dpi=300)
        plt.show()
    
    @staticmethod
    def flat_field_image(flat_field):
        """Visualize the flat field image."""
        plt.figure(figsize=(10, 10))
        plt.imshow(flat_field, cmap='gray', origin='lower', aspect='auto')
        plt.colorbar(label='Intensity')
        plt.title('Flat Field Frame Image')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.savefig(os.path.join("results", "flat_field_image.png"), dpi=300)
        plt.show()
        
    @staticmethod
    def flat_field_image2(flat_field):
        """Visualize the flat field image."""
        # Display the flat field image to visualize the orders
        plt.figure(figsize=(10, 10))
        plt.imshow(np.log(flat_field + 1), cmap='viridis')  # log scale for better visualization
        plt.colorbar(label='Log Intensity')
        plt.title('Flat Field Image (Log Scale)')
        plt.xlabel('Order Direction (Pixels)')
        plt.ylabel('Slit Direction (Pixels)')
        plt.savefig(os.path.join("results", "flat_field_image2.png"), dpi=300)
        plt.show()
        
    @staticmethod
    def spectral_edge_detection(flatfield, name):
        """Perform spectral edge detection on an image."""
        edge_sobel = sobel(flatfield)
        plt.figure(figsize=(10, 10))
        plt.imshow(edge_sobel, cmap='gray', origin='lower')
        plt.colorbar(label='Intensity')
        plt.title(f'Spectral Edge Detection for {name}')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.savefig(os.path.join("results", "spectral_edge_detection.png"), dpi=300)
        plt.show()
        
        return edge_sobel
    
    @staticmethod
    def spectral_edge_detection_distinct(flatfield, name):
        # Apply edge detection (Sobel filter)
        edges = filters.sobel(flatfield)

        # Plot the detected edges
        plt.figure(figsize=(10, 10))
        plt.imshow(flatfield, cmap="gray", origin="lower", aspect="auto")
        plt.colorbar(label="Intensity")
        plt.contour(edges, colors='red', linewidths=0.1)  # Highlight edges in red
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.title(f"Detected Spectral Orders (Edge Detection) for {name}")
        plt.savefig(os.path.join("results", "spectral_edge_detection_distinct.png"), dpi=300)	
        plt.show()
        
        return edges
    
    @staticmethod
    def slit_profile_and_detected_orders(slit_profile, order_centers):
        """Perform spectral edge detection on an image."""
        # Plot the slit profile and detected orders
        plt.figure(figsize=(12, 6))
        plt.plot(slit_profile)
        plt.plot(order_centers, slit_profile[order_centers], 'x', color='red')
        plt.title('Slit Direction Profile with Detected Orders')
        plt.xlabel('Pixel Position (Slit Direction)')
        plt.ylabel('Integrated Intensity')
        plt.grid(True)
        plt.savefig(os.path.join("results", "slit_profile_detected_orders.png"), dpi=300)
        plt.show()
        
    @staticmethod
    def polynomial_fit(flatfield, name):
        """Perform polynomial fit on the data."""
        # Select a vertical slice (middle of the image)
        col_index = flatfield.shape[1] // 2  # Middle column
        slice_intensity = flatfield[:, col_index]

        # Detect peaks (bright spectral orders)
        peaks, _ = find_peaks(slice_intensity, height=np.max(slice_intensity) * 0.5)

        # Fit a polynomial to the detected order positions
        poly_coeffs = np.polyfit(peaks, np.arange(len(peaks)), 3)  # 3rd-degree polynomial
        fitted_curve = np.polyval(poly_coeffs, np.arange(flatfield.shape[0]))

        # Plot detected edges and fitted polynomial
        plt.figure(figsize=(10, 10))
        plt.imshow(flatfield, cmap='gray', origin='lower')
        plt.plot(col_index * np.ones_like(fitted_curve), fitted_curve, 'r-', linewidth=2)  # Overlay fitted curve
        plt.title(f'Fitted Spectral Orders for {name}')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.savefig(os.path.join("results", "fitted_spectral_orders.png"), dpi=300)
        plt.show()
        
    @staticmethod
    def polynomial_fit_distinct(flatfield, name):
        # Define a quadratic function to fit
        def cubic_fit(x, a, b, c, d):
            return a*x**3 + b*x**2 + c*x + d

        # Example detected points (replace with actual detected order data)
        x_vals = np.linspace(0, 4000, 15)  # Sample X-coordinates
        y_vals = 2000 + 0.0001*x_vals**2  # Simulated curved spectral order

        # Fit a quadratic polynomial
        popt, _ = curve_fit(cubic_fit, x_vals, y_vals)

        # Generate fitted curve
        x_fit = np.linspace(0, 4000, 100)
        y_fit = cubic_fit(x_fit, *popt)

        # Plot detected orders and fit
        plt.figure(figsize=(10, 10))
        plt.imshow(np.zeros((4000, 4000)), cmap="gray", origin="lower", aspect="auto")  # Dummy image
        plt.scatter(x_vals, y_vals, color="red", marker="o", label="Detected Points")  # Detected points in red
        plt.plot(x_fit, y_fit, "b--", linewidth=2, label="Quadratic Fit")  # Fitted curve in blue dashed
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.legend()
        plt.title("Fitted Spectral Orders for Flat Frame")
        plt.savefig(os.path.join("results", "polynomial_fit_distinct.png"), dpi=300)
        plt.show()