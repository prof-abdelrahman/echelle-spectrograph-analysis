import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
import os

class Task_c_and_d_find_and_fit_the_orders:
    @staticmethod
    def flat_frame_image(flat):
        """
        Display the flat frame image.

        Args:
            flat (np.ndarray): The flat frame image.
        """
        plt.figure(figsize=(12, 6))
        plt.imshow(flat, cmap='viridis', vmin=np.percentile(flat, 5), vmax=np.percentile(flat, 95))
        plt.title('Flat Frame')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.colorbar(label='Intensity')
        plt.savefig(os.path.join("results","flat_frame_image.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def identify_orders(flat):
        """
        Identify the orders in the flat field image.
        
        Args:
            flat (np.ndarray): The flat field image.
            
        Returns:
            list: The row indices of the detected order centers.
        """
        # Create a profile by summing along columns (assuming orders run horizontally)
        order_profile = np.sum(flat, axis=1)
        
        # Find the peaks, which correspond to the centers of the orders
        peaks, _ = signal.find_peaks(order_profile, height=np.mean(order_profile), distance=20)
        
        plt.figure(figsize=(12, 6))
        plt.plot(order_profile)
        plt.plot(peaks, order_profile[peaks], 'ro')
        plt.title('Order Profile with Detected Order Centers')
        plt.xlabel('Row Index')
        plt.ylabel('Intensity')
        plt.legend(['Order Profile', 'Detected Order Centers'])
        plt.savefig(os.path.join("results","order_identification.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return peaks

    @staticmethod
    def fit_orders(flat, order_centers):
        """
        Fit the orders in the flat field image.
        
        Args:
            flat (np.ndarray): The flat field image.
            order_centers (list): The row indices of the detected order centers.
            
        Returns:
            order_fits (list): List of tuples containing (x, y_fit, poly_coeffs) for polynomial fits
            slit_fits (list): List of tuples containing (y, data, popt) for Gaussian fits
        """
        # Function to fit a Gaussian to the slit profile for a specific order
        def gaussian(x, amplitude, mean, sigma):
            return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))
        
        order_fits = []
        slit_fits = []
        successful_order_fits = 0
        successful_slit_fits = 0
        
        # Process all detected orders
        for order_center in order_centers:
            # Extract a horizontal profile near the order center
            order_slice = flat[order_center, :]
            x_order = np.arange(len(order_slice))
            
            # Fit the order direction profile with a high-order polynomial
            try:
                poly_coeffs = np.polyfit(x_order, order_slice, 10)
                poly_fit = np.polyval(poly_coeffs, x_order)
                order_fits.append((x_order, poly_fit, poly_coeffs))
                successful_order_fits += 1
            except Exception as e:
                print(f"Failed to fit polynomial for order at row {order_center}: {e}")
            
            # Extract a vertical profile (slit direction) centered on a bright part of the order
            bright_col = np.argmax(order_slice)
            slit_range = 100  # Can be Adjusted based on my data
            y_slit = np.arange(max(0, order_center - slit_range), min(flat.shape[0], order_center + slit_range))
            slit_slice = flat[y_slit, bright_col]
            
            # Fit a Gaussian to the slit profile
            try:
                p0 = [np.max(slit_slice), order_center, 10]  # Initial guess for Gaussian parameters
                popt, _ = curve_fit(gaussian, y_slit, slit_slice, p0=p0)
                slit_fits.append((y_slit, slit_slice, popt))
                successful_slit_fits += 1
            except Exception as e:
                print(f"Failed to fit Gaussian for order at row {order_center}: {e}")
        
        print(f"Total orders detected: {len(order_centers)}")
        print(f"Orders successfully fitted (order direction): {successful_order_fits}/{len(order_centers)}")
        print(f"Orders successfully fitted (slit direction): {successful_slit_fits}/{len(order_centers)}")
        
        # 1. First show a single order with detailed analysis
        if len(order_fits) > 0 and len(slit_fits) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Get the first order fit
            x, y_fit, poly_coeffs = order_fits[0]
            order_center = order_centers[0]
            
            # Plot order direction fit
            axes[0].plot(x, flat[order_center, :], 'k-', label='Data')
            axes[0].plot(x, y_fit, 'r-', label='Polynomial Fit')
            axes[0].set_title(f'Order Direction Fit (Order 1 at row {order_center})')
            axes[0].set_xlabel('Column Index')
            axes[0].set_ylabel('Intensity')
            axes[0].legend()
            
            # Plot slit direction fit
            y, data, popt = slit_fits[0]
            axes[1].plot(y, data, 'k-', label='Data')
            axes[1].plot(y, gaussian(y, *popt), 'r-', label=f'Gaussian Fit (σ={popt[2]:.2f})')
            axes[1].set_title(f'Slit Direction Fit (Order 1 at row {order_center})')
            axes[1].set_xlabel('Row Index')
            axes[1].set_ylabel('Intensity')
            axes[1].legend()
            
            plt.tight_layout()
            fig.savefig(os.path.join("results", "single_order_fit.png"), dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Show the first 10 orders
        if len(order_fits) >= 10:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot order direction fits for first 10 orders
            for i in range(10):
                x, y_fit, _ = order_fits[i]
                order_center = order_centers[i]
                color = plt.cm.viridis(i/10)  # Use a colormap for different orders
                axes[0].plot(x, flat[order_center, :], color=color, alpha=0.3)
                axes[0].plot(x, y_fit, color=color, alpha=0.7, label=f'Order {i+1}')
            
            axes[0].set_title('Order Direction Fits (First 10 Orders)')
            axes[0].set_xlabel('Column Index')
            axes[0].set_ylabel('Intensity')
            axes[0].legend(loc='upper right', fontsize=8)
            
            # Plot slit direction fits for first 10 orders
            for i in range(min(10, len(slit_fits))):
                y, data, popt = slit_fits[i]
                color = plt.cm.viridis(i/10)
                axes[1].plot(y, data, color=color, alpha=0.3)
                axes[1].plot(y, gaussian(y, *popt), color=color, alpha=0.7, label=f'Order {i+1}')
            
            axes[1].set_title('Slit Direction Fits (First 10 Orders)')
            axes[1].set_xlabel('Row Index')
            axes[1].set_ylabel('Intensity')
            axes[1].legend(loc='upper right', fontsize=8)
            
            plt.tight_layout()
            fig.savefig(os.path.join("results","first_10_orders_fits.png"), dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Show the remaining orders (assuming there are more than 10)
        if len(order_fits) > 10:
            remaining = len(order_fits) - 10
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot order direction fits for remaining orders
            for i in range(10, len(order_fits)):
                x, y_fit, _ = order_fits[i]
                order_center = order_centers[i]
                color = plt.cm.plasma((i-10)/remaining)  # Different colormap for distinction
                axes[0].plot(x, flat[order_center, :], color=color, alpha=0.3)
                axes[0].plot(x, y_fit, color=color, alpha=0.7, label=f'Order {i+1}')
            
            axes[0].set_title(f'Order Direction Fits (Remaining {remaining} Orders)')
            axes[0].set_xlabel('Column Index')
            axes[0].set_ylabel('Intensity')
            if remaining <= 15:  # Only show legend if not too crowded
                axes[0].legend(loc='upper right', fontsize=8)
            
            # Plot slit direction fits for remaining orders
            slit_remaining = max(0, len(slit_fits) - 10)
            for i in range(10, len(slit_fits)):
                y, data, popt = slit_fits[i]
                color = plt.cm.plasma((i-10)/slit_remaining if slit_remaining > 0 else 0)
                axes[1].plot(y, data, color=color, alpha=0.3)
                axes[1].plot(y, gaussian(y, *popt), color=color, alpha=0.7, label=f'Order {i+1}')
            
            axes[1].set_title(f'Slit Direction Fits (Remaining {slit_remaining} Orders)')
            axes[1].set_xlabel('Row Index')
            axes[1].set_ylabel('Intensity')
            if slit_remaining <= 15:  # Only show legend if not too crowded
                axes[1].legend(loc='upper right', fontsize=8)
            
            plt.tight_layout()
            fig.savefig(os.path.join("results", "remaining_orders_fits.png"), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Also save a figure showing all fits overlaid on the original flat field
        plt.figure(figsize=(14, 10))
        plt.imshow(flat, cmap='viridis', aspect='auto', 
                vmin=np.percentile(flat, 5), vmax=np.percentile(flat, 95))
        
        # Overlay the order centers
        for i, center in enumerate(order_centers):
            plt.axhline(y=center, color='r', alpha=0.5)
            
            # Add text labels for some orders (skip some to avoid overcrowding)
            if i % 5 == 0:
                plt.text(flat.shape[1]-240, center, f"Order {i+1}", color='white', fontsize=8)
        
        plt.title('Flat Field with All Detected Orders')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.colorbar(label='Intensity')
        plt.savefig(os.path.join("results","all_orders_overlay.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return order_fits, slit_fits