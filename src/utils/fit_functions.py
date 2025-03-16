# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit#
import os

class Fit_functions:
    @staticmethod
    def flat_fram_fit_function(flat_field,order_centers):
        """
        Function to fit the flat field image along the order direction.
        
        Args:
        flat_field (ndarray): 2D array representing the flat field image.
        order_centers (ndarray): Array of order centers.
        
        Returns:
        None
        """
        # Function to fit - typically echelle orders follow a polynomial distribution
        def polynomial_function(x, *params):
            order = len(params) - 1
            result = 0
            for i, param in enumerate(params):
                result += param * (x ** (order - i))
            return result

        # Function to fit Gaussian to each order cross-section
        def gaussian(x, amplitude, center, sigma):
            return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

        # For each order, extract and fit a profile along the order direction
        # We'll sample a few positions along the order
        order_fits = []
        order_indices = range(len(order_centers))

        plt.figure(figsize=(15, 10))

        for i, center in enumerate(order_centers):
            # Extract a thin slice along the order to get its profile
            # We use a band of a few pixels centered on the order
            half_width = 5  # Adjust based on order width
            order_band = flat_field[center-half_width:center+half_width+1, :]
            order_profile = np.mean(order_band, axis=0)
            
            # X values are the pixel positions along the order
            x = np.arange(len(order_profile))
            
            # Try polynomial fits of different degrees
            best_fit = None
            best_degree = 0
            lowest_error = float('inf')
            
            for degree in range(2, 6):  # Try polynomials from degree 2 to 5
                try:
                    # Initial guess for polynomial coefficients
                    p0 = np.ones(degree + 1)
                    
                    # Fit the polynomial
                    params, covariance = curve_fit(
                        lambda x, *params: polynomial_function(x, *params),
                        x, order_profile, p0=p0
                    )
                    
                    # Calculate fit error
                    fit_values = polynomial_function(x, *params)
                    error = np.sum((order_profile - fit_values)**2)
                    
                    if error < lowest_error:
                        lowest_error = error
                        best_fit = params
                        best_degree = degree
                except:
                    continue
            
            if best_fit is not None:
                # Store the fit parameters
                order_fits.append({
                    'order_index': i,
                    'center': center,
                    'degree': best_degree,
                    'params': best_fit
                })
                
                # Plot the original profile and the fit
                fit_values = polynomial_function(x, *best_fit)
                
                plt.subplot(3, 3, min(i+1, 9))  # Show first 9 orders
                plt.plot(x, order_profile, 'b-', alpha=0.5, label='Data')
                plt.plot(x, fit_values, 'r-', label=f'Poly fit (deg={best_degree})')
                plt.title(f'Order {i} at y={center}')
                plt.xlabel('Pixel Position (Order Direction)')
                plt.ylabel('Intensity')
                plt.legend()
                
                if i >= 8:  # Only plot first 9 orders
                    break

        plt.tight_layout()
        plt.savefig(os.path.join("results", "order_fits.png"), dpi=300)

        # Analyze the fitting results
        plt.figure(figsize=(10, 6))
        degrees = [fit['degree'] for fit in order_fits]
        plt.plot(degrees, 'o-')
        plt.title('Polynomial Degree Used for Each Order')
        plt.xlabel('Order Index')
        plt.ylabel('Polynomial Degree')
        plt.grid(True)
        plt.savefig(os.path.join("results", "fit_degrees.png"), dpi=300)

        # Save the fitting results
        np.save(os.path.join("results", "order_fits.npy"), order_fits)

        print(f"Total orders detected: {len(order_centers)}")
        print(f"Orders successfully fitted: {len(order_fits)}")
        print("Analysis complete!")