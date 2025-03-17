import numpy as np
import matplotlib.pyplot as plt
import os

class Task_b_dark_frame_statistics:
    @staticmethod
    # Task b functions
    def analyze_dark_frames(darks):
        """
        For each pixel compute the statistics across dark frames.
        Then summarize the statistics (mean and std distribution across the detector).
        """
        dark_mean = np.mean(darks, axis=0)
        dark_std = np.std(darks, axis=0)
        
        overall_mean = np.mean(dark_mean)
        overall_median = np.median(dark_mean)
        overall_std_mean = np.std(dark_mean)
        overall_mean_std = np.mean(dark_std)
        
        print("\nDark Frames Statistics:")
        print(f"- Mean bias level (Overall mean of pixel dark level): {overall_mean:.2f} counts")
        print(f"- Overall median of pixel dark level: {overall_median:.2f} counts")
        print(f"- Overall standard deviation of pixel dark level: {overall_std_mean:.2f} counts")
        print(f"- Mean noise level (std) (Overall mean of pixel dark level standard deviation): {overall_mean_std:.2f} counts")
        print(f"- Min/Max bias values: {np.min(dark_mean):.2f} / {np.max(dark_mean):.2f} counts")
        
        return dark_mean, dark_std, overall_mean, overall_median, overall_std_mean, overall_mean_std
    
    # Task b plots' functions
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