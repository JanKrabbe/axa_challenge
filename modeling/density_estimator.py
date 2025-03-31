from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from sklearn.neighbors import KernelDensity


class DensityEstimator():
    """
    Performs Kernel Density Estimation (KDE) and raster-based density estimation. Can be used to 
    plot the density and heatmaps.
    """
    def __init__(self, data, bandwidth=1, kernel='gaussian'):
        """
        Arguments:
            data (ndarray): Array of shape (n_samples, 2) (e.g., [[x, y], ...])
            bandwidth (float):  Bandwidth of KDE
            kernel (str): Sklearn KDE kernel to use 
        """
        self.data = data
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde_model = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde_model.fit(data)

        self.x_min = self.data[:, 0].min()
        self.x_max = self.data[:, 0].max()
        self.y_min = self.data[:, 1].min()
        self.y_max = self.data[:, 1].max()
    
    def evaluate_grid(self, grid_size=1000):
        """
        Evaluates the KDE on a grid.
        
        Arguments:
            grid_size (int): Number of grid points per axis
        
        Returns:
            xx, yy: Meshgrid arrays
            density: 2D array of density values
        """
        x_grid = np.linspace(self.x_min, self.x_max, grid_size)
        y_grid = np.linspace(self.y_min, self.y_max, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_samples = np.vstack([xx.ravel(), yy.ravel()]).T
        log_dens = self.kde_model.score_samples(grid_samples)
        density = np.exp(log_dens).reshape(xx.shape)
        
        mid = np.median(density)
        scale = np.std(density)
        sigmoid = lambda x: (1 / (1 + np.exp(-(x - mid) / scale)) - 0.5) * 2
        normalized_density = sigmoid(density)
        return xx, yy, normalized_density
    
    def histogram2d(self, bins=1000, density=False):
        """
        Computes a 2D histogram (raster-based density estimation) from coordinate data.
        
        Arguments:
            data (ndarray): Array of shape (n_samples, 2) (e.g., [[x, y], ...])
            bins (int): Number of bins per axis

        Returns:
            H: 2D histogram array
            xedges, yedges: Bin edges
        """
        x = self.data[:, 0]
        y = self.data[:, 1]
        H, xedges, yedges = np.histogram2d(
            x, y, bins=bins, range=[[self.x_min, self.x_max], [self.y_min, self.y_max]], density=density)

        return H, xedges, yedges
    
    @staticmethod
    def normalized_histogram2d(
        numerator: DensityEstimator, 
        denominator:DensityEstimator, 
        bins=1000,
        title='Histogram Heatmap'):
        """
        Normalizes one density estimate by dividing it by another.
        
        Arguments:
            numerator (ndarray): Density to be normalized
            denominator (ndarray): Density used for normalization
            bins (int): Number of bins to use along each axis for the 2D histogram
            title (str): Title of the heatmap
        
        Returns:
            normalized_density (ndarray): The result of the division
        """

        x = numerator.data[:, 0]
        y = numerator.data[:, 1]
        H_num, xedges_num, yedges_num = np.histogram2d(
            x, y, bins=bins, range=[[denominator.x_min, denominator.x_max], [denominator.y_min, denominator.y_max]])

        x = denominator.data[:, 0]
        y = denominator.data[:, 1]
        H_den, xedges_den, yedges_den = np.histogram2d(
            x, y, bins=bins, range=[[denominator.x_min, denominator.x_max], [denominator.y_min, denominator.y_max]])            

        if (xedges_den == xedges_num).sum() !=  xedges_den.shape:
            RuntimeError("Unexpected error: histogram bin edges to not match.")

        safe_denom = np.where(H_den == 0, 1e-10, H_den)
        H_norm = np.where(H_den < 1, 0, H_num/H_den)
        # H_norm = H_num / (H_den+1) #safe_denom

        extent = [denominator.x_min, denominator.x_max, denominator.y_min, denominator.y_max]
        DensityEstimator.plot_heatmap(H_norm, xedges_den, yedges_den, extent, title, density=True)


    def plot_histogram_heatmap(self, bins=1000, title='Histogram Heatmap', density=False):
        """
        Computes a 2D histogram (raster-based density) and plots it as a heatmap.
 
         Arguments:
            bins (int): Number of bins to use along each axis for the 2D histogram
            title (str): Title for the heatmap plot
            density (bool): If True, H is interpreted as a normalized density (probability density function) and plotted accordingly; if False, H is treated as raw counts and plotted using logarithmic normalization   
        """
   
        H, xedges, yedges = self.histogram2d(bins=bins, density=density)
        extent = [self.x_min, self.x_max, self.y_min, self.y_max]
        self.plot_heatmap(H, xedges, yedges, extent, title, density)

        
    @staticmethod
    def plot_heatmap(H, xedges, yedges, extent, title='Histogram Heatmap', density=False):
        """
        Plots a 2D heatmap from a histogram array using matplotlib.

        Arguments:
            H (ndarray): 2D array containing histogram counts or density values
            xedges (ndarray): Array of bin edges along the x-axis (not directly used, but typically provided)
            yedges (ndarray): Array of bin edges along the y-axis (not directly used, but typically provided)
            extent (list or tuple): The [x_min, x_max, y_min, y_max] boundaries for the plot
            title (str): Title of the heatmap
            density (bool): Flag indicating if H represents a probability density function (True) or raw counts (False)

        """
        plt.figure(figsize=(10, 8))
        if density:
            plt.imshow(H.T, extent=extent, origin='lower')
            plt.colorbar(label='Probability density function')
        else:
            plt.imshow(H.T, extent=extent, origin='lower', norm=LogNorm(vmin=1, vmax=H.max()))
            plt.colorbar(label='Counts (log scale)')

        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.title(title)
        plt.show()


    def plot_kde_heatmap(self, grid_size=1000, title='KDE Heatmap', show_scatter=False):
        """
        Plots the KDE-estimated density on a grid as a heatmap using logarithmic normalization.
        
        Arguments:
            grid_size (int): Number of grid points per axis
            title (str): Title of the plot
            show_scatter (bool): If True, overlays the original data points
        """

        xx, yy, density = self.evaluate_grid(grid_size=grid_size)
        plt.figure(figsize=(10, 8))
        plt.imshow(density, extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                   origin='lower', aspect='auto')
        plt.colorbar(label='Normalized Density')
        if show_scatter and self.data is not None:
            plt.scatter(self.data[:, 0], self.dataX[:, 1], s=1, c='blue', alpha=0.3)
        plt.xlabel("x (meters)")
        plt.ylabel("y (meters)")
        plt.title(title)
        plt.show()