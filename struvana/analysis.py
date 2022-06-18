"""
Set of classes and function to analyze the Stratus UV preprocessed data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from os.path import isdir
import matplotlib
import matplotlib.ticker as tck
from matplotlib.ticker import FuncFormatter, MultipleLocator
from . import plot

class OscillationAnalysis:
    """
    Class to study oscillations in the transient absorbtion map.
    
    Parameters
    ----------
    npz_path : str
        Path of the npz archive containing the preprocessed data.
    save_path : str, optional
        Directory where to save plots.
    save_name : str, optional
        Name of the dataset.
    
    Attributes
    ----------
    measure_map : ndarray
        Preprocessed measure map.
    delays : ndarray
        Delays of the pump-probe. 
    lambdas : ndarray
        Wavelength of data. 
    oscillations : ndarray
        Measure map with only the fast oscillations.
    frequency_map : ndarray
        Absolute value of the fourier transform of the oscillation map.
    phase_map : ndarray
        Phase of the fourier map.
    """
    def __init__(self, npz_path, save_path=None, save_name=None):
        data = np.load(npz_path)
        self.measure_map = data["measure_map"]
        self.delays = data["delays"]
        self.lambdas = data["lambdas"]
        self.save_path = save_path
        self.save_name = save_name
        self.oscillations = None
        self.frequency_map = None
        self.phase_map = None
    
    
    def plot_map(self, title=None, 
                       figsize=None, 
                       log=False,
                       maxDelay=None, 
                       delayRange=None, 
                       mOD=False,  
                       vmin=None, 
                       vmax=None, 
                       electronvolt=False):
        """
        Plot the measure map.

        Parameters
        ----------
        title : str, optional
            Title of the figure
        figsize : (value, value), optional
            Size of the figure. Refere to matplotlib doc.
        maxDelay : int, optional
            If provided plots the measure map only until this delay time. 
            `delayRange` overrides this argument if provided. 
        delayRange : tuple, optional
            Plots the map between the delays (start_time, end_time) if provided.
        log : bool, default: False
            If True plots in log scale. May be useful to find the valid wavelength.
        mOD : bool, default: False
            If True plots in mOD, if False in dT/T (percentage).
        vmax : float, optional
            Maximum 2D value to render. It has to be intended as a "cutoff":
            everything above this value will be threated as `vmax`.
        vmin : float, optional
            Same as `vmax`.
        electronvolt : bool, default: False
            If True converts the photon wavelength to eV.
        """
 

        plot.plot_measure_map(self.measure_map, 
                              self.delays, 
                              self.lambdas,  
                              title=title, 
                              figsize=figsize,  
                              colorbar=True,
                              vmin=vmin, vmax=vmax,
                              logscale=log,
                              mOD=mOD,
                              maxDelay=maxDelay, 
                              delayRange=delayRange,
                              electronvolt=electronvolt)

    @staticmethod
    def _average_smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    @staticmethod
    def _smooth( measure_map, delays, lambdas, smoothWindowDelayFS, smoothWindowLambdaNM):
        measure_map = measure_map.copy()
        # time
        smoothWindowDelay = np.round(smoothWindowDelayFS / (delays[1] - delays[0]))
        if np.mod(smoothWindowDelay,2) == 0:
            smoothWindowDelay += 1 
        # TODO is it necessary???
        for i in range(1):
            for j in range(measure_map.shape[0]):
                measure_map[j] = OscillationAnalysis._average_smooth(measure_map[j], 
                                                     int(smoothWindowDelay))
        # lambda
        smoothWindowLambda = np.round(smoothWindowLambdaNM / (lambdas[1] - lambdas[0]))
        if np.mod(smoothWindowLambda,2) == 0:
            smoothWindowLambda += 1 
        
        for i in np.arange(1,4+1).reshape(-1):
            for j in range(measure_map.shape[1]):
                measure_map[:,j] = OscillationAnalysis._average_smooth(measure_map[:,j], 
                                               int(smoothWindowLambda))
        return measure_map


    def get_oscillations(self, smoothWindowDelayFS, maxDelay=None,
                        delayRange=None, mOD=False, vmin=None, vmax=None, electronvolt=False, figsize=None ):
        """
        Finds fast oscillations. It subtracts to the original measured map a smoothed one 
        in the delay axis in order to remove the slowly varying features. 
        
        Parameters
        ----------
        smoothWindowDelayFS : int
            Moving average window size (used to smooth the map), should be large enough to smooth away
            the fast oscillations. Numbers between 100 and 250 may be appropriate.
        maxDelay : int, optional
            If provided plots the measure map only until this delay time. 
            `delayRange` overrides this argument if provided. 
        delayRange : tuple, optional
            Plots the map between the delays (start_time, end_time) if provided.
        mOD : bool, default: False
            If True plots in mOD, if False in dT/T (percentage).
        vmax : float, optional
            Maximum 2D value to render. It has to be intended as a "cutoff":
            everything above this value will be threated as `vmax`.
        vmin : float, optional
            Same as `vmax`.
        electronvolt : bool, default: False
            If True converts the photon wavelength to eV.
        figsize : (value, value), optional
            Size of the figure. Refere to matplotlib doc.
        """

        self.oscillations = self.measure_map - self._smooth(self.measure_map,
                                                           self.delays,
                                                           self.lambdas,
                                                           smoothWindowDelayFS,
                                                           0)
  
        # TODO is that necessary? 
        self.oscillations = self._smooth(self.oscillations,
                                        self.delays,
                                        self.lambdas,
                                        0,
                                        5)

        plot.plot_measure_map(self.oscillations, 
                              self.delays, 
                              self.lambdas,  
                              title="Oscillations", 
                              figsize=figsize,  
                              delayRange=delayRange,
                              maxDelay=maxDelay,
                              colorbar=True,
                              vmin=vmin, vmax=vmax,
                              electronvolt=electronvolt,
                              mOD=mOD)
       
        if self.save_name is not None and self.save_path is not None:
            plt.savefig(self.save_path+"/"+self.save_name+"_oscillations.pdf")
    

    def fourier_transform(self, start_delay, end_delay):
        """
        Computes and plots the fourier transform (not the FFT) of the transient absorption
        fast oscillation map. The fourier transform is computed in the range between `start_delay`
        and `end_delay`. It requires `get_oscillations` to be run before. If you want to compute 
        the fourier transform on the original preprocess data you have to set the attribute
         `oscillations`=`measure_map`.
        
        Parameters
        ----------
        start_delay : int
            Initial delay used to define the fourier transform range.
        end_delay : int
            Final delay used to define the fourier transform range.
        """

        if self.oscillations is None:
            print("Error: you have to compute oscillations first.")
            return
        idx_start = np.argwhere(self.delays>start_delay)[0][0]
        idx_end = np.argwhere(self.delays<end_delay)[-1][0]
        frequencyFFT = np.linspace(0,0.06,2000)
        fourier_transform = self.oscillations[:,idx_start:idx_end] @ np.exp(np.outer(self.delays[idx_start:idx_end], - 1j * 2 * np.pi * frequencyFFT))
        # to cm^-1
        frequencyFFT *= 33357
        self.frequency_map = np.abs(fourier_transform)
        self.frequency_map /= np.max(self.frequency_map)
        self.phase_map = np.angle(fourier_transform)

        max_index = np.argmin(np.abs(frequencyFFT-1000))
        plt.figure()
        #we don't have resolution above 1000 cm-1

        plt.imshow(self.frequency_map[:, :max_index], 
                       aspect="auto", 
                       extent=[frequencyFFT[0], 
                               frequencyFFT[max_index],
                               self.lambdas[0],
                               self.lambdas[-1]], 
                        origin="lower")
        
        plt.ylabel("Probe wavelength (nm)")
        plt.xlabel("Frequency (cm^-1)")
        plt.title("Fourier transform")
        if self.save_name is not None and self.save_path is not None:
            plt.savefig(self.save_path+"/"+self.save_name+"_frequency.pdf")
        fig, ax = plt.subplots()
        pos = ax.imshow(self.phase_map, 
                       aspect="auto", 
                       extent=[-np.pi, 
                               np.pi,
                               self.lambdas[0],
                               self.lambdas[-1]], 
                        origin="lower")
        # to set x axis scale to multiple of pi
        ax.xaxis.set_major_formatter(FuncFormatter(
                                        lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'
))
        ax.xaxis.set_major_locator(tck.MultipleLocator(base=np.pi))
        plt.ylabel("Probe wavelength (nm)")
        plt.xlabel("Phase")
        plt.title("Fourier transform")
        if self.save_name is not None and self.save_path is not None:
            plt.savefig(self.save_path+"/"+self.save_name+"_phase.pdf")