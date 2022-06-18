"""
Plot contains functions to plot transient absorbtion data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_measure_map(measure_map, delays, lambdas,  # data to plot
                     title=None, 
                     figsize=None,  
                     maxDelay=None,
                     delayRange=None,
                     colorbar = False, colorbar_label=None,
                     vmin=None, vmax=None,
                     cmap=None,
                     electronvolt=False,
                     logscale=False,
                     mOD=False):
        """
        Plots a Stratus UV measure map.
        
        Parameters
        ----------
        measure_map : numpy.array
            2D array to plot. Each element represents the DT/T or mOD value
            for a specific delay and wavelenght.
        delays : numpy.array
            1D array containing the experimental delays.
        lambdas : numpy.array
            1D array containing the wavelength recorded by the spectrometer.
        title : str, optional
            Title of the figure
        maxDelay : int, optional
            If provided plots the measure map only until this delay time. 
            `delayRange` overrides this argument if provided. 
        delayRange : tuple, optional
            Plots the map between the delays (start_time, end_time) if provided.
        figsize : (value, value), optional
            Size of the figure. Refere to matplotlib doc.
        colorbar : bool, default: False
            If `True` show the colorbar on the map. If `True` and `colorbar_label`
            is not provided, it assumes `colorbar_label`="dT/T %"
        colorbar_label : str, optional
            String to show over the colorbar. If provided plots the colorbar 
            even if `colorbar`is not provided.
        vmax : float, optional
            Maximum 2D value to render. It has to be intended as a "cutoff":
            everything above this value will be threated as `vmax`.
        vmin : float, optional
            Same as `vmax`.
        cmap : string, optional
            Matplotlib cmap. Refer to its documentation.
        electronvolt : bool, default: False
            If True converts the photon wavelength to eV.
        mOD : bool, default: False
            If True plots in mOD, if False in dT/T (percentage).
        logscale : bool, default: False
            If True plots in log scale.
        """
        #TODO add mod here!

        ### Set fig size
        if figsize==None:
            fig, ax = plt.subplots()
        else:    
            fig, ax = plt.subplots(figsize=figsize)
        
        # convert the lambda axis to electronvolt
        if electronvolt:
            h = 6.6261*1E-34
            c = 299792458
            ev = 1.602176565 * 1E-19
            lambdas =  (1E9*h*c/lambdas)
        
        if colorbar and (colorbar_label is None): #assume it is dt/t
            colorbar_label="dT/T %"
        if mOD:
            colorbar_label="mOD"
            measure_map = -1000*np.log10(measure_map/100 + 1)
        
        if delayRange is not None:
            if type(delayRange) is not tuple:
                raise TypeError("`rangeTimePlot` must be a two elements tuple")
            max_delay =  np.argmin(np.abs(delays-delayRange[1]))
            min_delay =  np.argmin(np.abs(delays-delayRange[0]))
            delays = delays[min_delay:max_delay]
            measure_map = measure_map[:, min_delay:max_delay]

        elif maxDelay is not None:
            max_delay =  np.argmin(np.abs(delays-maxDelay))
            delays = delays[:max_delay]
            measure_map = measure_map[:, :max_delay]
        
        ### Plot
        #pos = ax.pcolormesh(delays,lambdas, measure_map, cmap="seismic", shading="gouraud")
        if logscale:
            pos = ax.pcolormesh(delays,lambdas, measure_map,  shading="gouraud", vmin=vmin, vmax = vmax, cmap=cmap, norm=colors.LogNorm())
        else:
            pos = ax.pcolormesh(delays,lambdas, measure_map,  shading="gouraud", vmin=vmin, vmax = vmax, cmap=cmap)        
           
        ### Set colorbar
        if colorbar or colorbar_label:
           fig.colorbar(pos, ax=ax, label=colorbar_label)
        ### Label and title
        if electronvolt:
            plt.ylabel("Probe photon energy (eV)")
        else:
            plt.ylabel("Probe wavelength (nm)")

        plt.xlabel("Delay time (fs)")
        if title is not None:
            plt.title(title)