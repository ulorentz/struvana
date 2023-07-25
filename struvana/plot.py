"""
Plot contains functions to plot transient absorbtion data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Slider

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


def interactive_plot(measure_map, lambdas, delays, 
                     autoscale=True, 
                     figsize=None,  
                     vmin=None, vmax=None,
                     cmap=None,
                     electronvolt=False,
                     logscale=False,
                     mOD=False):
    """
    Plots interactively the Stratus UV measure map. Using two sliders the user can 
    select the wavelenght and the delay to plot.
    
    Parameters
    ----------
    measure_map : numpy.array
        2D array to plot. Each element represents the DT/T or mOD value
        for a specific delay and wavelenght.
    delays : numpy.array
        1D array containing the experimental delays.
    lambdas : numpy.array
        1D array containing the wavelength recorded by the spectrometer.
    autoscale : bool, default: True
        If true the scale of the delay and wavelenght plots is reset is time, 
        if false the scale is set between the minimum and maximum value of the measure map.
    mOD : bool, default: False
        If True plots in mOD, if False in dT/T (percentage).
    figsize : (value, value), optional
        Size of the figure. Refere to matplotlib doc.
    electronvolt : bool, default: False
        If True converts the photon wavelength to eV.
    logscale : bool, default: False
        If True plots in log scale.
    vmax : float, optional
        Maximum 2D value to render. It has to be intended as a "cutoff":
        everything above this value will be threated as `vmax`.
    vmin : float, optional
        Same as `vmax`.
    cmap : string, optional
        Matplotlib cmap. Refer to its documentation.
    
    Warnings
    --------
    This function works correctly only interactively. This is achieved or in standalone programs
    or using the `%notebook` magic cell in Jupyter.
    """

    # Set figure size
    if figsize==None:
        f, ax = plt.subplots(3, 3, gridspec_kw={'height_ratios': [10, 4, 1],
                                                'width_ratios': [1,10,4]})
    else:
        f, ax = plt.subplots(3, 3, figsize=figsize, gridspec_kw={'height_ratios': [10, 4, 1],
                                                                 'width_ratios': [1,10,4]})
    # convert the lambda axis to electronvolt
    if electronvolt:
        h = 6.6261*1E-34
        c = 299792458
        ev = 1.602176565 * 1E-19
        lambdas =  (1E9*h*c/lambdas)
        wavelabel="Probe photon energy (eV)"
    else:
        wavelabel="Probe wavelength (nm)"

    
    value_label = "dT/T %"
    if mOD:
        value_label = "mOD"
        measure_map = -1000*np.log10(measure_map/100 + 1)
    
    # plot measure map
    if logscale:
        pos = ax[0,1].pcolormesh(delays,lambdas, measure_map,  shading="gouraud", vmin=vmin, vmax = vmax, cmap=cmap,  norm=colors.LogNorm())
    else:
        pos = ax[0,1].pcolormesh(delays,lambdas, measure_map,  shading="gouraud", vmin=vmin, vmax = vmax, cmap=cmap)
    
    ax[0,1].set_xlabel("Delay time (fs)")
    ax[0,1].set_ylabel(wavelabel)
    ax[0,0].axis("off")
    ax[1,0].axis("off")
    ax[1,2].axis("off")
    ax[2,0].axis("off")
    ax[2,1].axis("off")
    ax[2,2].axis("off")
    plt.tight_layout()

    # Set the two sliders
    axlambdas = plt.axes([ax[0,0].get_position().x0+ax[0,0].get_position().width/2,
                          ax[0,0].get_position().y0,
                          ax[0,0].get_position().width/2, 
                          ax[0,0].get_position().height]
                        )
    lambdas_slider = Slider(
                            ax=axlambdas,
                            label='Lambdas',
                            valmin=lambdas[0],
                            valmax=lambdas[-1],
                            valinit=lambdas[0],
                            orientation="vertical"
                            )
    axdelay = plt.axes([ax[2,1].get_position().x0,
                        ax[2,1].get_position().y0+ax[2,1].get_position().height/2,
                        ax[2,1].get_position().width, 
                        ax[2,1].get_position().height/1]
                      )
    delays_slider = Slider(
                           ax=axdelay,
                           label='Delays',
                           valmin=delays[0],
                           valmax=delays[-1],
                           valinit=delays[0]
                           )


    max_value = np.max(measure_map)
    min_value = np.min(measure_map)

    line_lambdas, =  ax[1,1].plot(delays, measure_map[0])
    ax[1,1].set_xlabel("Delay time (fs)")
    ax[1,1].set_ylabel(value_label)
    line_delays, = ax[0,2].plot(measure_map[:,0], lambdas)
    ax[0,2].set_ylabel(wavelabel)
    ax[0,2].set_xlabel(value_label)
    if not autoscale:
        ax[1,1].set_ylim(min_value,max_value)
        ax[0,2].set_xlim(min_value,max_value)


    def update_lambdas(val):
        arg = np.argmin(np.abs(lambdas-lambdas_slider.val))
        line_lambdas.set_ydata(measure_map[arg])
        if autoscale:
            ax[1,1].relim()
            ax[1,1].autoscale()
        f.canvas.draw_idle()

    def update_delays(val):
        arg = np.argmin(np.abs(delays-delays_slider.val))
        line_delays.set_xdata(measure_map[:, arg])
        if autoscale:
            ax[0,2].relim()
            ax[0,2].autoscale()
        f.canvas.draw_idle()

    lambdas_slider.on_changed(update_lambdas)
    delays_slider.on_changed(update_delays)
    plt.show()
