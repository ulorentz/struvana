"""
Preprocess contains tools for preprocessing raw data from the Stratus UV setup.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from os.path import exists as file_exists
import scipy.signal as signal
from scipy import interpolate
from scipy.interpolate import interp1d
from os.path import isdir
import matplotlib
from . import plot

class RawDataManager:
    """
    Class to handle and preprocess raw data from spectrometer.
    Usually the order to proceede is the following:
    - cut scan;
    - denoise;
    - shift zero;
    - reject scans;
    - dechirp;
    - smooth.
    
    Parameters
    ----------
    file_path : str
        Path of the raw data. It should be the directory containing the ".dat" files followed
        by the name of the run without the "_number" and without the ".dat". Suppose in "dir"
        there is "measure_april.dat" and "measure_april_1.dat" then `file_path` should be 
        "dir/measure_april".
    save_path : str
        Directory to save the results.

    Attributes
    ----------
    scans : array
        Raw scans.
    delays : ndarray
        Delays of the pump-probe. `dechirp` changes the range in the dechirp process.
    lambdas : ndarray
        Wavelength of data. `cut_scans` changes the range. 
    measure_average : ndarray
        Statistical mean of the scans. After the preprocess this is the data to refer to.
    """
    def __init__(self, file_path, save_path):
        self.save_path = save_path
        self.file_path = file_path
        self.run_name = self.file_path.split("/")[-1] #assuming last is run name
        self.scans = []
        self.delays = None
        self.lambdas = None
        self.measure_average = None
        self.measure_stddev = None
        

        if save_path is not None:
            # check if saving path exists
            if not isdir(save_path):
                raise NotADirectoryError("'"+save_path + "' doesn't exists")
            # load data
        self._load()
    
    def _load(self):
        # check if file exists
        n = 0
        exists = file_exists(self.file_path+"_"+str(n)+".dat")
        if not exists:
            print("Measurement '"+self.file_path+"' not found!")
            print("Avoid '_scanNumber' and '.dat'!")
        while exists:
            # load data
            x = pd.read_csv(self.file_path+"_"+str(n)+".dat",  sep="\t", header=None, index_col=0)
            self.scans.append(x.to_numpy()[1:]*100) # first row is lambdas
            # extract delays and lambdas
            if n == 0:
                self.delays = x.iloc[0].to_numpy()
                self.lambdas = x.index.to_numpy()[1:]

            # increase counter and check again
            n+=1
            exists = file_exists(self.file_path+"_"+str(n)+".dat")

    def plot_map(self, title=None, 
                       figsize=None, 
                       maxDelay=None, 
                       delayRange=None, 
                       log=False, 
                       mOD=False, 
                       vmin=None, 
                       vmax=None, 
                       electronvolt=False):
        """
        Plot the average of the scans at the current preprocess step.

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
        if self.measure_average is None:
            self._average()
        
        plot.plot_measure_map(self.measure_average, 
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
    
    def denoise(self, max_delay = 0):
        """
        Removes the average background noise computed on negative delays. 

        Parameters
        --------
        max_delay : int, default: 0
            Computes the average background from start until this delay, default is zero, to
            be modified if signal starts before zero.
        """
        for i in range(len(self.scans)):
            arg = np.argmin(np.abs(self.delays-max_delay))
            self.scans[i]=(self.scans[i]-np.mean(self.scans[i][:, :arg], 1).reshape(-1,1))
        self._average()


    def shift_zero(self, lambda_shift, dry_run = False):
        """
        Aligns the zero delay of all the scans in case some are shifted.
        At the end computes the scans average. 
        
        Parameters
        ----------
        lambda_shift : int
            Wavelength to realign. The algorithm will check for shift of the zero
            for this wavelength in all the scans and if any shift is find will realign
            the scan using this wavelength. Choose a proper one.
        dry_run : bool, default: False
            If true the algorithm will run without modify the data. Can be used to check 
            if the choosen `lambda_shift` is a proper one. 
        Warnings
        --------
        The algorithm modifies the scans and it is irreversible. If not sure about the results 
        run a dry_run before the actual one. The only way to reverse is to load again the raw scans.
        """
        # dry_run : doesn't update the scans if true
        min_lambda = min(self.lambdas)
        max_lambda = max(self.lambdas)
        if lambda_shift < min_lambda or lambda_shift > max_lambda:
            raise ValueError("Lambda shift should be between " + str(min_lambda)
                             + " and " + str(max_lambda))

        nscans = len(self.scans)
        lambda_index = np.argmin(np.abs(self.lambdas-lambda_shift))
        

        plt.figure()
        plt.title("Before smoothing")
        for i in range(nscans):
            plt.plot(self.delays, self.scans[i][ lambda_index])
        plt.xlabel("fs")
    
        plt.figure()
        plt.title("After smoothing")
        smooth_scans = []
        
        # smooth selected lambda scans
        for i in range(nscans):
            smooth_scans.append(self._average_smooth(self.scans[i][lambda_index],10))
            
            plt.plot(self.delays, smooth_scans[i])       
        plt.xlabel("fs")

        # we assume all the scans are equally long
        scan_len = len(smooth_scans[0])
        lags = signal.correlation_lags(scan_len, scan_len, mode="full")
        # shift relative delay
        for i in range(1, nscans):
            max_corr = np.argmax(np.correlate(smooth_scans[0], smooth_scans[i], mode="full"))
            relative_delay = lags[max_corr]
            if not dry_run:
                self.scans[i] = np.roll(self.scans[i], relative_delay, 1)

        
            # plot shifted
        plt.figure()
        plt.title("After shifting")
        for i in range(nscans):
            plt.plot(self.delays, self.scans[i][ lambda_index])
        plt.xlabel("fs")
        if not dry_run:
            self._average()

    def cut_scans(self, lambda1, lambda2):
        """
        Cut the scans keeping data between `lambda1` and `lambda2`. At the end
        computes the scans average.

        Parameters
        ----------
        lambda1 : int
        lambda2 : int
        """
        index1 = np.argmin(np.abs(self.lambdas-lambda1))
        index2 = np.argmin(np.abs(self.lambdas-lambda2))
        self.lambdas = self.lambdas[index1:index2]
        for  i in range(len(self.scans)):
            self.scans[i] = self.scans[i][index1:index2]
        self._average()

    def reject_scans(self, threshold, dry_run = False):
        """"
        Computes the correlations and rejects the ones less correlated than `threshold`.

        Parameters
        ----------
        threshold : float
            Value between 0 and 1 (correlations squeezed between those values). All scans 
            with correlation below threshold will be deleted. Run a `dry_run` if unsure about
            the value.
        dry_run : bool, default: False
            If true the algorithm will run without modify the data. Can be used to check 
            if the choosen `threshold` is a proper one. 

        Warnings
        --------
        The algorithm deletes uncorrelated scans and it is irreversible. If not sure 
        about the results run a dry_run before the actual one. The only way to reverse is to 
        load again the raw scans.
        """
        dotMatrix = np.zeros((len(self.scans)-1, len(self.lambdas)))
        for i in range(1,len(self.scans)):
            for j in range(0, len(self.lambdas)):
                dotMatrix[i-1, j] = np.dot(self.scans[0][j]/np.linalg.norm(self.scans[0][j]), 
                                           self.scans[i][j]/np.linalg.norm(self.scans[i][j]))
        
        # plot correlation matrix
        fig, ax = plt.subplots()
        pos = ax.imshow(dotMatrix.T, 
                        aspect="auto",
                        extent=[0,
                               len(self.scans),
                               self.lambdas[0],
                               self.lambdas[-1]
                               ], 
                        origin="lower",)
        fig.colorbar(pos, ax=ax)
        plt.title("Correlation matrix")
        plt.xlabel("Scan number")
        plt.ylabel("Wavelenght")

        dotdot = np.sum(dotMatrix, axis=1)
        dotdot /= np.max(dotdot)
        # not clear this formula where it comes from
        dotSum = np.exp(- (np.multiply(15,(1 - dotdot) ** 2)) ** 4)
            
        plt.figure()
        plt.plot(np.arange(1,len(self.scans)), dotSum,'o', label="Correlations")
        plt.plot(np.arange(1,len(self.scans)), np.arange(1,len(self.scans))*threshold, label="Threshold")
        plt.ylabel("Correlation")
        plt.legend()
        plt.xlabel("Scan number")

        if not dry_run:
            n_reject = 0
            for i in range(len(self.scans)-1, 0, -1):
                if dotSum[i-1]<threshold:
                    self.scans.pop(i)
                    n_reject += 1
            
            print(str(n_reject)+" scans have been rejected")
            self._average()

    def _average(self):
        self.measure_average = np.mean(self.scans, axis=0)
        self.measure_stddev = np.std(self.scans, axis=0)
    
    @staticmethod
    def _average_smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def smooth(self, smoothWindowDelayFS, smoothWindowLambdaNM, dry_run=False):
        """
        Smooth with a moving window average the scans average in both the delay and lambda axis.
        
        Parameters
        ----------
        smoothWindowDelayFS: int
            Size in fs of the moving window in the delay axis. Should be three times the minimum step 
            of the data taking. 
        smoothWindowLambdaNM: int
            Size in nm of the moving window in the lambda axis. Should be three times the minimum step 
            of the data taking, so usually 3 nm.
        dry_run: bool, default: False
            If true the algorithm will run without modify the `measure_average`. Can be used to choose
            the smoothing parameters. The algorithm modifies only `measure_average` and not the scans.
        """
        # time
        smoothWindowDelay = np.round(smoothWindowDelayFS / (self.delays[1] - self.delays[0]))
        if np.mod(smoothWindowDelay,2) == 0:
            smoothWindowDelay += 1 

        tmp_measure = self.measure_average.copy()
        # TODO is it necessary???
        for i in range(1):
            for j in range(self.measure_average.shape[0]):
                tmp_measure[j] = self._average_smooth(self.measure_average[j], 
                                                         int(smoothWindowDelay))

        
        # lambda
        smoothWindowLambda = np.round(smoothWindowLambdaNM / (self.lambdas[1] - self.lambdas[0]))
        if np.mod(smoothWindowLambda,2) == 0:
            smoothWindowLambda += 1 
        
        
        for i in np.arange(1,4+1).reshape(-1):
            for j in range(self.measure_average.shape[1]):
                tmp_measure[:,j] = self._average_smooth(self.measure_average[:,j], 
                                                           int(smoothWindowLambda))
        if not dry_run:
            self.measure_average = tmp_measure
        plt.figure()
        plt.imshow(self.measure_average, 
                   aspect="auto", 
                   extent=[self.delays[0], 
                           self.delays[-1],
                           self.lambdas[0],
                           self.lambdas[-1]], 
                   origin="lower")
                   #cmap="seismic")
        plt.ylabel("Probe wavelength (nm)")
        plt.xlabel("Delay time (fs)")
        plt.title("Smooth")

    def reset_dechirp(self):
        """
        Computes the average of scans resetting the dechirp process
        """
        self._average()

    def dechirp(self, max_delay, save=True, n_points=8):
        """
        Corrects for group velocity dispersion inside the sample. The method requires to select
        eight points on the artifacts, it will fits a curve on this points and realigns  the fitted curve
        to zero correcting for the GVD.
        In order to be run interactively the method changes matplotlib backend to TkAgg and then 
        restores the original one. If run in a jupyter notebook it will open an external interactive
        window to select the points on the map.
       
        Parameters
        ----------
        max_delay : int
            Maximum delay to plot in the interactive window. Choose a proper one to enlarge the artifact.
        save : bool, default: True
            If True will save the coordinates of the selected points to a numpy array in the `save_path` 
            and the plot with the fitted curve.
        n_points : int, default: 8
            Number of points to be manually selected in the dechirping procedure. 
        """

        idx_max_delay = np.argmin(np.abs(self.delays-max_delay))
        max_delay = self.delays[idx_max_delay]
        # get current backend to be restored later
        backend = matplotlib.get_backend()
        # change backend for interactive plot
        matplotlib.use('TkAgg')
        figsize=(16,14)
        plt.figure(figsize=figsize)
        plt.pcolormesh(self.delays[:idx_max_delay],self.lambdas,  self.measure_average[:, :idx_max_delay], cmap="seismic")
        plt.ylabel("Probe wavelength (nm)")
        plt.xlabel("Delay time (fs)")
        plt.title("Select "+ str(n_points) +" points on the artifact")
        plt.draw()
        pts = np.asarray(plt.ginput(n_points, timeout=-1))
        if save:
            np.save(self.save_path+ "/"+self.run_name+"_dechirp_points", pts)
        
        x_delay = pts[:,0]
        y_wave = pts[:,1]
        # cubic interpolation
        tck = interpolate.splrep(y_wave, x_delay)
        delay_new = interpolate.splev( self.lambdas, tck)

        # restore previous backend
        matplotlib.use(backend)
        plt.figure()
        plt.title("Fitted chirp")
        plt.pcolormesh(self.delays[:idx_max_delay],self.lambdas,  self.measure_average[:, :idx_max_delay], cmap="seismic")
        plt.ylabel("Probe wavelength (nm)")
        plt.xlabel("Delay time (fs)")
        plt.plot( delay_new, self.lambdas, color="y")
        if save:
            plt.savefig(self.save_path+"/"+self.run_name+"_dechirp.pdf")

        for i in range(len(self.lambdas)):
            self.measure_average[i,:] = interp1d(self.delays - delay_new[i], 
                                                self.measure_average[i,:], 
                                                bounds_error=False
                                                )(self.delays )

        # index of not valid shifted points
        idx = np.argwhere(np.isnan(self.measure_average))
        #  delay axis
        idx = idx[:,1]
        
        self.delays = np.delete(self.delays, idx)
        self.measure_average = np.delete(self.measure_average, idx, axis=1)

    def save(self, process_step_name):
        """
        Save processed data to `save_path`. The data is saved in a compressed numpy archive. 
        Refer to numpy doc (`numpy.savez`). 
        The archive is saved inside the `save_path` with the same name of the raw data originally read
        and with "_'process_step_name'" added ad the end.
        
        Parameters
        ----------
        process_step_name: str
            Name of the preprocessing step, will be added at the end of the filename.
        """
        filename = self.save_path + "/" + self.run_name + "_" + process_step_name
        np.savez(filename, 
                measure_map = self.measure_average,
                delays = self.delays,
                lambdas = self.lambdas
                )


