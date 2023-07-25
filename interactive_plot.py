from struvana.plot import interactive_plot
from struvana.preprocess import RawDataManager
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser(description='Interactive plot transient absorption map')
arg = parser.add_argument
arg('--fixed_scale', action='store_true' , help="Keep a fixed scale in plots, elsewhere autoscale each update")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--npz_path", type=str,  help="Path to processed numpy archive")
group.add_argument("--raw_path", type=str, help="Path to raw data")
arg("--ev", action="store_true", help="Plot using electronvolt unit")
arg("--log", action="store_true", help="Plot the measure map in log scale")
arg("--mod", action="store_true", help="Plot value in mOD and not in DT/T" )
args = parser.parse_args()

if args.npz_path:
    x = np.load(args.npz_path)
    measure_map = x["measure_map"]
    delays = x["delays"]
    lambdas = x["lambdas"]

else:
    data = RawDataManager(file_path = args.raw_path, save_path = None)
    data._average()
    measure_map = data.measure_average
    delays = data.delays
    lambdas = data.lambdas


interactive_plot(measure_map, 
                 lambdas, 
                 delays, 
                 autoscale= not args.fixed_scale, 
                 figsize=None,  
                 vmin=None, vmax=None,
                 cmap=None,
                 electronvolt=False,
                 logscale=False,
                 mOD=False)
#plt.show()
