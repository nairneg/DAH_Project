import sys
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# import data
#xmass = np.loadtxt(sys.argv[1])
f = open("ups-15-small.bin","r")
datalist = np.fromfile(f,dtype=np.float32)

# number of events
nevent = int(len(datalist)/6)

xdata = np.split(datalist,nevent)
xdata = datalist.reshape(nevent, 6)



def extract_variables(data):
    """Extracts the 6 variable arrays (columns) from event data."""
    list_array = [data[:, i] for i in range(data.shape[1])]
    return list_array


list_array = extract_variables(xdata)

inv_mass = list_array[0]
xpair_trans_mom = list_array[1]
xrapidity = list_array[2]
xpair_mom = list_array[3]
xfirst_mom = list_array[4]
xsecond_mom = list_array[5]



labels = ['Invariant Mass of Muon Pairs', 'Transverse Momentum of Muon Pairs','Rapidity of Muon Pairs', 'Momentum of Muon Pairs', 'Transverse Momentum of First Muon', 'Transverse Momentum of Second Muon' ]
x_labels = ['Invariant Mass (GeV/c^2)', 'Transverse Momentum (GeV/c)', 'Rapidity', 'Momentum (GeV/c)', 'Transverse Momentum (GeV/c)', 'Transverse Momentum (GeV/c)' ]
y_label = 'Number of Events'
bins = [500, 500, 500, 500, 1000, 1000]
x_limits = [[min(inv_mass), max(inv_mass)], [0, 1000], [0, 1000], [0, 200], [0, 200]]


def hist_maker(in_data, label,  x_label, ylabel, bins):

    
    plt.figure(figsize=(8, 5))        # new figure each time
    plt.hist(in_data, bins=bins, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(label)
    plt.xlim(min(in_data), max(in_data))
    plt.tight_layout()
    plt.show()
    

for i in range(6):
    hist_maker(list_array[i], labels[i], x_labels[i], y_label, bins[i])