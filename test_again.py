import sys
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import integrate


# import data
#xmass = np.loadtxt(sys.argv[1])
f = open("ups-15-small.bin","r")
datalist = np.fromfile(f,dtype=np.float32)

# number of events
nevent = int(len(datalist)/6)

xdata = np.split(datalist,nevent)
print(xdata[0])


xdata = datalist.reshape(nevent, 6)

# extract variable arrays

def extract_variables(data):
    """Extracts the 6 variable arrays (columns) from event data."""
    list_array = [data[:, i] for i in range(data.shape[1])]
    return list_array


list_array = extract_variables(xdata)

xmass = list_array[0]
xpair_trans_mom = list_array[1]
xrapidity = list_array[2]
xpair_mom = list_array[3]
xfirst_mom = list_array[4]
xsecond_mom = list_array[5]
'''
# plot histograms of variables
plt.hist(xmass, bins=500, histtype='step', label='Invariant Mass of Muon Pairs')
plt.xlabel('Invariant Mass (GeV/c^2)')
plt.ylabel('Number of Events')
plt.xlim(min(xmass), max(xmass))
plt.title('Histogram of Invariant Mass of Muon Pairs')
plt.savefig('invariant_mass_histogram.svg', format='svg')
plt.show()

plt.hist(xpair_trans_mom, bins=500, histtype='step', label='Transverse Momentum of Muon Pairs', color='orange')
plt.xlabel('Transverse Momentum (GeV/c)')
plt.ylabel('Number of Events')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.title('Histogram of Transverse Momentum of Muon Pairs')
plt.savefig('transverse_momentum_histogram.svg', format='svg')
plt.show()

plt.hist(xrapidity, bins=500, histtype='step', label='Rapidity of Muon Pairs', color='green')
plt.xlabel('Rapidity')
plt.ylabel('Number of Events')
plt.xlim(min(xrapidity), max(xrapidity))
plt.title('Histogram of Rapidity of Muon Pairs')
plt.savefig('rapidity_histogram.svg', format='svg')
plt.show()

plt.hist(xpair_mom, bins=500, histtype='step', label='Momentum of Muon Pairs', color='red')
plt.xlabel('Momentum (GeV/c)')
plt.ylabel('Number of Events')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.title('Histogram of Momentum of Muon Pairs')   
plt.savefig('momentum_histogram.svg', format='svg')
plt.show()

plt.hist(xfirst_mom, bins=500, histtype='step', label='Transverse Momentum of First Muon', color='purple')
plt.xlabel('Transverse Momentum (GeV/c)')
plt.ylabel('Number of Events')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.title('Histogram of Transverse Momentum of First Muon')
plt.savefig('first_muon_transverse_momentum_histogram.svg', format='svg')
plt.show()

plt.hist(xsecond_mom, bins=500, histtype='step', label='Transverse Momentum of Second Muon', color='brown')
plt.xlabel('Transverse Momentum (GeV/c)')
plt.ylabel('Number of Events')
plt.title('Histogram of Transverse Momentum of Second Muon')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.savefig('second_muon_transverse_momentum_histogram.svg', format='svg')
plt.show()
'''


Ilower_bound = 9.3
Iupper_bound = 9.6


inv_mass_regionI = np.array([mass for mass in xmass if Ilower_bound <= mass <= Iupper_bound])


entries, bedges, ps = plt.hist(inv_mass_regionI, bins=100, histtype='step', label='Invariant Mass of Muon Pairs in Region I', color='cyan')
plt.show()


def gauss(x,  mu, sigma):
    return  np.exp(- (x - mu)**2 / (2 * sigma**2))

def exp(x):
    return np.exp(-x)

integral_A, errA = integrate.quad(gauss, Ilower_bound, Iupper_bound, args=[mu, sigma])
NA = 1/ integral_A

integral_B, errB = integrate.quad(exp, Ilower_bound, Iupper_bound)
NB = 1/ integral_B  

def normalized_gauss(x, NA, mu, sigma):
    return NA * np.exp(- (x - mu)**2 / (2 * sigma**2))     

def normalized_exp(x, NB):
    return NB * np.exp(-x)


def comp_model(x, NA, mu, sigma, NB):
    normal_gauss = normalized_gauss(x, NA, mu, sigma) + normalized_exp(x, NB)
    normal_exp = normalized_exp(x, NB)
    
    return normal_gauss + normal_exp

p0 = [max(entries), bin_centers[np.argmax(entries)], 0.1, 4000000, 0.01]  # A, mu, sigma, B, C  




