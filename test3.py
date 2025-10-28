import sys
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# import data
#xmass = np.loadtxt(sys.argv[1])
f = open("C:/Users/K_muk/OneDrive - University of Edinburgh/Physics Year 4/Data Acquisition & Handling/DAH Project/ups-15-small.bin","r")
datalist = np.fromfile(f,dtype=np.float32)

# number of events
nevent = int(len(datalist)/6)

xdata = np.split(datalist,nevent)
print(xdata[0])


# make a finction to extract varialbles from xdata
def extract_variables(input_data):

    # make list of invariant mass of events
    xmass = []
    for i in range(0,nevent):
        xmass.append(xdata[i][0])
    #    if i < 10:
    #        print(xmass[i])

    # make a list of the transverse momentum of muon pairs
    xpair_trans_mom = []
    for i in range(0,nevent):
        xpair_trans_mom.append(xdata[i][1])

    # make a list of rapidity of muon pair
    xrapidity = []
    for i in range(0,nevent):
        xrapidity.append(xdata[i][2])

    # make a list of momentum of muon mairs
    xpair_mom = []
    for i in range(0,nevent):
        xpair_mom.append(xdata[i][3])

    # make a list of the transverse momentum of first muon
    xfirst_mom = []
    for i in range(0,nevent):
        xfirst_mom.append(xdata[i][4])

    # make a list of the transverse momentum of second muon
    xsecond_mom = []
    for i in range(0,nevent):
        xsecond_mom.append(xdata[i][5])

    return xmass, xpair_trans_mom, xrapidity, xpair_mom, xfirst_mom, xsecond_mom

xmass, xpair_trans_mom, xrapidity, xpair_mom, xfirst_mom, xsecond_mom = extract_variables(xdata)

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



Ilower_bound = 9.3
Iupper_bound = 9.6


inv_mass_regionI = [mass for mass in xmass if Ilower_bound <= mass <= Iupper_bound]


entries, bedges, ps = plt.hist(inv_mass_regionI, bins=100, histtype='step', label='Invariant Mass of Muon Pairs in Region I', color='cyan')
plt.show()

def gauss_exp(x, A, mu, sigma, B, C, D):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2)) + B - C * np.exp(-D * x)

parameters, pcov = curve_fit(gauss_exp, bedges, entries)


print(parameters)