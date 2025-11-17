import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def file_read(f_path):
    """
    Reads in binary data from file and transforms it into a 2D numpy array. With columns representing different observables.
    """
    f_open = open(f_path,"r") 
    datalist = np.fromfile(f_open, dtype=np.float32)
    
    nevent = int(len(datalist) / 6)
    xdata = datalist.reshape(nevent, 6)
    return xdata



xdata = file_read('/Users/nairnegillespie/Desktop/Year 4/DAH Project/mc.bin')

def extract_variables(data):
    """
    Extracts the 6 observable arrays (columns) from 2D array.
    """
    return [data[:, i] for i in range(data.shape[1])]       

xmass, xpt, rap, p, p1, p2 = extract_variables(xdata)


# === replace these with your histogram arrays ===
# counts: 1D array of histogram counts
# bin_edges: histogram bin edges (len = len(counts)+1)
# Example if you have raw data:
# counts, bin_edges = np.histogram(data, bins=..., range=...)

counts, bin_edges = np.histogram(xmass, bins=500, range=(9.0, 11.0))

# For this snippet assume you already have counts & bin_edges:
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
x = bin_centers
y = counts
# uncertainties: sqrt(N) ; avoid zero by using 1 for empty bins or small floor
sigma = np.sqrt(y)
sigma[sigma == 0] = 1.0

# Model: sum of two gaussians + exponential background
def model(x, A1, mu1, s1, A2, mu2, s2, C, lam):
    g1 = A1 * np.exp(-0.5*((x-mu1)/s1)**2)
    g2 = A2 * np.exp(-0.5*((x-mu2)/s2)**2)
    b  = C  * np.exp(-lam * x)
    return g1 + g2 + b

# Initial guesses (important!)
# Choose guesses roughly from plot: peak heights, peak positions, widths
p0 = [
    max(y),       # A1
    x[np.argmax(y)],  # mu1 -- strongest peak
    (x[-1]-x[0]) / 20.0,  # s1
    max(y)/3.0,   # A2
    x[np.argmax(y)] + 0.5*(x[-1]-x[0])/10.0,  # mu2 (nearby)
    (x[-1]-x[0]) / 30.0,  # s2
    min(y)/2.0 + 1.0,    # C (background scale)
    1.0/(x[-1]-x[0])     # lam
]

# Optional bounds: keep widths positive, amplitudes >=0, lam >= 0
lower = [0, x[0], 1e-6, 0, x[0], 1e-6, 0, 0]
upper = [np.inf, x[-1], np.inf, np.inf, x[-1], np.inf, np.inf, np.inf]

popt, pcov = curve_fit(model, x, y, p0=p0, sigma=sigma, absolute_sigma=True, bounds=(lower, upper))

perr = np.sqrt(np.diag(pcov))
print("Best-fit parameters (curve_fit):")
names = ['A1','mu1','s1','A2','mu2','s2','C','lam']
for n,v,e in zip(names, popt, perr):
    print(f" {n} = {v:.4g} ± {e:.4g}")

# Goodness-of-fit
resid = (y - model(x, *popt))
chi2 = np.sum((resid/sigma)**2)
ndof = len(y) - len(popt)
print(f"chi2/ndof = {chi2:.2f}/{ndof} = {chi2/ndof:.2f}")

# Plot
plt.figure(figsize=(7,4))
plt.plot(x, y, label='data')
plt.plot(x, model(x, *popt), label='fit')
#plt.plot(xs, popt[0]*np.exp(-0.5*((xs-popt[1])/popt[2])**2), ls='--', label='G1')
#plt.plot(xs, popt[3]*np.exp(-0.5*((xs-popt[4])/popt[5])**2), ls='--', label='G2')
#plt.plot(xs, popt[6]*np.exp(-popt[7]*xs), ls=':', label='background')
plt.legend()
plt.xlabel('x')
plt.ylabel('counts')
plt.tight_layout()
plt.show()

"""residuals = y - model(x, *popt)"""
res = y - model(x, *popt)
plt.figure(figsize=(7,4))
plt.plot(x, res, marker='o', ls='', label='residuals')
plt.axhline(0, color='gray', ls='--')
plt.xlabel('invariant mass')
plt.ylabel('residuals')
plt.tight_layout()
plt.show()