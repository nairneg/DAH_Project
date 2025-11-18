import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import crystalball
import minuit as mn

def file_read(f_path):
    """
    Reads in binary data from file and transforms it into a 2D numpy array. With columns representing different observables.
    """
    f_open = open(f_path,"r") 
    datalist = np.fromfile(f_open, dtype=np.float32)
    
    nevent = int(len(datalist) / 6)
    xdata = datalist.reshape(nevent, 6)
    return xdata

xdata = file_read('C:/Users/K_muk/OneDrive - University of Edinburgh/Physics Year 4/Data Acquisition & Handling/DAH Project/mc.bin')
xdata_LHCB = file_read('C:/Users/K_muk/OneDrive - University of Edinburgh/Physics Year 4/Data Acquisition & Handling/DAH Project/ups-15-small.bin')

def extract_variables(data):
    """
    Extracts the 6 observable arrays (columns) from 2D array.
    """
    return [data[:, i] for i in range(data.shape[1])]       

xmass, xpt, rap, p, p1, p2 = extract_variables(xdata)
xmass_LHCB, xpt_LHCB, rap_LHCB, p_LHCB, p1_LHCB, p2_LHCB = extract_variables(xdata_LHCB)


# === replace these with your histogram arrays ===
# counts: 1D array of histogram counts
# bin_edges: histogram bin edges (len = len(counts)+1)
# Example if you have raw data:
# counts, bin_edges = np.histogram(data, bins=..., range=...)

counts, bin_edges = np.histogram(xmass, bins=500, range=(9.0, 11.0))
counts_LHCB, bin_edges_LHCB = np.histogram(xmass_LHCB, bins=500, range=(9.0, 11.0))


bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
x = bin_centers
y = counts


bin_centres_LHCB = 0.5*(bin_edges_LHCB[:-1] + bin_edges_LHCB[1:])
x_LHCB = bin_centres_LHCB
y_LHCB = counts_LHCB



# uncertainties: sqrt(N) ; avoid zero by using 1 for empty bins or small floor
sigma = np.sqrt(y)
sigma[sigma == 0] = 1.0
"""

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


def fitting_to_LHCb_data(x_LHCB, params_found_for_mc):
    y_fit = model(x_LHCB, *params_found_for_mc)
    y_LHCB_own_params = curve_fit(model, x_LHCB, y_LHCB, p0=params_found_for_mc)[0]
    yfit_with_own_params = model(x_LHCB, *y_LHCB_own_params)
    
    return y_fit, y_LHCB_own_params, yfit_with_own_params
    

yfit_with_mc_params, y_LHCB_own_params, yfit_with_own_params = fitting_to_LHCb_data(x_LHCB, popt)



#plt.plot(x_LHCB, y_LHCB, label='LHCb Data', marker='o', linestyle='', markersize=3)
plt.plot(x_LHCB, yfit_with_mc_params, label='Fit to LHCb Data using MC Parameters', color='red')
plt.plot(x_LHCB, yfit_with_own_params, label='Fit to LHCb Data using Own Parameters', color='green')
plt.hist(xmass_LHCB, bins=500, range=(9.0, 11.0), alpha=0.4, label='LHCb data histogram')
plt.xlim(9.3, 9.6)
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Counts')
plt.title('Fit to LHCb Invariant Mass Distribution using MC Parameters')    
plt.legend()
plt.show()

def not_a_clue(x_LHCB):
    some_fixed_params_Popt, some_fixed_params_pcov =curve_fit(model, x_LHCB, y_LHCB, p0=[popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], 500, popt[7]])
    yy_lhcb_data_some_fixed = model(x_LHCB, *some_fixed_params_Popt)
    return yy_lhcb_data_some_fixed, some_fixed_params_Popt, some_fixed_params_pcov
                                    
                                    

yy_lhcb_data_some_fixed, some_fixed_params_Popt, some_fixed_params_pcov = not_a_clue(x_LHCB)
plt.plot(x_LHCB, yy_lhcb_data_some_fixed, label='Fit to LHCb Data with some fixed params', color='purple')
plt.hist(xmass_LHCB, bins=500, range=(9.0, 11.0), alpha=0.4, label='LHCb data histogram')
plt.legend()
plt.xlim(9.3, 9.6)
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Counts')
plt.title('Fit to LHCb Invariant Mass Distribution with some fixed params')
plt.show()



some_fixed_perr = np.sqrt(np.diag(some_fixed_params_pcov))
print("Best-fit parameters (curve_fit):")
names = ['A1_sf','mu1_sf','s1_sf','A2_sf','mu2_sf','s2_sf','C_sf','lam_sf']
for n,v,e in zip(names, some_fixed_params_Popt, some_fixed_perr):
    print(f" {n} = {v:.4g} ± {e:.4g}")

# Goodness-of-fit
resid_some_fixed = (y_LHCB - model(x_LHCB, *some_fixed_params_Popt))
chi2_some_fixed = np.sum((resid_some_fixed/sigma)**2)
ndof = len(y_LHCB) - len(some_fixed_params_Popt)
print(f"chi2/ndof = {chi2_some_fixed:.2f}/{ndof} = {chi2_some_fixed/ndof:.2f}")

plt.figure(figsize=(7,4))
plt.plot(x_LHCB, resid_some_fixed, marker='o', ls='', label='residuals')
plt.axhline(0, color='gray', ls='--')
plt.xlabel('invariant mass')
plt.ylabel('residuals')
plt.title('Residuals for LHCb fit with some fixed params')
plt.xlim(9.3, 9.6)
plt.tight_layout()
plt.show()

"""
'''
def crystal_ball_man(x, alpha, n, mu, sigma):
    x = np.asarray(x)
    t = (x - mu) / sigma
    a = abs(alpha)

    A = (n/a)**n * np.exp(-0.5 * a**2)
    B = (n/a) - a

    y = np.zeros_like(x)
    mask = t > -a

    y[mask] = np.exp(-0.5 * t[mask]**2)
    y[~mask] = A * (B - t[~mask])**(-n)

    return y
# Fit using curve_fit
'''

def crystal_ball_with_background(x, mu, sigma, C, lam):

    cb = crystalball.pdf(x, beta=2, m=1, loc=mu, scale=sigma)
    bg = C * np.exp(-lam * x)

    return cb + bg

def scipy_crystal_ball_fit(x, y, p0, sigma):
    popt, pcov = curve_fit(crystal_ball_with_background, x, y, p0=p0, sigma=sigma, absolute_sigma=True)
    return popt, pcov

p0_cb = [9.46, 0.035, 1.0, 0.1]  # mu, sigma, C, lam

pcb, pcov_cb = scipy_crystal_ball_fit(x, y, p0_cb, sigma)

print("\nBest-fit parameters (Crystal Ball - scipy function):")
names_scb = ['mu','sigma','C','lam']
for n,v,e in zip(names_scb, pcb, np.sqrt(np.diag(pcov_cb))):
    print(f" {n} = {v:.4g} ± {e:.4g}")

yfit_cb = crystal_ball_with_background(x, *pcb)

def background(x, C, lam):
    return C * np.exp(-lam * x)

yfit_bg = background(x, pcb[2], pcb[3])

plt.figure(figsize=(10,6))
plt.hist(xmass, bins=500, range=(9.0, 11.0), color='cyan', alpha=0.5, label='MC Data')
plt.plot(x, yfit_cb, 'r-', linewidth=2, label='Crystal Ball + Exp Fit (scipy)')
plt.plot(x, yfit_bg, 'k--', linewidth=2, label='Background Component')
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Counts')
plt.title('Fit to Invariant Mass Distribution')
plt.legend()
plt.show()

"""
p_man, pcov_man = curve_fit(crystal_ball_man, x, y, p0=p0_cb, sigma=sigma, absolute_sigma=True)
print(p_man, pcov_man, np.sqrt(np.diag(pcov_man)))

plt.plot(x, crystalball.pdf(x, *pcb), 'b--', label='Scipy Crystal Ball Fit')
#plt.hist(xmass, bins=500, range=(9.0, 11.0), color='cyan', alpha=0.5, label='MC Data')
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Counts')
plt.title('Fit to Invariant Mass Distribution')
plt.legend()
plt.show()      
    


p0_cb = [1.5, 5.0, 9.46, 0.035]  # alpha, n, mu, sigma
popt_cb, pcov_cb = curve_fit(crystal_ball, x, y, p0=p0_cb, sigma=sigma, absolute_sigma=True)
perr_cb = np.sqrt(np.diag(pcov_cb))
print("\nBest-fit parameters (Crystal Ball):")
names_cb = ['alpha','n','mu','sigma']
for n,v,e in zip(names_cb, popt_cb, perr_cb):
    print(f" {n} = {v:.4g} ± {e:.4g}")  




p0_scb = [1.5, 5.0, 9.46, 0.035]  # alpha, n, mu, sigma
popt_scb, pcov_scb = scipy_crystal_ball_fit(x, y, p0_scb, sigma)
perr_scb = np.sqrt(np.diag(pcov_scb))
print("\nBest-fit parameters (Crystal Ball - scipy function):")
names_scb = ['alpha','n','mu','sigma']
for n,v,e in zip(names_scb, popt_scb, perr_scb):
    print(f" {n} = {v:.4g} ± {e:.4g}") 
"""
    
plt.figure(figsize=(10,6))
plt.hist(xmass, bins=500, range=(9.0, 11.0), color='cyan', alpha=0.5, label='MC Data')
#plt.plot(x, model(x, *popt), 'r-', linewidth=2, label='Two Gaussians + Exp Fit')
plt.plot(x, 'g--', linewidth=2, label='Crystal Ball Fit')
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Counts')
plt.legend()
plt.title('Fit to Invariant Mass Distribution')
plt.show()