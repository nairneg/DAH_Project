from matplotlib.pylab import beta, norm
from networkx import sigma
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line
import scipy
from scipy.optimize import curve_fit
from scipy import integrate
#from scipy.optimize import minimize
import iminuit
from iminuit import minimize  # has same interface as scipy.optimize.minimize
from iminuit import Minuit, describe
from iminuit.cost import LeastSquares
from iminuit.typing import Annotated, Gt
from scipy.stats import crystalball
from scipy.special import erf

#print("iminuit version:", iminuit.__version__)
# import data
#xmass = np.loadtxt(sys.argv[1])
f = open('/Users/nairnegillespie/Desktop/Year 4/DAH Project/mc.bin',"r") #using mc data that only has upsilon (1S) peak
datalist = np.fromfile(f,dtype=np.float32)
# ---------------------------
MC_FILE   = "/Users/nairnegillespie/Desktop/Year 4/DAH Project/mc.bin"      # MC with only 1S
DATA_FILE = "/Users/nairnegillespie/Desktop/Year 4/DAH Project/ups-15-small.bin"  # real data

# ---------------------------
# Load helper to read file (6 floats per event)
# ---------------------------
def load_masses(path):
    with open(path, "rb") as f:
        arr = np.fromfile(f, dtype=np.float32)
    nevent = int(len(arr) / 6)
    data = arr.reshape(nevent, 6)
    return data[:, 0]   # invariant mass column

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

order = np.argsort(xmass) #order the xmass array
xmass = xmass[order]

entries, bedges, _ = plt.hist(xmass, bins=500, color='cyan', label='Data')   
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Entries per bin")
plt.title("Υ(1S) region (mc data)")
plt.legend()
plt.show()


def crystal_ball_function(x, alpha, n, xbar, sigma):
    
    A = (n / abs(alpha))**n * np.exp(-0.5 * abs(alpha)**2)
    B = n / abs(alpha) - abs(alpha)
    C = n/(abs(alpha)*(n - 1)) * np.exp(-0.5 * abs(alpha)**2)
    D = np.sqrt(np.pi/2) * (1 + erf(abs(alpha)/np.sqrt(2)))
    
    N = 1 / (sigma * (C + D))
    
    mask = (x - xbar) / sigma > -alpha
    gaussian_core = N * np.exp(-0.5 * ((x[mask] - xbar)/sigma)**2)
    power_law_tail = N * A * (B - (x[~mask] - xbar)/sigma)**(-n)
    
    YY = np.zeros_like(x)
    YY[mask] = gaussian_core
    YY[~mask] = power_law_tail
    
    #YY = np.maximum(YY, 1e-300)
    
    return YY, mask
    

import numpy as np
from scipy.optimize import minimize

def neg_log_likelihood(params, data, eps=1e-300):
    # we will optimize over unconstrained params using transforms if desired,
    # but here params = [alpha, n, xbar, log_sigma]
    alpha, n, xbar, log_sigma = params
    sigma = np.exp(log_sigma)
    pdf_vals = crystal_ball_function(data, alpha, n, xbar, sigma)
    # protect against zeros
    pdf_vals = np.maximum(pdf_vals, eps)
    return -np.sum(np.log(pdf_vals))

# example usage
data = np.array(xmass)  # replace with your data array

# initial guesses: alpha, n, xbar, log_sigma
p0 = [1.0, 3.0, np.mean(data), np.log(np.std(data))]

# bounds: alpha unconstrained, n>1, xbar unconstrained, log_sigma any real
# implement n>1 by using a penalty or transform; here use bounds in minimize:
bnds = [(None, None), (1.0001, None), (None, None), (None, None)]

res = minimize(neg_log_likelihood, p0, args=(data,), bounds=bnds, method='L-BFGS-B')

if res.success:
    alpha_hat, n_hat, xbar_hat, log_sigma_hat = res.x
    sigma_hat = np.exp(log_sigma_hat)
    print("Fit success:", alpha_hat, n_hat, xbar_hat, sigma_hat)
else:
    print("Fit failed:", res.message)














"""      

def crystal_ball_trunc(x, alpha, n, xbar, sigma, a, b):

    # PDF values (raw normalized to full line)
    vals = crystal_ball_function(x, alpha, n, xbar, sigma) [0]

    # compute normalization over [a,b] by integrating the raw CB function
    # integrate.quad expects scalar function; wrap raw cb for scalar input
def cb_scalar(t):
    return crystal_ball_function(np.array([t]), alpha, n, xbar, sigma)[0]

I, _ = integrate.quad(cb_scalar, a, b, limit=200)
if I <= 0:
        # numerical safeguard
        return np.full_like(vals, 1e-300)
    return vals / I


def exp_trunc(x, lam, a, b):
    lam = float(lam)
    if lam <= 0:
        return np.full_like(x, 1.0 / (b - a))
    denom = 1.0 - np.exp(-lam * (b - a))
    return (lam * np.exp(-lam * (x - a))) / denom

# ---------------------------
def nll_cb_mc(f_s, alpha, n, mu, sigma, lam, data, a, b):
    # guard
    if f_s <= 0 or f_s >= 1 or sigma <= 0 or n <= 1 or lam <= 0:
        return 1e12

    # compute signal pdf values normalized on [a,b]
    sig = crystal_ball_trunc(data, alpha, n, mu, sigma, a, b)
    bkg = exp_trunc(data, lam, a, b)
    pdf_vals = f_s * sig + (1 - f_s) * bkg
    if np.any(pdf_vals <= 0):
        return 1e12
    return -np.sum(np.log(pdf_vals))

mc = load_masses(MC_FILE)
# choose a tight window around 1S for MC
a_mc, b_mc = 9.3, 9.6
mc_sel = mc[(mc >= a_mc) & (mc <= b_mc)]
print("MC events in window", len(mc_sel))

# initial guesses
init_mc = dict(f_s=0.95, alpha=1.0, n=3.0, mu=9.46, sigma=0.04, lam=0.5)

# wrap nll for Minuit (Minuit likes named params)
def nll_mc_wrap(f_s, alpha, n, mu, sigma, lam):
    return nll_cb_mc(f_s, alpha, n, mu, sigma, lam, mc_sel, a_mc, b_mc)

m_mc = Minuit(nll_mc_wrap, **init_mc)
m_mc.errordef = Minuit.LIKELIHOOD

# reasonable limits
m_mc.limits["f_s"] = (1e-3, 0.999)
m_mc.limits["alpha"] = (0.1, 5.0)
m_mc.limits["n"] = (1.1, 50.0)
m_mc.limits["mu"] = (9.4, 9.5)
m_mc.limits["sigma"] = (1e-3, 0.2)
m_mc.limits["lam"] = (1e-3, 5.0)
m_mc.strategy = 1

m_mc.migrad()
m_mc.hesse()

print("\n--- MC fit results (Crystal Ball) ---")
for p in m_mc.parameters:
    print(f"{p:6s} = {m_mc.values[p]:12.6f} ± {m_mc.errors[p]:.6f}")

# Save MC shape params
alpha_mc = m_mc.values["alpha"]
n_mc = m_mc.values["n"]
alpha_mc_err = m_mc.errors["alpha"]
n_mc_err = m_mc.errors["n"]

# ---------------------------
# Plot MC fit (binned for display)
# ---------------------------
nbins = 150
hist_vals, edges = np.histogram(mc_sel, bins=nbins, range=(a_mc, b_mc))
centres = 0.5 * (edges[:-1] + edges[1:])
width = edges[1] - edges[0]

# model (signal fraction from MC fit)
fs_mc = m_mc.values["f_s"]
sig_vals = crystal_ball_trunc(centres, alpha_mc, n_mc, m_mc.values["mu"], m_mc.values["sigma"], a_mc, b_mc)
bkg_vals = exp_trunc(centres, m_mc.values["lam"], a_mc, b_mc)
model_counts = (fs_mc * sig_vals + (1 - fs_mc) * bkg_vals) * len(mc_sel) * width

plt.figure(figsize=(8,5))
plt.errorbar(centres, hist_vals, yerr=np.sqrt(hist_vals), fmt='o', ms=4, label="MC (binned)")
xx = np.linspace(a_mc, b_mc, 1000)
plt.plot(xx, (fs_mc * crystal_ball_trunc(xx, alpha_mc, n_mc, m_mc.values["mu"], m_mc.values["sigma"], a_mc, b_mc) + 
              (1-fs_mc) * exp_trunc(xx, m_mc.values["lam"], a_mc, b_mc)) * len(mc_sel) * width,
         'r-', label="CB+exp fit")
plt.xlabel("Invariant mass (GeV)")
plt.ylabel("Counts")
plt.legend()
plt.title("MC fit: Crystal Ball + exponential")
plt.show()

# ---------------------------
# Fit DATA now using MC-informed CB shape
# Options:
#  - Fix alpha,n to MC values (recommended)
#  - Or fix only alpha and float n slightly, or float both with tight priors (not implemented here)
# ---------------------------
data = load_masses(DATA_FILE)
a_data, b_data = 9.3, 10.7
data_sel = data[(data >= a_data) & (data <= b_data)]
print("Data events in window", len(data_sel))

# Use fraction-based nll for data but with alpha,n fixed
def nll_data_wrap(f_s, mu, sigma, lam, alpha_fix=alpha_mc, n_fix=n_mc):
    # Note: Minuit will pass f_s, mu, sigma, lam
    return nll_cb_mc(f_s, alpha_fix, n_fix, mu, sigma, lam, data_sel, a_data, b_data)

# initial guesses for data
init_data = dict(f_s=0.15, mu=9.456, sigma=0.042, lam=0.6)

m_data = Minuit(nll_data_wrap, **init_data)
m_data.errordef = Minuit.LIKELIHOOD

m_data.limits["f_s"] = (1e-6, 0.999)
m_data.limits["mu"] = (9.42, 9.49)
m_data.limits["sigma"] = (1e-3, 0.2)
m_data.limits["lam"] = (1e-3, 5.0)
m_data.strategy = 1

m_data.migrad()
m_data.hesse()

print("\n--- Data fit results (alpha,n fixed to MC) ---")
print(f"alpha (fixed) = {alpha_mc:.6f} ± {alpha_mc_err:.6f}")
print(f"n     (fixed) = {n_mc:.6f} ± {n_mc_err:.6f}")
for p in m_data.parameters:
    print(f"{p:6s} = {m_data.values[p]:12.6f} ± {m_data.errors[p]:.6f}")

# Convert f_s -> yield
f_s_data = m_data.values["f_s"]
N_data = len(data_sel)
N_signal = f_s_data * N_data
N_signal_err = m_data.errors["f_s"] * N_data
print(f"\nData signal yield: {N_signal:.1f} ± {N_signal_err:.1f} (events)")

# ---------------------------
# Plot data fit & residuals
# ---------------------------
# top panel
nbins = 500
hist_vals, edges = np.histogram(data_sel, bins=nbins, range=(a_data, b_data))
centres = 0.5*(edges[:-1] + edges[1:])
width = edges[1] - edges[0]

# model for plotting
sig_plot = crystal_ball_trunc(centres, alpha_mc, n_mc, m_data.values["mu"], m_data.values["sigma"], a_data, b_data)
bkg_plot = exp_trunc(centres, m_data.values["lam"], a_data, b_data)
model_counts = (m_data.values["f_s"] * sig_plot + (1 - m_data.values["f_s"]) * bkg_plot) * len(data_sel) * width

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,8), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
ax1.errorbar(centres, hist_vals, yerr=np.sqrt(hist_vals), fmt='o', ms=4, label="Data")
xx = np.linspace(a_data, b_data, 2000)
ax1.plot(xx, (m_data.values["f_s"]*crystal_ball_trunc(xx, alpha_mc, n_mc, m_data.values["mu"], m_data.values["sigma"], a_data, b_data) +
              (1-m_data.values["f_s"]) * exp_trunc(xx, m_data.values["lam"], a_data, b_data)) * len(data_sel) * width,
         'r-', lw=2, label="Fit (CB fixed from MC)")
ax1.set_ylabel("Counts")
ax1.legend()

# residuals/pulls
resid = hist_vals - model_counts
data_err = np.sqrt(hist_vals)
pulls = resid / (data_err + 1e-9)
ax2.axhline(0, color='k', lw=1)
ax2.plot(centres, pulls, 'o', ms=3)
ax2.set_ylabel("Pull")
ax2.set_xlabel("Invariant mass (GeV)")
ax2.set_ylim(-5,5)
plt.tight_layout()
plt.show()



Ilower_bound, Iupper_bound = 9.3, 9.6
inv_mass_regionI = np.array([m for m in xmass if Ilower_bound <= m <= Iupper_bound])

entries, bedges, _ = plt.hist(inv_mass_regionI, bins=100, color='cyan', label='Data')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Entries per bin")
plt.xlim(9.3, 9.6)  
plt.title("Υ(1S) region (data)")
plt.legend()
plt.show()

bin_centers = 0.5 * (bedges[:-1] + bedges[1:])


def normalized_gauss(x, mu, sigma, a, b):

    integral, _ = integrate.quad(lambda xx: np.exp(-0.5*((xx - mu)/sigma)**2), a, b)
    N = 1.0 /integral
    return N * np.exp(-0.5 * ((x - mu)/sigma)**2)

def normalized_exp(x, lamb, a, b):

    if abs(lamb) < 1e-12: 
        return np.full_like(x, 1/ (b - a))
    integral, _ = integrate.quad(lambda xx: np.exp(-lamb * xx), a, b)
    norm =  np.exp(-lamb*x) / integral
    return norm

# --- Composite model ---
def comp_model(x, mu, sigma, lamb, f_s, a, b):
    
    return f_s * normalized_gauss(x, mu, sigma, a, b) + (1 - f_s) * normalized_exp(x, lamb, a, b)

# --- Fit model to histogram (binned likelihood via curve_fit) ---
# convert counts to densities
ydata1 = entries / np.trapz(entries, bin_centers)

# initial guesses: μ, σ, λ, f_s
p0 = [9.46, 0.05, 2.0, 0.5]
bounds = ([9.3, 0.01, 0.1, 0.0], [9.6, 0.2, 10.0, 1.0])

parameters, pcov = curve_fit(lambda x, mu, sigma, lamb, f_s: comp_model(x, mu, sigma, lamb, f_s, Ilower_bound, Iupper_bound), bin_centers, ydata1, p0=p0, bounds=bounds)
errors = np.sqrt(np.diag(pcov))
mu_fit, sigma_fit, lamb_fit, f_s_fit = parameters


# define your negative log-likelihood function
def nll(mu, sigma, lamb, f_s):
    pdf_vals = comp_model(inv_mass_regionI, mu, sigma, lamb, f_s, Ilower_bound, Iupper_bound)
    if np.any(pdf_vals <= 0):
        return 1e10
    return -np.sum(np.log(pdf_vals))



# initial guesses
m = Minuit(nll, mu=9.46, sigma=0.05, lamb=2.0, f_s=0.5)
m.limits = ((9.3, 9.6), (0.01, 0.2), (0.1, 10.0), (0.0, 1.0))
m.migrad()   # run the minimizer
m.hesse()    # compute errors

print(f'hello{m.values[0]}')  # best-fit parameters
print(m.errors[0])


print(f"μ = {mu_fit:.4f} GeV")
print(f"σ = {sigma_fit:.4f} GeV")
print(f"λ = {lamb_fit:.3f} GeV⁻¹")
print(f"f_s = {f_s_fit:.3f}")

plt.hist(inv_mass_regionI, bins=500, density=True, color='cyan', label='Data')
m_plot = np.linspace(Ilower_bound, Iupper_bound, 500)
plt.plot(m_plot, comp_model(m_plot, mu_fit, sigma_fit, lamb_fit, f_s_fit, Ilower_bound, Iupper_bound),
         'black', label='Gaussian + Exponential (fit)')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Probability density")
plt.title("Υ(1S) composite normalized fit")
plt.xlim(9.3, 9.6)  
plt.legend()
plt.show()

model_vals = comp_model(bin_centers, mu_fit, sigma_fit, lamb_fit, f_s_fit, Ilower_bound, Iupper_bound)

residuals = ydata1 - model_vals
residualsd = (ydata1 - model_vals) / np.sqrt(ydata1)

plt.plot(bin_centers, residuals, 'o', label='Residuals')
#plt.plot(bin_centers, residualsd, '+', label='Normalized Residuals')
plt.axhline(0, color='black', linestyle='--')     
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Residuals")
plt.title("Residuals of Fit to Υ(1S) Region")
plt.legend()
plt.show()        

def testing(inv_mass_regionX, lims):
    
    entries, bedges, _ = plt.hist(inv_mass_regionX, bins=500,  color='cyan', label='Data')
    plt.xlabel("Invariant mass (GeV/c²)")
    plt.ylabel("Entries per bin")
    plt.title("Υ(1S) region (data)")
    plt.xlim(lims[0], lims[1])
    plt.legend()
    plt.show()
    bin_centers = 0.5 * (bedges[:-1] + bedges[1:])

    norm = np.trapz(entries, bin_centers)
    if norm == 0:
        ydata = entries
    else:
        ydata = entries / norm

    #ydata = entries / np.trapz(entries, bin_centers)
    return ydata, bin_centers

area, error = integrate.quad(lambda xx: comp_model(xx, mu_fit, sigma_fit, lamb_fit, f_s_fit, Ilower_bound, Iupper_bound), Ilower_bound, Iupper_bound)
print(f"Composite PDF integral = {area} ± {error}")



Ilower_bound_P2, Iupper_bound_P2 = 9.7, 10.1
inv_mass_regionII = np.array([mass for mass in xmass if Ilower_bound_P2 <= mass <= Iupper_bound_P2])


entries2, bedges2, p2 = plt.hist(inv_mass_regionII, bins=100, color='cyan', label='Data')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Entries per bin")
plt.xlim(9.9, 10.1)  
plt.title("Υ(2S) region (data)")
plt.legend()
plt.show()

bin_centers2 = 0.5 * (bedges2[:-1] + bedges2[1:])
ydata2 = entries2 / np.trapz(entries2, bin_centers2)

p02= [10.01, 0.05, 2.0, 0.5]
bounds2 = ([9.9, 0.01, 0.1, 0.0], [10.1, 0.2, 10.0, 1.0])

parameters2, pcov2 = curve_fit(lambda x, mu, sigma, lamb, f_s: comp_model(x, mu, sigma, lamb, f_s, Ilower_bound_P2, Iupper_bound_P2), bin_centers2, ydata2, p0=p02, bounds=bounds2)
errors2 = np.sqrt(np.diag(pcov2))
mu_fit2, sigma_fit2, lamb_fit2, f_s_fit2 = parameters2


print(f"μ = {mu_fit2:.4f} GeV")
print(f"σ = {sigma_fit2:.4f} GeV")
print(f"λ = {lamb_fit2:.3f} GeV⁻¹")
print(f"f_s = {f_s_fit2:.3f}")

plt.hist(inv_mass_regionII, bins=100, density=True, color='cyan', label='Data')
m_plot2 = np.linspace(Ilower_bound_P2, Iupper_bound_P2, 500)
plt.plot(m_plot2, comp_model(m_plot2, mu_fit2, sigma_fit2, lamb_fit2, f_s_fit2, Ilower_bound_P2, Iupper_bound_P2),
         'black', label='Gaussian + Exponential (fit)')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Probability density")
plt.title("Υ(2S) composite normalized fit")
plt.xlim(9.9, 10.1)  
plt.legend()
plt.show()

model_vals2 = comp_model(bin_centers2, mu_fit2, sigma_fit2, lamb_fit2, f_s_fit2, Ilower_bound_P2, Iupper_bound_P2)

residuals2 = ydata2 - model_vals2
residualsd2 = (ydata2 - model_vals2) / np.sqrt(ydata2)

plt.plot(bin_centers2, residuals2, 'o', label='Residuals')
#plt.plot(bin_centers2, residualsd2, '+', label='Normalized Residuals')
plt.axhline(0, color='black', linestyle='--')     
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Residuals")
plt.title("Residuals of Fit to Υ(2S) Region")
plt.legend()
plt.show()   



Ilower_bound_P3, Iupper_bound_P3 = 10.2, 10.5
inv_mass_regionIII = np.array([mass for mass in xmass if Ilower_bound_P3 <= mass <= Iupper_bound_P3])


entries3, bedges3, p3 = plt.hist(inv_mass_regionIII, bins=100, color='cyan', label='Data')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Entries per bin")
plt.xlim(10.2, 10.5)  
plt.title("Υ(3S) region (data)")
plt.legend()
plt.show()

bin_centers3 = 0.5 * (bedges3[:-1] + bedges3[1:])
ydata3 = entries3 / np.trapz(entries3, bin_centers3)

p03 = [10.35, 0.05, 2.0, 0.5]  
bounds3= ([10.2, 0.01, 0.1, 0.0], [10.5, 0.2, 10.0, 1.0])


parameters3, pcov3 = curve_fit(lambda x, mu, sigma, lamb, f_s: comp_model(x, mu, sigma, lamb, f_s, Ilower_bound_P3, Iupper_bound_P3), bin_centers3, ydata3, p0=p03, bounds=bounds3)
errors3 = np.sqrt(np.diag(pcov3))
mu_fit3, sigma_fit3, lamb_fit3, f_s_fit3 = parameters3


print(f"μ = {mu_fit3:.4f} GeV")
print(f"σ = {sigma_fit3:.4f} GeV")
print(f"λ = {lamb_fit3:.3f} GeV⁻¹")
print(f"f_s = {f_s_fit3:.3f}")

plt.hist(inv_mass_regionIII, bins=100, density=True, color='cyan', label='Data')
m_plot3 = np.linspace(Ilower_bound_P3, Iupper_bound_P3, 500)
plt.plot(m_plot3, comp_model(m_plot3, mu_fit3, sigma_fit3, lamb_fit3, f_s_fit3, Ilower_bound_P3, Iupper_bound_P3),
         'black', label='Gaussian + Exponential (fit)')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Probability density")
plt.title("Υ(3S) composite normalized fit")
plt.xlim(10.2, 10.5)  
plt.legend()
plt.show()


model_vals3 = comp_model(bin_centers3, mu_fit3, sigma_fit3, lamb_fit3, f_s_fit3, Ilower_bound_P3, Iupper_bound_P3)
residuals3 = ydata3 - model_vals3
residualsd3 = (ydata3 - model_vals3) / np.sqrt(ydata3)


plt.plot(bin_centers3, residuals3, 'o', label='Residuals')
#plt.plot(bin_centers3, residualsd3, '+', label='Normalized Residuals')
plt.axhline(0, color='black', linestyle='--')     
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Residuals")
plt.title("Residuals of Fit to Υ(3S) Region")
plt.legend()
plt.show()   






from scipy.special import erf
from scipy import stats

n=2
alpha = 1

def crystal_ball(x, alpha, n, xbar, sigma):

    x = np.asarray(x, dtype=float)
    eps = 1e-12  # small offset to avoid division by zero
    
    # Ensure parameters are valid
    sigma = max(sigma, eps)
    alpha = np.sign(alpha) * max(abs(alpha), eps)
    n = max(n, 1.0001)  # avoid n=1

    # Precompute constants
    n_over_alpha = n / abs(alpha)
    exp_term = np.exp(-0.5 * alpha**2)
    A = (n_over_alpha)**n * exp_term
    B = n_over_alpha - abs(alpha)
    C = n_over_alpha / (n - 1.0) * exp_term
    D = np.sqrt(0.5 * np.pi) * (1.0 + erf(abs(alpha) / np.sqrt(2.0)))
    N = 1.0 / (sigma * (C + D) + eps)

    # Gaussian and power-law regions
    mask = (x - xbar) / sigma > -alpha
    result = np.zeros_like(x)

    # Gaussian tail
    result[mask] = N * np.exp(-0.5 * ((x[mask] - xbar) / sigma) ** 2)

    # Power-law tail
    power_arg = B - (x[~mask] - xbar) / sigma
    power_arg = np.clip(power_arg, eps, None)  # ensure positive base
    result[~mask] = N * A * power_arg ** -n

    # Normalize numerically (optional, but safe)
    area = np.trapz(result, x)
    if not np.isfinite(area) or area <= eps:
        area = 1.0  # fallback
    result /= area

    return result, mask



def crystal_ball(x, alpha, n, xbar, sigma):

    n_over_alpha = n / abs(alpha)
    exp = np.exp(-0.5 * alpha**2)
    A = (n_over_alpha)**n * exp
    B = n_over_alpha - abs(alpha)
    C = n_over_alpha / (n - 1) * exp
    D = np.sqrt(0.5 * np.pi) * (1 + erf(abs(alpha) / np.sqrt(2)))
    N = 1 / (sigma * (C + D))

    mask = (x - xbar) / sigma > -alpha
    result = np.zeros_like(x)
    result[mask] = N * np.exp(-0.5 * ((x[mask] - xbar) / sigma) ** 2)
    result[~mask] = N * A * (B - (x[~mask] - xbar) / sigma) ** -n

    # Ensure total area = 1
    result /= np.trapz(result, x)
    return result, mask

# --- Example usage ---
alpha = 6
n = 10
mu = 9.45
sigma = 0.05


ns = [2,3,4,5,6,7,8,9,10]

for n in ns:

    x_sorted = np.sort(inv_mass_regionI)
    y, mask = crystal_ball(x_sorted, alpha, n, mu, sigma)

    plt.hist(inv_mass_regionI, bins=100, density=True, alpha=0.5, label="Data", color='cyan')
    plt.plot(x_sorted, y, color='black', lw=2, label=f"Crystal Ball fit, n={n}" )
    plt.xlabel("Invariant mass (GeV/c²)")
    plt.ylabel("Probability density")
    plt.title("Υ(1S) Region with Crystal Ball Model")
    plt.legend()
    plt.show()
 

    
alpha = 2
n = 1
mu = 9.45
sigma = 0.05

mu2 = 10.01
sigma2 = 0.05
    
    
def comp_model_crystal_ball_exp (x, mu, sigma, lamb, f_s, a, b):
    
    return f_s * crystal_ball(x, alpha, n, mu, sigma)[0] + (1 - f_s) * normalized_exp(x, lamb, a, b)


parameters_cb, pcov_cb = curve_fit(lambda x, mu, sigma, lamb, f_s: comp_model_crystal_ball_exp(x, mu, sigma, lamb, f_s, Ilower_bound, Iupper_bound), bin_centers, ydata1, p0=p0, bounds=bounds)
errors_cb = np.sqrt(np.diag(pcov_cb))
mu_fit_cb, sigma_fit_cb, lamb_fit_cb, f_s_fit_cb = parameters_cb    

xplot = np.linspace(Ilower_bound, Iupper_bound, 500)

plt.hist(inv_mass_regionI, bins=100, density=True, color='cyan', label='Data')
plt.plot(xplot, comp_model_crystal_ball_exp(xplot, mu_fit_cb, sigma_fit_cb, lamb_fit_cb, f_s_fit_cb, Ilower_bound, Iupper_bound),
         'black', label='Crystal Ball + Exponential (fit)')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Probability density")
plt.title("Υ(1S) Composite Normalized Fit with Crystal Ball")       
plt.legend()
plt.show()  

residuals_cb = ydata1 - comp_model_crystal_ball_exp(bin_centers, mu_fit_cb, sigma_fit_cb, lamb_fit_cb, f_s_fit_cb, Ilower_bound, Iupper_bound)   
plt.plot(bin_centers, residuals_cb, 'o', label='Residuals')
plt.axhline(0, color='black', linestyle='--')     
plt.xlabel("Invariant mass (GeV/c²)")   
plt.ylabel("Residuals")
plt.title("Residuals of Fit to Υ(1S) Region with Crystal Ball")
plt.legend()
plt.show()  


xplot2 = np.linspace(Ilower_bound_P2, Iupper_bound_P2, 500)

parameters_cb2, pcov_cb2 = curve_fit(lambda x, mu2, sigma2, lamb, f_s: comp_model_crystal_ball_exp(x, mu2, sigma2, lamb, f_s, Ilower_bound_P2, Iupper_bound_P2), bin_centers2, ydata2, p0=p02, bounds=bounds2)
errors_cb2 = np.sqrt(np.diag(pcov_cb2))
mu_fit_cb2, sigma_fit_cb2, lamb_fit_cb2, f_s_fit_cb2 = parameters_cb2    


plt.hist(inv_mass_regionII, bins=100, density=True, color='cyan', label='Data')
plt.plot(xplot2, comp_model_crystal_ball_exp(xplot2, mu_fit_cb2, sigma_fit_cb2, lamb_fit_cb2, f_s_fit_cb2, Ilower_bound_P2, Iupper_bound_P2),
         'black', label='Crystal Ball + Exponential (fit)')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Probability density")
plt.title("Υ(2S) Composite Normalized Fit with Crystal Ball")       
plt.legend()
plt.show()  

residuals_cb2 = ydata2 - comp_model_crystal_ball_exp(bin_centers2, mu_fit_cb2, sigma_fit_cb2, lamb_fit_cb2, f_s_fit_cb2, Ilower_bound_P2, Iupper_bound_P2)   
plt.plot(bin_centers2, residuals_cb2, 'o', label='Residuals')
plt.axhline(0, color='black', linestyle='--')     
plt.xlabel("Invariant mass (GeV/c²)")   
plt.ylabel("Residuals")
plt.title("Residuals of Fit to Υ(2S) Region with Crystal Ball")
plt.legend()
plt.show()  



xplot3 = np.linspace(Ilower_bound_P3, Iupper_bound_P3, 500)

parameters_cb3, pcov_cb3 = curve_fit(lambda x, mu3, sigma3, lamb, f_s: comp_model_crystal_ball_exp(x, mu3, sigma3, lamb, f_s, Ilower_bound_P3, Iupper_bound_P3), bin_centers3, ydata3, p0=p03, bounds=bounds3)
errors_cb3 = np.sqrt(np.diag(pcov_cb3))
mu_fit_cb3, sigma_fit_cb3, lamb_fit_cb3, f_s_fit_cb3 = parameters_cb3    

plt.hist(inv_mass_regionIII, bins=100, density=True, color='cyan', label='Data')
plt.plot(xplot3, comp_model_crystal_ball_exp(xplot3, mu_fit_cb3, sigma_fit_cb3, lamb_fit_cb3, f_s_fit_cb3, Ilower_bound_P3, Iupper_bound_P3),
         'black', label='Crystal Ball + Exponential (fit)')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Probability density")
plt.title("Υ(3S) Composite Normalized Fit with Crystal Ball")       
plt.legend()
plt.show()  

residuals_cb3 = ydata3 - comp_model_crystal_ball_exp(bin_centers3, mu_fit_cb3, sigma_fit_cb3, lamb_fit_cb3, f_s_fit_cb3, Ilower_bound_P3, Iupper_bound_P3)   
plt.plot(bin_centers3, residuals_cb3, 'o', label='Residuals')
plt.axhline(0, color='black', linestyle='--')     
plt.xlabel("Invariant mass (GeV/c²)")   
plt.ylabel("Residuals")
plt.title("Residuals of Fit to Υ(3S) Region with Crystal Ball")
plt.legend()
plt.show()  
"""