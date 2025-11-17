import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import crystalball


import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.special import erf
from scipy.integrate import quad
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import crystalball


def file_read(f_path):
    """
    Reads in binary data from file and transforms it into a 2D numpy array. With columns representing different observables.
    """
    f_open = open(f_path,"r") 
    datalist = np.fromfile(f_open, dtype=np.float32)
    
    nevent = int(len(datalist) / 6)
    xdata = datalist.reshape(nevent, 6)
    return xdata

f_mc = '/Users/nairnegillespie/Desktop/Year 4/DAH Project/mc.bin' #monte carlo data
xdata_mc = file_read(f_mc)

f_small = '/Users/nairnegillespie/Desktop/Year 4/DAH Project/ups-15-small.bin' #smaller file of real data from LHCb for testing
xdata_small = file_read(f_small)

def extract_variables(data):
    """
    Extracts the 6 observable arrays (columns) from 2D array.
    """
    return [data[:, i] for i in range(data.shape[1])]

xmass, xpt, rap, p, p1, p2 = extract_variables(xdata_mc)
xmass_small, xpt_small, rap_small, p_small, p1_small, p2_small = extract_variables(xdata_small)


# ---------------------------------------------------
# 1) Crystal Ball PDF
# ---------------------------------------------------
def crystal_ball_pdf(x, alpha, n, mu, sigma):
    """
    Normalized Crystal Ball PDF
    """
    return crystalball.pdf(x, beta=alpha, m=n, loc=mu, scale=sigma)

# ---------------------------------------------------
# 2) Negative Log-Likelihood for unbinned data
# ---------------------------------------------------
def nll_for_minuit(alpha, n, mu, sigma):
    """
    alpha: tail parameter (>0)
    n: tail exponent (>1)
    mu: mean
    sigma: width (>0)
    """
    # safety guards
    if sigma <= 0 or n <= 0 or alpha <= 0:
        return 1e20  # very bad NLL

    pdf_vals = crystal_ball_pdf(xmass, alpha, n, mu, sigma)
    # avoid log(0)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return -np.sum(np.log(pdf_vals))

# ---------------------------------------------------
# 3) Initial guesses for the parameters
# ---------------------------------------------------
alpha0 = 1.5
n0 = 1.5
mu0 = 9.46
sigma0 = 0.035

# ---------------------------------------------------
# 4) Minuit fit
# ---------------------------------------------------
m = Minuit(
    nll_for_minuit,
    alpha=alpha0,
    n=n0,
    mu=mu0,
    sigma=sigma0
)

# Enforce parameter bounds
m.limits["alpha"] = (0.1, 5)
m.limits["n"] = (0.1, 5)
m.limits["sigma"] = (1e-4, 0.2)


m.errordef = 0.5  # likelihood
m.migrad()
m.hesse()

# Extract fitted parameters
alpha_fit = m.values["alpha"]
n_fit = m.values["n"]
mu_fit = m.values["mu"]
sigma_fit = m.values["sigma"]

print("Fitted parameters:")
print(f"alpha = {alpha_fit:.3f}, n = {n_fit:.3f}, mu = {mu_fit:.5f}, sigma = {sigma_fit:.5f}")

# ---------------------------------------------------
# 5) Plot histogram + fitted PDF
# ---------------------------------------------------
nbins = 500
hist_range = (9.0, 11.0)
entries_mc, bin_edges_mc = np.histogram(xmass, bins=nbins, range=hist_range)
bin_centers_mc = 0.5 * (bin_edges_mc[1:] + bin_edges_mc[:-1])
bin_width = bin_edges_mc[1] - bin_edges_mc[0]

# PDF evaluated on fine grid



#x_plot = np.linspace(hist_range[0], hist_range[1], 1000)
pdf_vals = crystal_ball_pdf(bin_centers_mc, alpha_fit, n_fit, mu_fit, sigma_fit)

# Scale PDF to histogram counts
pdf_scaled = pdf_vals * len(xmass) * bin_width

# Plot
plt.figure(figsize=(10,6))
plt.hist(xmass, bins=nbins, range=hist_range, color='cyan', alpha=0.5, label='MC Data')
plt.plot(bin_centers_mc, pdf_scaled, 'r-', linewidth=2, label='Crystal Ball Fit')
plt.xlabel("Mass [GeV]")
plt.ylabel("Counts")
plt.title("Unbinned Crystal Ball Fit to MC Mass")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#plot residuals
res_cb = entries_mc - pdf_scaled
plt.figure(figsize=(10,4))
plt.bar(bin_centers_mc, res_cb, width=bin_width, color='gray', alpha=0.7)
plt.xlabel("Mass [GeV]")
plt.ylabel("Residuals (Data - Fit)")
plt.title("Residuals of Crystal Ball Fit")
plt.grid(alpha=0.3)
plt.show()

    