import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import crystalball
from iminuit import Minuit as mn

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


def crystal_ball_with_background(x, mu, sigma, A, C, lam):
    """
    Crystal Ball signal (pdf) scaled by amplitude A plus an exponential background.
    A has units of counts per bin (so the model returns counts comparable to histogram y).
    """
    cb_pdf = crystalball.pdf(x, beta=2, m=1, loc=mu, scale=sigma)
    cb = A * cb_pdf
    bg = C * np.exp(-lam * x)

    return cb + bg

def minuit_crystal_ball_fit(x, y, p0, sigma_err):
    # chi2 to minimise (using provided sigma_err per bin)
    # Note: parameter name for the width must match the Minuit keyword below ("sigma"),
    # so we name the fit parameter "sigma" and use sigma_err for the per-bin uncertainties.
    def chi2(mu, sigma, A, C, lam):
        ymod = crystal_ball_with_background(x, mu, sigma, A, C, lam)
        return np.sum(((y - ymod) / sigma_err) ** 2)

    # create Minuit instance with sensible initial errors and chi2 errordef=1
    # p0 expected: [mu, sigma, A, C, lam]
    m = mn(chi2, mu=p0[0], sigma=p0[1], A=p0[2], C=p0[3], lam=p0[4], errordef=1.0)

    m.migrad()   # minimise
    try:
        m.hesse()    # compute errors / covariance
    except Exception:
        # if hesse fails, continue with whatever migrad provided
        pass

    # extract fitted values in parameter order [mu, sigma, A, C, lam]
    pcb_vals = np.array([m.values["mu"], m.values["sigma"], m.values["A"], m.values["C"], m.values["lam"]])

    # try to get covariance matrix; fall back to diagonal from errors if not available
    try:
        pcov = m.np_matrix()
    except Exception:
        errs = np.array([m.errors["mu"], m.errors["sigma"], m.errors["A"], m.errors["C"], m.errors["lam"]])
        pcov = np.diag(errs ** 2)

    return pcb_vals, pcov

# initial guess: [mu, sigma, A, C, lam]
p0_cb = [9.46, 0.035, np.max(y), 10.0, 0.5]

pcb, pcov_cb = minuit_crystal_ball_fit(x, y, p0_cb, sigma)

print("\nBest-fit parameters (Crystal Ball + Exp):")
names_scb = ['mu', 'sigma', 'A', 'C', 'lam']
for n, v, e in zip(names_scb, pcb, np.sqrt(np.diag(pcov_cb))):
    print(f" {n} = {v:.4g} ± {e:.4g}")

yfit_cb = crystal_ball_with_background(x, *pcb)

def background(x, C, lam):
    return C * np.exp(-lam * x)

yfit_bg = background(x, pcb[3], pcb[4])


plt.figure(figsize=(10,6))
plt.hist(xmass, bins=500, range=(9.0, 11.0), color='cyan', alpha=0.5, label='MC Data')
plt.plot(x, yfit_cb, 'r-', linewidth=2, label='Crystal Ball + Exp Fit (scipy)')
plt.plot(x, yfit_bg, 'k--', linewidth=2, label='Background Component')
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Counts')
plt.title('Fit to Invariant Mass Distribution')
plt.legend()
plt.show()



plt.figure(figsize=(10,6))
plt.hist(xmass_LHCB, bins=500, range=(9.0, 11.0), color='lightgreen', alpha=0.5, label='LHCb Data')
plt.plot(x_LHCB, yfit_cb, 'r-', linewidth=2, label='Crystal Ball + Exp Fit (scipy)')
plt.plot(x_LHCB, yfit_bg, 'k--', linewidth=2, label='Background Component')
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Counts')
plt.title('Fit to Invariant Mass Distribution (LHCb Data)')
plt.legend()
plt.show()