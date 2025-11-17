import numpy as np
from scipy import special
from iminuit import Minuit
import matplotlib.pyplot as plt

f = open('ups-15-small.bin',"r")
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



Ilower_bound, Iupper_bound = 9.3, 9.6
inv_mass_regionI = np.array([m for m in xmass if Ilower_bound <= m <= Iupper_bound])



# ======================================================================
# Fit window and data
# ======================================================================
a = Ilower_bound
b = Iupper_bound
data = inv_mass_regionI
N_data = len(data)

# ======================================================================
# Numerically stable normalization integrals (NumPy only)
# ======================================================================

def gauss_int(mu, sigma):
    """Analytic Gaussian integral between a and b, using NumPy + erf."""
    z1 = (a - mu) / (np.sqrt(2) * sigma)
    z2 = (b - mu) / (np.sqrt(2) * sigma)
    return sigma * np.sqrt(np.pi / 2) * (special.erf(z2) - special.erf(z1))


def exp_int(lamb):
    """
    Stable exponential integral using NumPy:
    ∫_a^b e^{-λx} dx = e^{-λa} * (1 - e^{-λ(b-a)}) / λ
    Uses np.expm1 to avoid underflow.
    """
    if lamb < 1e-12:  # treat as flat background
        return b - a

    width = b - a
    return np.exp(-lamb * a) * (-np.expm1(-lamb * width)) / lamb


# ======================================================================
# PDFs normalized to 1 on [a,b]
# ======================================================================

def pdf_signal(x, mu, sigma):
    I = gauss_int(mu, sigma)
    return np.exp(-0.5 * ((x - mu)/sigma)**2) / I

def pdf_background(x, lamb):
    I = exp_int(lamb)
    return np.exp(-lamb * x) / I


# ======================================================================
# Extended unbinned negative log-likelihood
# ======================================================================

def nll(mu, sigma, lamb, Nsig, Nbkg):
    # constraints for stability
    if sigma <= 0 or Nsig < 0 or Nbkg < 0:
        return 1e12

    ps = pdf_signal(data, mu, sigma)
    pb = pdf_background(data, lamb)

    total = Nsig * ps + Nbkg * pb
    if np.any(total <= 0):
        return 1e12

    return (Nsig + Nbkg) - np.sum(np.log(total))


# ======================================================================
# Minuit fit
# ======================================================================

m = Minuit(
    nll,
    mu=9.46,
    sigma=0.03,
    lamb=1.0,
    Nsig=0.5*N_data,
    Nbkg=0.5*N_data
)

m.limits["mu"] = (a, b)
m.limits["sigma"] = (0.001, 0.2)
m.limits["lamb"] = (0.0, 10.0)
m.limits["Nsig"] = (0, 10*N_data)
m.limits["Nbkg"] = (0, 10*N_data)

m.errordef = Minuit.LIKELIHOOD

m.migrad()
m.hesse()

# ======================================================================
# Print results
# ======================================================================
print("\n======= Υ(1S) Fit Results (NumPy Only) =======")
for p in ["mu", "sigma", "lamb", "Nsig", "Nbkg"]:
    print(f"{p:5s} = {m.values[p]:.6f} ± {m.errors[p]:.6f}")
print("==============================================\n")


# ======================================================================
# Plotting
# ======================================================================

nbins = 80
counts, edges = np.histogram(data, bins=nbins, range=(a,b))
centers = 0.5*(edges[:-1] + edges[1:])
bw = edges[1] - edges[0]

m_plot = np.linspace(a, b, 600)

ps = pdf_signal(m_plot, m.values["mu"], m.values["sigma"])
pb = pdf_background(m_plot, m.values["lamb"])

sig_curve = m.values["Nsig"] * ps
bkg_curve = m.values["Nbkg"] * pb
tot_curve = sig_curve + bkg_curve

plt.figure(figsize=(8,5))
plt.hist(data, bins=nbins, range=(a,b), alpha=0.6, label="Data")
plt.plot(m_plot, tot_curve*bw, 'k-',  lw=2, label="Total fit")
plt.plot(m_plot, sig_curve*bw, 'r--', lw=2, label="Signal")
plt.plot(m_plot, bkg_curve*bw, 'b:',  lw=2, label="Background")
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Counts per bin")
plt.title("Υ(1S) Fit — Gaussian + Exponential (NumPy Only)")
plt.legend()
plt.show()


# ======================================================================
# Residuals
# ======================================================================

expected = np.interp(centers, m_plot, tot_curve) * bw
residuals = counts - expected
errs = np.sqrt(np.maximum(counts, 1))

plt.figure(figsize=(8,3))
plt.errorbar(centers, residuals, yerr=errs, fmt='o')
plt.axhline(0, color='k', linestyle='--')
plt.title("Residuals")
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Data - Fit")
plt.show()