import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from iminuit import Minuit

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
with open("ups-15-small.bin", "rb") as f:
    datalist = np.fromfile(f, dtype=np.float32)

nevent = int(len(datalist) / 6)
xdata = datalist.reshape(nevent, 6)
xmass = xdata[:, 0]

# Fit mass window
fit_low, fit_high = 9.3, 10.7
x = xmass[(xmass >= fit_low) & (xmass <= fit_high)]
Nobs = len(x)
print("Events in unbinned fit =", Nobs)

# --------------------------------------------------------------
# Normalised PDFs (truncated to the fit window)
# --------------------------------------------------------------
def gauss_trunc(x, mu, sigma, a, b):
    A = (a - mu) / (sigma * np.sqrt(2))
    B = (b - mu) / (sigma * np.sqrt(2))
    Z = 0.5 * (erf(B) - erf(A))
    return np.exp(-0.5*((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi) * Z)

def exp_trunc(x, lam, a, b):
    Z = 1 - np.exp(-lam * (b - a))
    return (lam * np.exp(-lam * (x - a))) / Z

# -------------------------------------------------------------------
# Negative log-likelihood using FRACTIONS
# -------------------------------------------------------------------
def nll_frac(f1, mu1, sigma1,
             f2, mu2, sigma2,
             f3, mu3, sigma3,
             lam):

    # enforce physical domain
    if (f1 < 0) or (f2 < 0) or (f3 < 0):
        return 1e12
    fsum = f1 + f2 + f3
    if fsum >= 1:
        return 1e12
    if sigma1 <= 0 or sigma2 <= 0 or sigma3 <= 0:
        return 1e12
    if lam <= 0:
        return 1e12

    g1 = gauss_trunc(x, mu1, sigma1, fit_low, fit_high)
    g2 = gauss_trunc(x, mu2, sigma2, fit_low, fit_high)
    g3 = gauss_trunc(x, mu3, sigma3, fit_low, fit_high)
    bg = exp_trunc(x, lam, fit_low, fit_high)

    pdf = f1*g1 + f2*g2 + f3*g3 + (1 - fsum)*bg
    if np.any(pdf <= 0):
        return 1e12

    return -np.sum(np.log(pdf))

# -------------------------------------------------------------------
# Minuit setup
# -------------------------------------------------------------------
init = dict(
    f1=0.12, mu1=9.46, sigma1=0.045,
    f2=0.03, mu2=10.02, sigma2=0.045,
    f3=0.015, mu3=10.36, sigma3=0.045,
    lam=0.6
)

m = Minuit(nll_frac, **init)
m.errordef = Minuit.LIKELIHOOD

# Limits
for f in ["f1","f2","f3"]:
    m.limits[f] = (1e-6, 0.5)

m.limits["mu1"] = (9.2, 9.7)
m.limits["mu2"] = (9.7, 10.1)
m.limits["mu3"] = (10.1, 10.6)

for s in ["sigma1","sigma2","sigma3"]:
    m.limits[s] = (1e-3, 0.15)

m.limits["lam"] = (1e-3, 10)

m.strategy = 2

m.migrad()
m.hesse()

print("\n===== FRACTION FIT RESULTS =====")
for p in m.parameters:
    print(f"{p:7s} = {m.values[p]:10.6f} ± {m.errors[p]:.6f}")

# -------------------------------------------------------------------
# Convert fractions → yields
# -------------------------------------------------------------------
f1, df1 = m.values["f1"], m.errors["f1"]
f2, df2 = m.values["f2"], m.errors["f2"]
f3, df3 = m.values["f3"], m.errors["f3"]

fsum = f1 + f2 + f3

N1, dN1 = f1*Nobs, df1*Nobs
N2, dN2 = f2*Nobs, df2*Nobs
N3, dN3 = f3*Nobs, df3*Nobs
Nb = (1 - fsum)*Nobs

print("\n===== YIELDS =====")
print(f"N1 = {N1:.1f} ± {dN1:.1f}")
print(f"N2 = {N2:.1f} ± {dN2:.1f}")
print(f"N3 = {N3:.1f} ± {dN3:.1f}")
print(f"Nb = {Nb:.1f}")

# -------------------------------------------------------------------
# Plot fit on top of binned data
# -------------------------------------------------------------------
xx = np.linspace(fit_low, fit_high, 2000)

pdf_total = (
    f1 * gauss_trunc(xx, m.values["mu1"], m.values["sigma1"], fit_low, fit_high) +
    f2 * gauss_trunc(xx, m.values["mu2"], m.values["sigma2"], fit_low, fit_high) +
    f3 * gauss_trunc(xx, m.values["mu3"], m.values["sigma3"], fit_low, fit_high) +
    (1 - fsum) * exp_trunc(xx, m.values["lam"], fit_low, fit_high)
)

plt.hist(x, bins=300, range=(fit_low, fit_high), density=True, alpha=0.4)
plt.plot(xx, pdf_total, "k-", lw=2)
plt.show()

# ============================================================
# RESIDUALS / PULLS (requires binning)
# ============================================================

# Make a histogram for display
nbins = 150
hist_vals, bin_edges = np.histogram(x, bins=nbins, range=(fit_low, fit_high))
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width = bin_edges[1] - bin_edges[0]

# Convert fitted fractions → expected bin counts
def model_counts(x):
    g1 = N1 * gauss_trunc(x, m.values["mu1"], m.values["sigma1"], fit_low, fit_high)
    g2 = N2 * gauss_trunc(x, m.values["mu2"], m.values["sigma2"], fit_low, fit_high)
    g3 = N3 * gauss_trunc(x, m.values["mu3"], m.values["sigma3"], fit_low, fit_high)
    b  = Nb * exp_trunc(x, m.values["lam"], fit_low, fit_high)
    return g1 + g2 + g3 + b

expected_counts = model_counts(bin_centres) * bin_width

# DATA UNCERTAINTY
data_err = np.sqrt(hist_vals)

# Residuals
residuals = hist_vals - expected_counts

# Pulls (correct definition)
pulls = residuals / (data_err + 1e-9)

# -------------------------------------------------------------------
# Plot fit + residuals
# -------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True
)

# -----------------------------
# TOP PANEL (ax1): Data + Fit
# -----------------------------
ax1.errorbar(
    bin_centres, hist_vals,
    yerr=data_err,
    fmt='o', color='black', markersize=3,
    label="Data"
)

xx = np.linspace(fit_low, fit_high, 2000)
ax1.plot(xx, model_counts(xx) * bin_width, 'r-', lw=2, label="Fit")

ax1.set_ylabel("Counts per bin")
ax1.legend()
ax1.set_title("Full-range fit: Υ(1S), Υ(2S), Υ(3S) + background")

# -----------------------------
# BOTTOM PANEL (ax2): Pulls
# -----------------------------
ax2.axhline(0, color='black', lw=1)

ax2.plot(bin_centres, pulls, 'o', color='blue', markersize=3)

ax2.set_ylabel("Pull")
ax2.set_xlabel("Invariant mass (GeV)")
ax2.set_ylim(-5, 5)

plt.tight_layout()
plt.show()