import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from iminuit import Minuit

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
f = open('ups-15-small.bin', "r")
datalist = np.fromfile(f, dtype=np.float32)

nevent = int(len(datalist) / 6)
xdata = datalist.reshape(nevent, 6)
xmass = xdata[:, 0]

# Fit mass window
a, b = 9.3, 10.7
x = xmass[(xmass >= a) & (xmass <= b)]
Nobs = len(x)
print("Events in fit window =", Nobs)

# -------------------------------------------------------------------
# Normalised truncated PDFs
# -------------------------------------------------------------------
def gauss_trunc(x, mu, sigma, a, b):
    # normalization of truncated gaussian
    A = (a - mu) / (sigma * np.sqrt(2))
    B = (b - mu) / (sigma * np.sqrt(2))
    Z = 0.5 * (erf(B) - erf(A))
    return np.exp(-0.5*((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi) * Z)

def exp_trunc(x, lam, a, b):
    Z = 1 - np.exp(-lam * (b - a))
    return (lam * np.exp(-lam * (x - a))) / Z

# -------------------------------------------------------------------
# Likelihood using fractions
# -------------------------------------------------------------------
def nll_frac(f1, mu1, sigma1,
             f2, mu2, sigma2,
             f3, mu3, sigma3,
             lam):

    # Guard against invalid regions
    if (f1 < 0) or (f2 < 0) or (f3 < 0):
        return 1e12
    fsum = f1 + f2 + f3
    if fsum >= 1:
        return 1e12
    if sigma1 <= 0 or sigma2 <= 0 or sigma3 <= 0:
        return 1e12
    if lam <= 0:
        return 1e12

    g1 = gauss_trunc(x, mu1, sigma1, a, b)
    g2 = gauss_trunc(x, mu2, sigma2, a, b)
    g3 = gauss_trunc(x, mu3, sigma3, a, b)
    bg = exp_trunc(x, lam, a, b)

    pdf = f1*g1 + f2*g2 + f3*g3 + (1 - fsum)*bg
    if np.any(pdf <= 0):
        return 1e12

    return -np.sum(np.log(pdf))

# -------------------------------------------------------------------
# Minuit fit
# -------------------------------------------------------------------
init = dict(
    f1=0.12, mu1=9.46, sigma1=0.045,
    f2=0.03, mu2=10.02, sigma2=0.045,
    f3=0.015, mu3=10.36, sigma3=0.045,
    lam=0.6
)

m = Minuit(nll_frac, **init)
m.errordef = Minuit.LIKELIHOOD

# Parameter limits
m.limits['f1'] = (1e-6, 0.5)
m.limits['f2'] = (1e-6, 0.5)
m.limits['f3'] = (1e-6, 0.5)

m.limits['mu1'] = (9.2, 9.7)
m.limits['mu2'] = (9.7, 10.1)
m.limits['mu3'] = (10.1, 10.6)

m.limits['sigma1'] = (1e-3, 0.15)
m.limits['sigma2'] = (1e-3, 0.15)
m.limits['sigma3'] = (1e-3, 0.15)

m.limits['lam'] = (1e-3, 10)

m.strategy = 2

m.migrad()
m.hesse()

print("\n===== FRACTION FIT RESULTS =====")
for p in m.parameters:
    print(f"{p:7s} = {m.values[p]:10.6f} ± {m.errors[p]:.6f}")

# -------------------------------------------------------------------
# Convert fractions → physical yields
# -------------------------------------------------------------------
f1 = m.values['f1']; df1 = m.errors['f1']
f2 = m.values['f2']; df2 = m.errors['f2']
f3 = m.values['f3']; df3 = m.errors['f3']
fsum = f1 + f2 + f3

N1 = f1 * Nobs; dN1 = df1 * Nobs
N2 = f2 * Nobs; dN2 = df2 * Nobs
N3 = f3 * Nobs; dN3 = df3 * Nobs
Nb = (1 - fsum) * Nobs

print("\n===== YIELDS =====")
print(f"N1 = {N1:.1f} ± {dN1:.1f}")
print(f"N2 = {N2:.1f} ± {dN2:.1f}")
print(f"N3 = {N3:.1f} ± {dN3:.1f}")
print(f"Nb = {Nb:.1f}")

# -------------------------------------------------------------------
# Plot fit over data
# -------------------------------------------------------------------
xx = np.linspace(a, b, 2000)

pdf1 = m.values['f1'] * gauss_trunc(xx, m.values['mu1'], m.values['sigma1'], a, b)
pdf2 = m.values['f2'] * gauss_trunc(xx, m.values['mu2'], m.values['sigma2'], a, b)
pdf3 = m.values['f3'] * gauss_trunc(xx, m.values['mu3'], m.values['sigma3'], a, b)
pdfb = (1 - fsum) * exp_trunc(xx, m.values['lam'], a, b)

total_pdf = pdf1 + pdf2 + pdf3 + pdfb

plt.hist(x, bins=300, range=(a, b), density=True, alpha=0.4)
plt.plot(xx, total_pdf, 'k-', lw=2)
plt.plot(xx, pdf1, 'r--')
plt.plot(xx, pdf2, 'g--')
plt.plot(xx, pdf3, 'b--')
plt.plot(xx, pdfb, 'm--')
plt.show()