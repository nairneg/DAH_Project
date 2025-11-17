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

# -----------------------------
#   DATA: SELECT THE 1S REGION
# -----------------------------
a = 9.0    # lower bound of fit range
b = 9.8     # upper bound of fit range

data = xmass[(xmass >= a) & (xmass <= b)]
Ndata = len(data)

# -----------------------------
#   NORMALISED PDFs
# -----------------------------
def gauss_int(mu, sigma, a=a, b=b):
    z1 = (a - mu) / (np.sqrt(2)*sigma)
    z2 = (b - mu) / (np.sqrt(2)*sigma)
    return sigma * np.sqrt(np.pi/2) * (special.erf(z2) - special.erf(z1))

def pdf_gauss(x, mu, sigma):
    I = gauss_int(mu, sigma)
    return np.exp(-0.5*((x - mu)/sigma)**2) / I

def exp_int(lamb, a=a, b=b):
    if lamb < 1e-12:
        return b - a
    return (np.exp(-lamb*a) - np.exp(-lamb*b)) / lamb

def pdf_exp(x, lamb):
    I = exp_int(lamb)
    return np.exp(-lamb*x) / I

# -----------------------------
#   EXTENDED NEGATIVE LOG-LIKE
# -----------------------------
def nll(mu, sigma, Ns, lamb, Nbkg):
    if sigma <= 0 or Ns < 0 or Nbkg < 0 or lamb < 0:
        return 1e12

    ps = pdf_gauss(data, mu, sigma)
    pb = pdf_exp(data, lamb)

    model = Ns * ps + Nbkg * pb
    if np.any(model <= 0):
        return 1e12

    Ntot = Ns + Nbkg            # extended term
    return Ntot - np.sum(np.log(model))

# -----------------------------
#   MINUIT FIT
# -----------------------------
m = Minuit(nll,
           mu=9.46, sigma=0.05, Ns=20000,
           lamb=1.0, Nbkg=2000)

m.errordef = Minuit.LIKELIHOOD

# Parameter limits
m.limits["mu"]    = (9.2, 9.7)
m.limits["sigma"] = (0.005, 0.15)
m.limits["Ns"]    = (0, 1e6)
m.limits["lamb"]  = (0, 20)
m.limits["Nbkg"]  = (0, 1e6)

m.migrad()
m.hesse()

print(m)

# -----------------------------
#   PLOT THE RESULT
# -----------------------------
xx = np.linspace(a, b, 800)
bw = (b - a) / 100

signal   = m.values["Ns"]   * pdf_gauss(xx, m.values["mu"], m.values["sigma"]) * bw
bkg      = m.values["Nbkg"] * pdf_exp(xx, m.values["lamb"]) * bw
total    = signal + bkg

plt.figure(figsize=(8,5))
plt.hist(data, bins=100, range=(a,b), alpha=0.6, color="gray", label="Data")

plt.plot(xx, total, 'k-', lw=2, label="Total Fit")
plt.plot(xx, signal, 'r--', lw=2, label="Gaussian (Υ1S)")
plt.plot(xx, bkg, 'b--', lw=2, label="Background")

plt.xlabel("Invariant mass (GeV)")
plt.ylabel("Counts per bin")
plt.title("Υ(1S) Fit: Gaussian + Exponential Background")
plt.legend()
plt.show()