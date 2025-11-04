import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
# import data
#xmass = np.loadtxt(sys.argv[1])
f = open("ups-15-small.bin","r")
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


# --- Select Υ(1S) region ---
Ilower_bound, Iupper_bound = 9.3, 9.6
inv_mass_regionI = np.array([m for m in xmass if Ilower_bound <= m <= Iupper_bound])

entries, bedges, _ = plt.hist(inv_mass_regionI, bins=100, histtype='step', color='cyan', label='Data')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Entries per bin")
plt.title("Υ(1S) region (data)")
plt.legend()
plt.show()

bin_centers = 0.5 * (bedges[:-1] + bedges[1:])

# --- Define normalized components ---
def normalized_gauss(x, mu, sigma, a, b):
    """Normalized Gaussian on [a,b]."""
    integral, _ = integrate.quad(lambda xx: np.exp(-0.5*((xx - mu)/sigma)**2), a, b)
    N = 1.0 / (sigma * np.sqrt(2*np.pi) * integral)
    return N * np.exp(-0.5 * ((x - mu)/sigma)**2)

def normalized_exp(x, lamb, a, b):
    """Normalized falling exponential on [a,b]."""

    if abs(lamb) < 1e-9: 
        return np.full_like(x, 1/ (b - a))
    integral, _ = integrate.quad(lambda xx: np.exp(-lamb * xx), a, b)
    norm = lamb * np.exp(-lamb*x) / integral
    return norm

# --- Composite model ---
def comp_model(x, mu, sigma, lamb, f_s, a=Ilower_bound, b=Iupper_bound):
    """Composite PDF: signal fraction f_s * Gaussian + (1-f_s) * exponential."""
    return f_s * normalized_gauss(x, mu, sigma, a, b) + (1 - f_s) * normalized_exp(x, lamb, a, b)

# --- Fit model to histogram (binned likelihood via curve_fit) ---
# convert counts to densities
ydata = entries / np.trapz(entries, bin_centers)

# initial guesses: μ, σ, λ, f_s
p0 = [9.46, 0.05, 2.0, 0.5]
bounds = ([9.3, 0.01, 0.1, 0.0], [9.6, 0.2, 10.0, 1.0])

parameters, pcov = curve_fit(lambda x, mu, sigma, lamb, f_s: comp_model(x, mu, sigma, lamb, f_s),
                             bin_centers, ydata, p0=p0, bounds=bounds)

mu_fit, sigma_fit, lamb_fit, f_s_fit = parameters
print(f"μ = {mu_fit:.4f} GeV")
print(f"σ = {sigma_fit:.4f} GeV")
print(f"λ = {lamb_fit:.3f} GeV⁻¹")
print(f"f_s = {f_s_fit:.3f}")

# --- Plot result ---
plt.hist(inv_mass_regionI, bins=100, density=True, histtype='step', color='cyan', label='Data')
m_plot = np.linspace(Ilower_bound, Iupper_bound, 500)
plt.plot(m_plot, comp_model(m_plot, mu_fit, sigma_fit, lamb_fit, f_s_fit),
         'r-', label='Gaussian + Exponential (fit)')
plt.xlabel("Invariant mass (GeV/c²)")
plt.ylabel("Probability density")
plt.title("Υ(1S) composite normalized fit")
plt.legend()
plt.show()


from scipy import integrate

area, error = integrate.quad(lambda xx: comp_model(xx, mu_fit, sigma_fit, lamb_fit, f_s_fit,
                                                   Ilower_bound, Iupper_bound),
                             Ilower_bound, Iupper_bound)


print(f"Composite PDF integral = {area:.6f} ± {error:.2e}")