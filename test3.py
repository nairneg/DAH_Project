import sys
import numpy as np
import matplotlib.pyplot as plt

# import data
#xmass = np.loadtxt(sys.argv[1])
f = open("../ups-15-small.bin","r")
datalist = np.fromfile(f,dtype=np.float32)

# number of events
nevent = int(len(datalist)/6)

xdata = np.split(datalist,nevent)
print(xdata[0])

# make list of invariant mass of events
xmass = []
for i in range(0,nevent):
    xmass.append(xdata[i][0])
#    if i < 10:
        print(xmass[i])

# make a list of the transverse momentum of muon pairs
xpair_trans_mom = []
for i in range(0,nevent):
    xpair_trans_mom.append(xdata[i][1])

# make a list of rapidity of muon pair
xrapidity = []
for i in range(0,nevent):
    xrapidity.append(xdata[i][2])

# make a list of momentum of muon mairs
xpair_mom = []
for i in range(0,nevent):
    xpair_mom.append(xdata[i][3])

# make a list of the transverse momentum of first muon
xfirst_mom = []
for i in range(0,nevent):
    xfirst_mom.append(xdata[i][4])

# make a list of the transverse momentum of second muon
xsecond_mom = []
for i in range(0,nevent):
    xsecond_mom.append(xdata[i][5])