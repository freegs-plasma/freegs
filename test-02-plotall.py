import freegs

import matplotlib.pyplot as plt

import pickle

resolutions = [33, 65, 129, 257, 513]#, 1025]

boundaries = [("A", 0.1, 2.0, -1.0, 1.0, '-')
              ,("B", 0.5, 1.75, -0.8, 1.1, '--')
             ]

psiloc = (1.2, 0.1) # (R,Z) location to record psi

psivals = {}
coilcurrents = {}

fig, ax1 = plt.subplots()
ax1.set_xlabel("Resolution")
ax1.set_ylabel(r"$\psi(R=%.1f, Z=%.1f)$" % psiloc)
ax2 = ax1.twinx()


for bndry_name, Rmin, Rmax, Zmin, Zmax,linestyle in boundaries:
    psivals[bndry_name] = []
    coilcurrents[bndry_name] = []
    
    for n in resolutions:
        with open("test-02-"+bndry_name+"-"+str(n)+".pkl", "rb") as f:
            n2 = pickle.load(f)
            bndry = pickle.load(f)
            eq = pickle.load(f)
        
        psivals[bndry_name].append(eq.psiRZ(*psiloc)[0][0])
        coilcurrents[bndry_name].append(eq.tokamak.coils[0][1].current)
        coilname = eq.tokamak.coils[0][0]
    ax1.plot(resolutions, psivals[bndry_name], 'ok', linestyle=linestyle, label="Boundary "+bndry_name)
    ax2.plot(resolutions, coilcurrents[bndry_name], 'sg', linestyle=linestyle, label="Boundary "+bndry_name)
ax2.set_ylabel(coilname + " current [kA]")

plt.legend()

plt.savefig("test-02-converge.pdf")
plt.show()
