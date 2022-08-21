import freegs
import numpy as np
import matplotlib.pyplot as plt
import pickle
np.set_printoptions(threshold=np.inf)

# Loading Machine
tokamak = freegs.machine.EfitTestMachine(createVessel=True, Nfils=100)


for sensor in tokamak.sensors:
    if isinstance(sensor, freegs.machine.RogowskiSensor):
        print(sensor.name + " current= " + str(
            sensor.measurement))

    elif isinstance(sensor, freegs.machine.PoloidalFieldSensor):
        print(sensor.name +" field=" + str(sensor.measurement))

    elif isinstance(sensor, freegs.machine.FluxLoopSensor):
        print(sensor.name +  " flux=" + str(
            sensor.measurement))
tokamak.printCurrents()

eq = freegs.equilibrium.Equilibrium(tokamak)

freegs.plotting.plotEquilibrium(eq)



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
for filament in tokamak.vessel:
    if isinstance(filament, freegs.machine.Filament_Group):
        for fil in filament.filaments:
            ax.scatter(fil.R, fil.Z, color='green')

        for i in range(filament.eigenbasis.shape[1]):
            eigenfunction = tokamak.eigenbasis[:, i]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            for j, fil in enumerate(filament.filaments):
                size = abs(eigenfunction[j]) * 400

                if eigenfunction[j] >= 0:
                    ax.scatter(fil.R, fil.Z, color='red', s=size)
                else:
                    ax.scatter(fil.R,fil.Z, color='black', s=size)

    elif isinstance(filament,freegs.machine.Filament):
        ax.scatter(filament.R, filament.Z, color='black')

    plt.show()
