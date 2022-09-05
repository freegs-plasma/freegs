from freegs import machine, reconstruction
import pickle
show=False
"""
Script to save tokamak objects with pickle
Run this to save 8 pkl files to your freegs library
This greatly decrease time of test_reconstruction script
if these equilibria are preconstructed
"""

with open('DD.pkl', 'wb') as outp:
    x_z1 = 0.6
    x_z2 = -0.6
    x_r1 = 1.1
    x_r2 = 1.1

    tokamak = machine.EfitTestMachine()

    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=1,
                                         alpha_n=2, x_z1=x_z1, x_z2=x_z2, x_r1=x_r1, x_r2=x_r2,
                                         show=show)
    pickle.dump(eq.tokamak, outp, pickle.HIGHEST_PROTOCOL)

with open('DS.pkl', 'wb') as outp:
    x_z1 = 0.7
    x_z2 = -0.6
    x_r1 = 1.1
    x_r2 = 1.1

    tokamak = machine.EfitTestMachine()

    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=1,
                                              alpha_n=2, x_z1=x_z1,
                                              x_z2=x_z2, x_r1=x_r1,
                                              x_r2=x_r2,
                                              show=show)
    pickle.dump(eq.tokamak, outp, pickle.HIGHEST_PROTOCOL)



with open('LD.pkl', 'wb') as outp:
    x_z1 = 0.6
    x_z2 = -0.6
    x_r1 = 1
    x_r2 = 1

    tokamak = machine.EfitTestMachine()

    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=1,
                                         alpha_n=2, x_z1=x_z1, x_z2=x_z2, x_r1=x_r1, x_r2=x_r2,
                                         show=show)
    pickle.dump(eq.tokamak, outp, pickle.HIGHEST_PROTOCOL)


with open('LS.pkl', 'wb') as outp:
    x_z1 = 0.7
    x_z2 = -0.6
    x_r1 = 1
    x_r2 = 1

    tokamak = machine.EfitTestMachine()

    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=1,
                                         alpha_n=2, x_z1=x_z1, x_z2=x_z2, x_r1=x_r1, x_r2=x_r2,
                                         show=show)
    pickle.dump(eq.tokamak, outp, pickle.HIGHEST_PROTOCOL)

with open('DDV.pkl', 'wb') as outp:
    tokamak = machine.EfitTestMachine()

    i = 4
    eigenfunction = tokamak.eigenbasis[:, i]

    fil_num = 0
    for fil in tokamak.vessel:

        if isinstance(fil,machine.Filament):
            r, z = fil.R, fil.Z
            size = (eigenfunction[fil_num]) * 400
            tokamak.coils.append(('fil' + str(fil_num), machine.Coil(r, z, control=False, current=size)))
            fil_num+=1

        if isinstance(fil, machine.Passive):
            for subfil in fil.filaments:
                r, z = subfil.R, subfil.Z
                size = (eigenfunction[fil_num]) * 400
                tokamak.coils.append(('fil'+str(fil_num),machine.Coil(r, z, control=False, current=size)))
                fil_num += 1

    x_z1 = 0.6
    x_z2 = -0.6
    x_r1 = 1.1
    x_r2 = 1.1


    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=1,
                                              alpha_n=2, x_z1=x_z1,
                                              x_z2=x_z2, x_r1=x_r1,
                                              x_r2=x_r2,
                                              show=show)
    pickle.dump(eq.tokamak, outp, pickle.HIGHEST_PROTOCOL)

with open('DSV.pkl', 'wb') as outp:
    tokamak = machine.EfitTestMachine()

    i = 4
    eigenfunction = tokamak.eigenbasis[:, i]

    fil_num = 0
    for fil in tokamak.vessel:

        if isinstance(fil,machine.Filament):
            r, z = fil.R, fil.Z
            size = (eigenfunction[fil_num]) * 400
            tokamak.coils.append(('fil' + str(fil_num), machine.Coil(r, z, control=False, current=size)))
            fil_num+=1

        if isinstance(fil, machine.Passive):
            for subfil in fil.filaments:
                r, z = subfil.R, subfil.Z
                size = (eigenfunction[fil_num]) * 400
                tokamak.coils.append(('fil'+str(fil_num),machine.Coil(r, z, control=False, current=size)))
                fil_num += 1

    x_z1 = 0.7
    x_z2 = -0.6
    x_r1 = 1.1
    x_r2 = 1.1


    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=1,
                                              alpha_n=2, x_z1=x_z1,
                                              x_z2=x_z2, x_r1=x_r1,
                                              x_r2=x_r2,
                                              show=show)
    pickle.dump(eq.tokamak, outp, pickle.HIGHEST_PROTOCOL)


with open('LDV.pkl', 'wb') as outp:
    tokamak = machine.EfitTestMachine()

    i = 4
    eigenfunction = tokamak.eigenbasis[:, i]

    fil_num = 0
    for fil in tokamak.vessel:

        if isinstance(fil,machine.Filament):
            r, z = fil.R, fil.Z
            size = (eigenfunction[fil_num]) * 400
            tokamak.coils.append(('fil' + str(fil_num), machine.Coil(r, z, control=False, current=size)))
            fil_num+=1

        if isinstance(fil, machine.Passive):
            for subfil in fil.filaments:
                r, z = subfil.R, subfil.Z
                size = (eigenfunction[fil_num]) * 400
                tokamak.coils.append(('fil'+str(fil_num),machine.Coil(r, z, control=False, current=size)))
                fil_num += 1

    x_z1 = 0.6
    x_z2 = -0.6
    x_r1 = 1
    x_r2 = 1


    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=1,
                                              alpha_n=2, x_z1=x_z1,
                                              x_z2=x_z2, x_r1=x_r1,
                                              x_r2=x_r2,
                                              show=show)
    pickle.dump(eq.tokamak, outp, pickle.HIGHEST_PROTOCOL)


with open('LSV.pkl', 'wb') as outp:
    tokamak = machine.EfitTestMachine()

    i = 4
    eigenfunction = tokamak.eigenbasis[:, i]

    fil_num = 0
    for fil in tokamak.vessel:

        if isinstance(fil,machine.Filament):
            r, z = fil.R, fil.Z
            size = (eigenfunction[fil_num]) * 400
            tokamak.coils.append(('fil' + str(fil_num), machine.Coil(r, z, control=False, current=size)))
            fil_num+=1

        if isinstance(fil, machine.Passive):
            for subfil in fil.filaments:
                r, z = subfil.R, subfil.Z
                size = (eigenfunction[fil_num]) * 400
                tokamak.coils.append(('fil'+str(fil_num),machine.Coil(r, z, control=False, current=size)))
                fil_num += 1

    x_z1 = 0.7
    x_z2 = -0.6
    x_r1 = 1
    x_r2 = 1


    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=1,
                                              alpha_n=2, x_z1=x_z1,
                                              x_z2=x_z2, x_r1=x_r1,
                                              x_r2=x_r2,
                                              show=show)
    pickle.dump(eq.tokamak, outp, pickle.HIGHEST_PROTOCOL)

