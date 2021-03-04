from numbers import Number

from .gradshafranov import mu0
import numpy as np

def getForces(Rs, Zs, coil, equilibrium):
    """
    Calculate forces on the coils in Newtons.

    Rs: float or array - radial position where the force is evaluated
    Zs: float or array - vertical position where the force is evaluated

    Returns an array of two elements or arrays: [ Fr, Fz ]


    Force on coil due to its own current:
        Lorentz self‐forces on curved current loops
        Physics of Plasmas 1, 3425 (1998); https://doi.org/10.1063/1.870491
        David A. Garren and James Chen
    """
    current = coil.current  # current per turn

    # Calculated coil force or force acting on each element.
    if isinstance(Rs, Number) or (hasattr(Rs, '__len__') and len(Rs) == 1):
        total_current = current * coil.turns  # Total toroidal current
    elif not hasattr(Rs, '__len__'):
        return TypeError("The Rs and Zs should be either arrays or numbers.")
    else:
        total_current = current

    # Calculate field at this coil due to all other coils
    # and plasma. Need to zero this coil's current
    coil.current = 0.0
    Br = equilibrium.Br(Rs, Zs)
    Bz = equilibrium.Bz(Rs, Zs)
    coil.current = current

    # Assume circular cross-section for hoop (coil) force
    minor_radius = np.sqrt(coil.area / np.pi)

    # coil inductance factor, depending on internal current
    # distribution. 0.5 for uniform current, 0 for surface current
    coil_inductance = 0.5

    # Force per unit length.
    # In cgs units f = I^2/(c^2 * R) * (ln(8*R/a) - 1 + xi/2)
    # In SI units f = mu0 * I^2 / (4*pi*R) * (ln(8*R/a) - 1 + xi/2)
    coil_fr = (mu0 * total_current ** 2 / (4.0 * np.pi * Rs)) * (
            np.log(8.0 * Rs / minor_radius) - 1 + coil_inductance / 2.0
    )

    Ltor = 2 * np.pi * coil.R  # Length of coil
    return np.array(
        [
            (total_current * Bz + coil_fr)
            * Ltor,  # Jphi x Bz = Fr, coil force always outwards
            -total_current * Br * Ltor,
        ]
    )  # Jphi x Br = - Fz
