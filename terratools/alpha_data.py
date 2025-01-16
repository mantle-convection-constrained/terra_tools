import numpy as np
from scipy.interpolate import interp1d

# Note that the depth-varying values of alpha implemented here are for Earth only.
# Future job will be to implement proper equation of states so that we can calculate
# the depth varying alpha (and other properties) for other bodies.

# In the dictionary alph_eos we store the depth varying alpha from an mt=128 simulation.
# For TerraModels of different radial resolution we interpolate from this to the radii of
# the TerraModel.


class NoEosError(Exception):
    """
    Exception type raised when eos string passed which does not match any given
    """

    def __init(self, eos):
        self.message = f"alpha for equation of state '{eos}' not implemented."
        super().__init__(self.message)


def alpha(eos, model):
    """
    :param eos: string to denote the equation of state
    :type eos: str
    :param model: radii of layers
    :type model: TerraModel
    """
    alph_eos = {
        "murnaghan": [
            3.7904e-05,
            3.6672e-05,
            3.5511e-05,
            3.4426e-05,
            3.3401e-05,
            3.2438e-05,
            3.1525e-05,
            3.0666e-05,
            2.9848e-05,
            2.9076e-05,
            2.8340e-05,
            2.7643e-05,
            2.6977e-05,
            2.6344e-05,
            2.5738e-05,
            2.5162e-05,
            2.4608e-05,
            2.4080e-05,
            2.3573e-05,
            2.3089e-05,
            2.2622e-05,
            2.2175e-05,
            2.1744e-05,
            2.1331e-05,
            2.0932e-05,
            2.0549e-05,
            2.0178e-05,
            1.9822e-05,
            1.9477e-05,
            1.9145e-05,
            1.8822e-05,
            1.8512e-05,
            1.8210e-05,
            1.7919e-05,
            1.7636e-05,
            1.7363e-05,
            1.7097e-05,
            1.6839e-05,
            1.6589e-05,
            1.6346e-05,
            1.6109e-05,
            1.5880e-05,
            1.5656e-05,
            1.5439e-05,
            1.5226e-05,
            1.5020e-05,
            1.4818e-05,
            1.4623e-05,
            1.4431e-05,
            1.4244e-05,
            1.4061e-05,
            1.3883e-05,
            1.3708e-05,
            1.3538e-05,
            1.3371e-05,
            1.3208e-05,
            1.3048e-05,
            1.2891e-05,
            1.2737e-05,
            1.2587e-05,
            1.2439e-05,
            1.2294e-05,
            1.2152e-05,
            1.2012e-05,
            1.1874e-05,
        ],
    }

    # check if passed in eos is implemented
    if eos.lower() not in alph_eos.keys():
        raise NoEosError(eos)

    a = alph_eos[eos.lower()]

    # The data in alph_eos is pasted from TERRA output files.
    # Layer indexing is reversed relative to the terra grid so need to flip
    a = np.flip(a)

    radii = model.get_radii()
    if len(radii) == len(a):
        return a
    else:
        interp = interp1d(np.linspace(np.max(radii), np.min(radii), len(a)), a)

        return interp(radii)
