from .jfinder import JumpFinder
from .jclassifier import JumpClassifier
from .core import KData
from .discontinuity import Discontinuity, DiscontinuitySet
from .models import Slope, Jump, Jump2, Transit, Flare

__all__ = 'correct_jumps KData JumpFinder JumpClassifier Discontinuity DiscontinuitySet'.split()

def correct_jumps(data, jumps):
    """Correct jumps
    
    Parameters
    ----------
    data  : KData
            Cadence and flux values
            
    jumps : list or DiscontinuitySet
            A list of jumps found by JumpFinder and classified
            by JumpClassifier

    jc    : JumpClassifier
    """
    kd = KData(data._cadence, data._flux, data._quality)
    for j in jumps:
        if isinstance(j.type, (Jump,Jump2)):
            kd._flux[kd._mask] -= kd.median*j.global_model_wo_baseline()
    return kd
