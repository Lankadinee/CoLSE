

from enum import Enum


class UpdateTypes(str, Enum):
    IND_02 = "ind_0.2"
    COR_02 = "cor_0.2"
    SKEW_02 = "skew_0.2"
    
    def __repr__(self):
        return self.value
    
    def __str__(self):
        return self.value
    
class WorkloadTypes(str, Enum):
    MIXED_RATIO25 = "mixed_ratio25"
    MIXED_RATIO50 = "mixed_ratio50"
    MIXED_RATIO75 = "mixed_ratio75"

    def __repr__(self):
        return self.value
    
    def __str__(self):
        return self.value