

from enum import Enum


class UpdateTypes(str, Enum):
    IND_02 = "ind_0.2"
    COR_02 = "cor_0.2"
    SKEW_02 = "skew_0.2"
    
    def __repr__(self):
        return self.value
    
    def __str__(self):
        return self.value