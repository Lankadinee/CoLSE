from enum import Enum


class CopulaTypes(Enum):
    GUMBEL = 1
    FRANK = 2
    CLAYTON = 3
    COPULA_NETWORK = 4
    COPULA_MODEL_V1 = 5
    GUMBEL_RANGE = 6
    COPULA_MODEL_V2 = 7
    JOE = 8

    def __str__(self):
        return self.name


class ArchCopulaTypes(str, Enum):
    GUMBEL = "Gumbel"
    CLAYTON = "Clayton"
    FRANK = "Frank"
    GUMBEL_90 = "Gumbel-90"
    GUMBEL_180 = "Gumbel-180"
    GUMBEL_270 = "Gumbel-270"
    CLAYTON_90 = "Clayton-90"
    CLAYTON_180 = "Clayton-180"
    CLAYTON_270 = "Clayton-270"

    def is_gumbel_type(self):
        return self in [ArchCopulaTypes.GUMBEL, ArchCopulaTypes.GUMBEL_90, ArchCopulaTypes.GUMBEL_180, ArchCopulaTypes.GUMBEL_270]
    
    def is_clayton_type(self):
        return self in [ArchCopulaTypes.CLAYTON, ArchCopulaTypes.CLAYTON_90, ArchCopulaTypes.CLAYTON_180, ArchCopulaTypes.CLAYTON_270]
    
    def is_frank_type(self):
        return self == ArchCopulaTypes.FRANK

    def __str__(self):
        return self.name


def get_arch_type(family_str):
    family_str = family_str.strip().lower()
    if "gumbel" in family_str:
        if "180" in family_str:
            return ArchCopulaTypes.GUMBEL_180
        elif "90" in family_str:
            return ArchCopulaTypes.GUMBEL_90
        elif "270" in family_str:
            return ArchCopulaTypes.GUMBEL_270
        else:
            return ArchCopulaTypes.GUMBEL
    elif "clayton" in family_str:
        if "180" in family_str:
            return ArchCopulaTypes.CLAYTON_180
        elif "90" in family_str:
            return ArchCopulaTypes.CLAYTON_90
        elif "270" in family_str:
            return ArchCopulaTypes.CLAYTON_270
        else:
            return ArchCopulaTypes.CLAYTON
    elif "frank" in family_str:
        return ArchCopulaTypes.FRANK
    else:
        raise ValueError(f"Unknown family type: {family_str}")