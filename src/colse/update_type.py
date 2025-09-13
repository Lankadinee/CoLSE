from enum import Enum


class UpdateTypes(str, Enum):
    IND_005 = "ind_0.05"
    IND_01 = "ind_0.10"
    IND_015 = "ind_0.15"
    IND_02 = "ind_0.2"
    COR_005 = "cor_0.05"
    COR_01 = "cor_0.10"
    COR_015 = "cor_0.15"
    COR_02 = "cor_0.2"
    SKEW_005 = "skew_0.05"
    SKEW_01 = "skew_0.10"
    SKEW_015 = "skew_0.15"
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


class CustomUpdateTypes(str, Enum):
    NO_OF_TRAINING_QUERIES = "notq"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value
