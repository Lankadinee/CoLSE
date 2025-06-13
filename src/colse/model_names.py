


from enum import Enum


class ModelNames(str, Enum):
    COLSE = "dvine"
    MSCN = "mscn"
    LWNN = "lwnn"
    LWTREE = "lwtree"
    DEEP_DB = "spn"
    NARU = "resmade"
    
    def is_ours(self):
        return self is ModelNames.COLSE
    
    def __repr__(self):
        return self.value
    
    def __str__(self):
        return self.value

if __name__ == "__main__":
    # loop through all the model names and print the name and the value
    for model_name in ModelNames:
        print(model_name, model_name.value)