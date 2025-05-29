from abc import ABC, abstractmethod



class CDFBase(ABC):
    @abstractmethod
    def fit(self, X):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_range_cdf(self, lb, ub):
        pass