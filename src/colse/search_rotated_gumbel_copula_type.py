
"""
Reference - https://chatgpt.com/share/6710fe42-ae68-800d-a7ba-e26adadda78d
"""

from enum import Enum
import numpy as np

from colse.copula_types import ArchCopulaTypes

class GumbelCopulaSearchMethods(Enum):
    TAIL_DEPENDENCE = 1

class SearchGumbelCopulaType:
    def __init__(self, method=None) -> None:
        self.method = GumbelCopulaSearchMethods.TAIL_DEPENDENCE if method is None else method

    def predict(self, x, y):
        match self.method:
            case GumbelCopulaSearchMethods.TAIL_DEPENDENCE:
                return self.tail_dep(x, y)
            case _:
                raise ValueError(f"Invalid copula search method: {self.method}")
    

    def tail_dep(self, x, y):
        upper_tail_dependence = self._tail_dependence(x, y, tail='upper')
        lower_tail_dependence = self._tail_dependence(x, y, tail='lower')

        # print(f"Upper Tail Dependence: {upper_tail_dep}")
        # print(f"Lower Tail Dependence: {lower_tail_dep}")

        # Rewriting the code after execution reset


        # Thresholds for determining significant tail dependence
        high_tail_threshold = 0.2  # You can adjust this threshold based on context

        # Decision based on tail dependence
        if upper_tail_dependence > high_tail_threshold and lower_tail_dependence <= high_tail_threshold:
            return ArchCopulaTypes.GUMBEL #(strong upper tail dependence)
        elif lower_tail_dependence > high_tail_threshold and upper_tail_dependence <= high_tail_threshold:
            return ArchCopulaTypes.GUMBEL_180 #(strong lower tail dependence)
        elif upper_tail_dependence > high_tail_threshold and lower_tail_dependence > high_tail_threshold:
        # Determine whether it’s 90° or 270° based on whether upper or lower tail is stronger
            if upper_tail_dependence > lower_tail_dependence:
                return ArchCopulaTypes.GUMBEL_90 #(upper tail dependence for one variable, lower tail for the other)
            else:
                return ArchCopulaTypes.GUMBEL_270 #(lower tail dependence for one variable, upper tail for the other)
        else:
            return ArchCopulaTypes.GUMBEL


    def _tail_dependence(self, x, y, tail='upper', threshold=0.95):
        if tail == 'upper':
            # Upper tail dependence
            X_thresh = np.quantile(x, threshold)
            Y_thresh = np.quantile(y, threshold)
            return np.mean((x > X_thresh) & (y > Y_thresh))
        else:
            # Lower tail dependence
            X_thresh = np.quantile(x, 1 - threshold)
            Y_thresh = np.quantile(y, 1 - threshold)
            return np.mean((x < X_thresh) & (y < Y_thresh))


if __name__ == "__main__":
    
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    search_copula = SearchGumbelCopulaType()
    copula_type = search_copula.predict(x, y)
    print(copula_type) # Output: CopulaTypes.FRANK