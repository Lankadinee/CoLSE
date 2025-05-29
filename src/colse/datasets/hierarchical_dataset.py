from loguru import logger
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from cest.copula.h_archimedian_copula import HArchCopula
from cest.extras.storage import load_object, save_object
from tqdm import tqdm


class NINOUT:
    def __init__(self, n_type=None, no_of_cols=None, previous_pair=None) -> None:
        self.n_type = n_type
        self.no_of_cols = no_of_cols
        self.previous_pair = previous_pair

        self.cdf_1 = None
        self.cdf_2 = None
        self.range_ub_1 = None
        self.range_ub_2 = None
        self.range_lb_1 = None
        self.range_lb_2 = None
        self.ce = None

    def set_values(self, X, col_pair, lb, ub, result_1=None, result_2=None, cdf_1=None, cdf_2=None):
        no_of_rows = X.shape[0]
        no_of_cols = X.shape[1]

        if result_1 is None and cdf_1 is None:
            x_col_1 = X[:, col_pair[0]]
            self.range_lb_1 = lb[col_pair[0]]
            self.range_ub_1 = ub[col_pair[0]]
            result_1 = (self.range_lb_1 < x_col_1) * (x_col_1 < self.range_ub_1)
            self.cdf_1 = np.sum(result_1) / no_of_rows
        else:
            self.cdf_1 = cdf_1

        if result_2 is None and cdf_2 is None:
            x_col_2 = X[:, col_pair[1]]
            self.range_lb_2 = lb[col_pair[1]]
            self.range_ub_2 = ub[col_pair[1]]
            result_2 = (self.range_lb_2 < x_col_2) * (x_col_2 < self.range_ub_2)
            self.cdf_2 = np.sum(result_2) / no_of_rows
        else:
            self.cdf_2 = cdf_2

        self.result = result_1 * result_2
        self.ce = np.sum(self.result) / no_of_rows

    def get_input(self):
        return self.result, self.ce

    def clean(self):
        self.result = None


class HDataset:
    def __init__(self) -> None:

        self.no_of_cols = 5
        self.sample_count = 1000
        self.df = self.create_dataset()

        self.arch_copula = HArchCopula()
        self.arch_copula.fit(self.df.to_numpy().transpose())
        self.arch_copula.generate(no_of_cols=self.no_of_cols)

        self.query_l, self.query_r = None, None
        self.generate_queries()
        self.generate_true_values()

    @staticmethod
    def get_type(is_oob_pair):
        if is_oob_pair[0] and is_oob_pair[1]:
            return "tree"
        elif is_oob_pair[0] or is_oob_pair[1]:
            return "branch"
        else:
            return "leaf"

    def generate_true_values(self):
        self.true_card_objects = []
        remove_indexes = []

        loop = tqdm(range(self.sample_count))

        for idx in loop:
            lb = self.query_l[idx]
            ub = self.query_r[idx]
            object_dict = {}

            for k, s_pair in enumerate(self.arch_copula.selected_pairs):
                result_1 = result_2 = ce_1 = ce_2 = None

                is_oob_pair = (s_pair[0] >= self.no_of_cols, s_pair[1] >= self.no_of_cols)
                if is_oob_pair[0]:
                    result_1, ce_1 = object_dict[s_pair[0]].get_input()

                if is_oob_pair[1]:
                    result_2, ce_2 = object_dict[s_pair[1]].get_input()

                obj = NINOUT(n_type=self.get_type(is_oob_pair), no_of_cols=self.no_of_cols, previous_pair=s_pair)
                obj.set_values(self.df.to_numpy(), s_pair, lb, ub, result_1, result_2, ce_1, ce_2)
                object_dict[self.no_of_cols + k] = obj

            for key, obj in object_dict.items():
                obj.clean()

            if obj.ce > 0:
                self.true_card_objects.append(object_dict)
            else:
                remove_indexes.append(idx)
        
        self.true_card_objects = np.array(self.true_card_objects)

        if len(remove_indexes) > 0:
            self.true_card_objects = np.delete(self.true_card_objects, remove_indexes)
            self.query_l = np.delete(self.query_l, remove_indexes)
            self.query_r = np.delete(self.query_r, remove_indexes)
            self.sample_count = len(self.true_card_objects)
    
        logger.info(f"Removed {len(remove_indexes)} samples")

    def generate_queries(self):
        sample_data = self.df.sample(n=self.sample_count * 2).to_numpy().transpose()
        data_samples = np.array([[d[: self.sample_count], d[self.sample_count :]] for d in sample_data])
        data_samples = np.sort(data_samples, axis=1)
        data_samples = data_samples.transpose(1, 2, 0)
        self.query_l, self.query_r = data_samples[0], data_samples[1]

    def create_dataset(self):
        nrows = 500000
        attr1 = np.random.normal(0, 20, size=nrows)
        attr2 = np.random.normal(0, 20, size=nrows)
        attr3 = attr1 + attr2 + np.random.normal(0, 10)
        attr4 = (attr1 + np.random.normal(0, 5, size=nrows)) * 2.5 + 5
        attr5 = (attr2 + np.random.normal(0, 5, size=nrows)) * 2.5 + 5

        assert self.no_of_cols <= 5, "Number of columns should be less than or equal to 5"
        cols = [attr1, attr2, attr3, attr4, attr5]
        data = np.column_stack(tuple(cols[: self.no_of_cols]))
        df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(1, self.no_of_cols + 1)])

        """normalize dataset using standard scaler"""
        ss = StandardScaler()
        df = pd.DataFrame(ss.fit_transform(df), columns=df.columns)
        return df

    def save(self, path):
        save_object(self, path)

    @staticmethod
    def load(path, mclass):
        assert isinstance(mclass, NINOUT)
        return load_object(path)


if __name__ == "__main__":
    hd = HDataset()
    print(hd.df.head())
    hd.save("hd_v2.pkl")
    # hd = HDataset.load("hd.pkl")
    # print(hd.df.head())
