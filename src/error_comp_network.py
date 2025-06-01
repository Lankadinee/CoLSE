import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


from colse.custom_data_generator import CustomDataGen
from colse.dataset_names import DatasetNames
from colse.error_comp_model import ErrorCompModel
from colse.res_utils import decode_label, encode_label, multiply_pairs_norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ErrorCompensationNetwork:
    def __init__(self, model_path, dataset, output_len=3):
        logger.info(f"Loading model from - {model_path}")
        state = torch.load(model_path, map_location=DEVICE, weights_only=False)
        self.model = ErrorCompModel(
            state["fea_num"], "256_256_128_64", output_len=3
        ).to(DEVICE)
        self.output_len = output_len
        logger.info(f"Overall LWNN model size = {state['model_size']:.2f}MB")
        self.model.load_state_dict(state["model_state_dict"])
        self.max_values = dataset.scaler.data_max_
        self.min_values = dataset.scaler.data_min_
        indices = np.arange(len(self.min_values) * 2) // 2
        self.min_values_double = self.min_values[indices]
        self.diff = self.max_values - self.min_values
        self.diff_double = self.diff[indices]
        self.no_of_rows = dataset.no_of_rows

    def report_model(self, blacklist=None):
        ps = []
        for name, p in self.model.named_parameters():
            if blacklist is None or blacklist not in name:
                ps.append(np.prod(p.size()))
        num_params = sum(ps)
        mb = num_params * 4 / 1024 / 1024
        logger.info(f"Number of model parameters: {num_params} (~= {mb:.2f}MB)")
        return mb

    def pre_process(self, query, cdf, y_bar):
        # Vectorized normalization - much faster than list comprehension
        q_np = (
            query.flatten() if hasattr(query, "flatten") else np.array(query).flatten()
        )
        # Create index array for min/max values (each pair uses same index)
        norm_q = (q_np - self.min_values_double) / self.diff_double

        # norm_q = (query - self.min_values) / self.diff
        norm_q[norm_q == -np.inf] = 0
        norm_q[norm_q == np.inf] = 1

        # Log AVI estimate
        avi_card = multiply_pairs_norm(cdf) * self.no_of_rows
        avi_card_log = encode_label(avi_card)

        # Log y_bar
        y_bar_ranged = np.clip(y_bar, 0, 1)
        y_bar_log = encode_label(y_bar_ranged * self.no_of_rows)

        x = np.concatenate([norm_q.flatten(), [avi_card_log], [y_bar_log]])

        # return torch.tensor(x, dtype=torch.float32).to(DEVICE)
        return torch.tensor(x, dtype=torch.float32).to(DEVICE)

    def post_process(self, y_pred, y_bar):
        # 1. Get absolute prediction (third output)
        valid_preds_np = F.relu(y_pred[:, 2]).detach().cpu().numpy()
        v_preds_abs_d = np.maximum(np.round(decode_label(valid_preds_np)), 0.0)

        # 2. Sign logic
        positive_sign_logits = y_pred[:, 0]
        negative_sign_logits = y_pred[:, 1]

        if positive_sign_logits > negative_sign_logits:
            valid_preds_sign = 1
        elif positive_sign_logits < negative_sign_logits:
            valid_preds_sign = -1
        else:
            valid_preds_sign = 0

        y_bar_actual = y_bar * self.no_of_rows

        # 4. Final prediction
        selectivity = (
            v_preds_abs_d * valid_preds_sign + y_bar_actual
        ) / self.no_of_rows
        return selectivity

    def inference(self, query, cdf, y_bar):
        x = self.pre_process(query, cdf, y_bar)
        y_pred = self.predict(x)
        y = self.post_process(y_pred, y_bar)
        return y

    def predict(self, x):
        return self.model(x).reshape(-1, self.output_len)



if __name__ == "__main__":
    from colse.data_path import get_model_path
    dataset_type = DatasetNames("forest")
    model_file = get_model_path() / f"error_comp_model.pt"
    dataset = CustomDataGen(
        no_of_rows=None,
        no_of_queries=None,
        dataset_type=dataset_type,
        data_split="test",
        selected_cols=None,
        scalar_type="min_max",  # 'min_max' or 'standard
        dequantize=False,
        seed=1,
        is_range_queries=True,
        verbose=False,
    )
    error_comp_model_path = get_model_path(dataset_type.value) / "error_comp_model.pt"
    error_comp_model = ErrorCompensationNetwork(error_comp_model_path, dataset)