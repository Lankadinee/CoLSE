import os
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from colse.custom_data_generator import CustomDataGen
from colse.dataset_names import DatasetNames
from colse.error_comp_model import ErrorCompModel
from colse.res_utils import decode_label, encode_label, multiply_pairs_norm


logger.level(os.getenv("LOG_LEVEL", "INFO"))

# Set device to GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ErrorCompensationNetwork:
    def __init__(self, model_path: str, dataset: CustomDataGen, output_len: int = 3):
        logger.info(f"Loading model from - {model_path}")
        # Load model state from file
        state = torch.load(model_path, map_location=DEVICE, weights_only=False)
        # Initialize model architecture
        self.model = ErrorCompModel(
            state["fea_num"], "256_256_128_64", output_len=3
        ).to(DEVICE)
        self.output_len = output_len
        logger.info(
            f"Overall Error Compensation model size = {state['model_size']:.2f}MB"
        )
        # Load model weights
        self.model.load_state_dict(state["model_state_dict"])
        # Store normalization parameters
        # TODO: Store these three values in the model state dict in the training stage
        self.max_values = dataset.scaler.data_max_
        self.min_values = dataset.scaler.data_min_
        self.no_of_rows = dataset.no_of_rows
        # Prepare double indices for normalization
        indices = np.arange(len(self.min_values) * 2) // 2
        self.min_values_double = self.min_values[indices]
        self.diff = self.max_values - self.min_values
        self.diff_double = self.diff[indices]
        self.diff_double[self.diff_double == 0] = 1

    def report_model(self, blacklist=None):
        ps = []
        # Count parameters, skipping those in blacklist
        for name, p in self.model.named_parameters():
            if blacklist is None or blacklist not in name:
                ps.append(np.prod(p.size()))
        num_params = sum(ps)
        mb = num_params * 4 / 1024 / 1024
        logger.info(f"Number of model parameters: {num_params} (~= {mb:.2f}MB)")
        return mb

    def pre_process(self, query, cdf, y_bar):
        try:
            # Vectorized normalization - much faster than list comprehension
            q_np = (
                query.flatten() if hasattr(query, "flatten") else np.array(query).flatten()
            )
            # Create index array for min/max values (each pair uses same index)
            # logger.debug(f"q_np: {q_np}")
            # logger.debug(f"self.min_values_double: {self.min_values_double}")
            # logger.debug(f"self.diff_double: {self.diff_double}")
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

            # Concatenate normalized query, AVI, and y_bar for model input
            x = np.concatenate([norm_q.flatten(), [avi_card_log], [y_bar_log]])

            # return torch.tensor(x, dtype=torch.float32).to(DEVICE)
            return torch.tensor(x, dtype=torch.float32).to(DEVICE)
        except Exception as e:
            logger.error(f"query: {query}")
            logger.error(f"cdf: {cdf}")
            logger.error(f"y_bar: {y_bar}")
            logger.error(f"Error in pre_process: {e}")
            raise e

    def post_process(self, y_pred, y_bar):
        # 1. Get absolute prediction (third output)
        valid_preds_np = F.relu(y_pred[:, 2]).detach().cpu().numpy()
        v_preds_abs_d = np.maximum(np.round(decode_label(valid_preds_np)), 0.0)

        # 2. Sign logic
        positive_sign_logits = y_pred[:, 0]
        negative_sign_logits = y_pred[:, 1]

        # Determine sign of prediction
        if positive_sign_logits > negative_sign_logits:
            valid_preds_sign = 1
        elif positive_sign_logits < negative_sign_logits:
            valid_preds_sign = -1
        else:
            valid_preds_sign = 0

        # Denormalize y_bar
        y_bar_actual = y_bar * self.no_of_rows
        y_bar_rounded = np.maximum(np.round(y_bar_actual), 0.0)

        # 4. Final prediction
        selectivity = (
            v_preds_abs_d * valid_preds_sign + y_bar_rounded
        ) / self.no_of_rows
        return selectivity

    def inference(self, query, cdf, y_bar):
        # Preprocess input, predict, and postprocess output
        x = self.pre_process(query, cdf, y_bar)
        y_pred = self.predict(x)
        y = self.post_process(y_pred, y_bar)
        return y

    def predict(self, x):
        # Run model forward pass
        return self.model(x).reshape(-1, self.output_len)
