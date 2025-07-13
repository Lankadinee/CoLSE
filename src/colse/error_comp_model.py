import torch.nn as nn
from loguru import logger
from rich.console import Console
from rich.table import Table

class HiddenLayer(nn.Module):
    def __init__(self, input_len, output_len, dropout_prob=None, freeze=False):
        super().__init__()
        layers = [nn.Linear(input_len, output_len), nn.ReLU(inplace=True)]

        # Add dropout only if dropout_prob is not None
        if dropout_prob is not None:
            layers.append(nn.Dropout(p=dropout_prob))

        self.layer = nn.Sequential(*layers)
        if freeze:
            for param in self.layer.parameters():
                param.requires_grad = False

    def forward(self, X):
        return self.layer(X)


class ErrorCompModel(nn.Module):
    def __init__(self, input_len, hid_units, output_len=1, dropout_prob=None, freeze_layer_count=0):
        super().__init__()
        total_layers = 0
        self.hid_units = hid_units

        self.hid_layers = nn.Sequential()
        for l, out_len in enumerate([int(u) for u in hid_units.split("_")]):
            self.hid_layers.add_module(
                "layer_{}".format(l), HiddenLayer(input_len, out_len, dropout_prob, freeze_layer_count > l)
            )
            input_len = out_len
            # logger.info(f"Layer {l+1} input_len: {input_len:>3} | output_len: {out_len:>3} | dropout_prob: {dropout_prob} | frozen: {freeze_layer_count > l}")
            total_layers += 1

        self.final = nn.Linear(input_len, output_len)
        if freeze_layer_count > total_layers:
            for param in self.final.parameters():
                param.requires_grad = False

    def forward(self, X):
        mid_out = self.hid_layers(X)
        pred = self.final(mid_out)

        return pred

    def name(self):
        return f"lwnn_hid{self.hid_units}"

    def has_trainable_parameters(self):
        return any(param.requires_grad for param in self.parameters())

    def show_architecture(self):
        table = Table(title="Model Architecture")
        table.add_column("Name", justify="left")
        table.add_column("Shape", justify="left")
        table.add_column("Trainable", justify="left")
        table.add_column("Num Params", justify="left")
        
        for name, param in self.named_parameters():
            is_trainable = param.requires_grad
            shape = f"{param.shape}"
            name = name.replace(".", "_")
            num_params = param.numel()
            table.add_row(name, shape, str(is_trainable), str(num_params))
        console = Console()
        console.print(table)

if __name__ == "__main__":
    input_feature_len = 21
    model = ErrorCompModel(
        input_feature_len, "256_256_128_64", output_len=2, dropout_prob=0.5
    )
    print(model)
    print(model.name())
    model.show_architecture()
    """print model structure"""

    # from torchsummary import summary
    # summary(model, (input_feature_len,))
