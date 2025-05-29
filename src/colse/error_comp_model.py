import torch.nn as nn


class HiddenLayer(nn.Module):
    def __init__(self, input_len, output_len, dropout_prob=None):
        super().__init__()
        layers = [nn.Linear(input_len, output_len), nn.ReLU(inplace=True)]

        # Add dropout only if dropout_prob is not None
        if dropout_prob is not None:
            layers.append(nn.Dropout(p=dropout_prob))

        self.layer = nn.Sequential(*layers)

    def forward(self, X):
        return self.layer(X)


class ErrorCompModel(nn.Module):
    def __init__(self, input_len, hid_units, output_len=1, dropout_prob=None):
        super().__init__()
        self.hid_units = hid_units

        self.hid_layers = nn.Sequential()
        for l, out_len in enumerate([int(u) for u in hid_units.split("_")]):
            self.hid_layers.add_module(
                "layer_{}".format(l), HiddenLayer(input_len, out_len, dropout_prob)
            )
            input_len = out_len

        self.final = nn.Linear(input_len, output_len)

    def forward(self, X):
        mid_out = self.hid_layers(X)
        pred = self.final(mid_out)

        return pred

    def name(self):
        return f"lwnn_hid{self.hid_units}"


if __name__ == "__main__":
    input_feature_len = 21
    model = ErrorCompModel(
        input_feature_len, "256_256_128_64", output_len=2, dropout_prob=0.5
    )
    print(model)
    print(model.name())
    """print model structure"""

    # from torchsummary import summary
    # summary(model, (input_feature_len,))
