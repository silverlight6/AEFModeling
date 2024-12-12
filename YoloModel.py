import torch


class YoloModel(torch.nn.Module):
    def __init__(self,) -> torch.nn.Module:
        super().__init__()

        def feature_encoder(input_size, feature_layer_sizes, feature_output_size):
            return torch.nn.Sequential(
                AlternateFeatureEncoder(input_size, feature_layer_sizes, feature_output_size, 'cuda'),
                Normalize()
            )

        self.encoder = feature_encoder(97, [512, 512], 256)

        self.neck = ResidualBlock(torch.nn.Sequential(feature_encoder(256, [512, 512], 256),
                                                      feature_encoder(256, [512, 512], 256),
                                                      feature_encoder(256, [512, 512], 256)))

        self.head = feature_encoder(256, [512, 512], 2)
        self.to('cuda')

    def forward(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.head(x)

        return x


class AlternateFeatureEncoder(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, device, dropout_rate=0.1, use_layer_norm=True):
        super(AlternateFeatureEncoder, self).__init__()

        layers = []
        current_size = input_size

        # Add hidden layers
        for size in layer_sizes:
            layers.append(torch.nn.Linear(current_size, size))
            layers.append(torch.nn.ReLU())

            # Optionally add layer normalization
            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(size))

            # Optionally add dropout
            layers.append(torch.nn.Dropout(dropout_rate))

            current_size = size

        # Final layer
        layers.append(torch.nn.Linear(current_size, output_size))

        self.network = torch.nn.Sequential(*layers).to(device)
        self.device = device

    def forward(self, x):
        # x = x.to(self.device)  # Ensure the input is on the correct device
        return self.network(x)

def normalize(x):
    min_encoded_state = x.min(1, keepdim=True)[0]
    max_encoded_state = x.max(1, keepdim=True)[0]
    scale_encoded_state = max_encoded_state - min_encoded_state
    scale_encoded_state = torch.where(scale_encoded_state < 1e-5, scale_encoded_state + 1e-5, scale_encoded_state)
    # scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
    encoded_state_normalized = (
        x - min_encoded_state
    ) / scale_encoded_state
    return encoded_state_normalized

# Turn normalize into a torch layer
class Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return normalize(x)


class ResidualBlock(torch.nn.Module):
    """
    A simple residual block with a skip connection.
    """
    def __init__(self, block, norm_size=0, local_norm=False):
        super(ResidualBlock, self).__init__()
        self.block = block
        self.normalize = local_norm
        if normalize:
            self.batch_norm = torch.nn.BatchNorm1d(norm_size)

    def forward(self, x):
        x = x + self.block(x)
        if self.normalize:
            x = self.batch_norm(x)
        return x
