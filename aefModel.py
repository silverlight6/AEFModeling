import torch

class AefModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_state_size, nhead=config.num_heads)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

        layer_sizes = [config.hidden_state_size // 2] * config.mlp_layers
        self.sensor_encoder_cnn1 = ConvBlock(1, 64)
        self.sensor_encoder_cnn2 = ConvBlock(64, 64)
        self.sensor_encoder_cnn3 = ConvBlock(64, 8)
        self.sensor_flattener = torch.nn.Flatten()
        self.sensor_encoder_mlp = mlp(4000, layer_sizes, config.hidden_state_size,
                                      torch.nn.GELU, torch.nn.GELU)

        self.pos_emb = torch.nn.Parameter(torch.zeros(1, config.hidden_state_size))
        self.drop = torch.nn.Dropout(config.embd_pdrop)

        self.dynamics_network = DynNetwork(input_size=config.hidden_state_size,
                                           num_layers=config.num_rnn_cells,
                                           hidden_size=config.lstm_size,
                                           config=config)
        self.config = config

    def forward(self, sensor_input: torch.Tensor, prev_hidden_state: torch.tensor) -> (torch.Tensor, torch.Tensor):
        sensor_input = torch.unsqueeze(sensor_input, dim=1)
        sensor_encoder = self.sensor_encoder_cnn1(sensor_input)
        sensor_encoder = self.sensor_encoder_cnn2(sensor_encoder)
        sensor_encoder = self.sensor_encoder_cnn3(sensor_encoder)

        sensor_encoder = self.sensor_flattener(sensor_encoder)
        sensor_encoded = self.sensor_encoder_mlp(sensor_encoder)

        sensor_encoded = self.drop(sensor_encoded + self.pos_emb)

        src_mask = torch.nn.Transformer.generate_square_subsequent_mask(len(sensor_encoded)).to('cuda')

        hidden_state = self.encoder(sensor_encoded, src_mask)
        hidden_state, logits = self.dynamics_network(hidden_state, prev_hidden_state)

        return {"prediction": logits, "hidden_state": hidden_state}

    def configure_optimizers(self, train_config):
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


def mlp(input_size,
        layer_sizes,
        output_size,
        output_activation,
        activation):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers).to('cuda')


class DynNetwork(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, config):
        super().__init__()

        def memory():
            return torch.nn.Sequential(
                MemoryLayer(input_size, num_layers, hidden_size, config),
                Normalize()
            ).to(config.device)

        self.dynamics_memory = memory()

        # 6 is the number of separate dynamic networks
        self.prediction_value_network = mlp(config.hidden_state_size,
                                            [config.hidden_state_size // 2] * config.mlp_layers,
                                            3, torch.nn.Identity, torch.nn.GELU)
        self.softmax = torch.nn.Softmax()
        self.model_config = config

    def forward(self, hidden_state, encoder_state) -> [torch.nn.Module, torch.nn.Module]:

        inputs = encoder_state[:, None, :]

        new_hidden_state = self.dynamics_memory((inputs, hidden_state))

        logits = self.prediction_value_network(new_hidden_state)
        logits = self.softmax(logits)

        return new_hidden_state, logits


class MemoryLayer(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, config):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, num_layers=num_layers,
                                  hidden_size=hidden_size, batch_first=True).to(config.device)
        self.model_config = config

    def forward(self, x):
        inputs, hidden_state = x
        lstm_state = self.flat_to_lstm_input(hidden_state)
        h0, c0 = list(zip(*lstm_state))
        _, new_nested_states = self.lstm(inputs, (torch.stack(h0, dim=0), torch.stack(c0, dim=0)))

        return self.rnn_to_flat(new_nested_states)

    def flat_to_lstm_input(self, state):
        """Maps flat token to LSTM state."""
        tensors = []
        cur_idx = 0
        for size in self.model_config.rnn_sizes:
            states = (state[Ellipsis, cur_idx:cur_idx + size],
                      state[Ellipsis, cur_idx + size:cur_idx + 2 * size])

            cur_idx += 2 * size
            tensors.append(states)
        # assert cur_idx == state.shape[-1]
        return tensors

    @staticmethod
    def rnn_to_flat(state):
        """Maps LSTM state to flat token."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return torch.cat(states, dim=-1)


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


class ConvBlock(torch.nn.Module):
    def __init__(self, input_channels, n_kernels):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(input_channels, n_kernels, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(n_kernels)
        self.conv2 = torch.nn.Conv2d(n_kernels, n_kernels, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(n_kernels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x) -> torch.nn.Module:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out)

    def __call__(self, x):
        return self.forward(x)
