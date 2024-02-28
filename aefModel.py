import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class AefModel(torch.nn.Module):
    # I will fill in more comments later on when I finalize this section of the code.
    # Mostly a phase 2 and 3 part of the project.
    def __init__(self, config):
        super().__init__()

        image_height, image_width = pair((config.max_timestep, config.sensor_count))
        patch_height, patch_width = pair((config.sensor_patch_height, config.sensor_patch_width))

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = config.sensor_channels * patch_height * patch_width

        dim = config.hidden_state_size

        self.to_patch_embedding = torch.nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            torch.nn.LayerNorm(patch_dim),
            torch.nn.Linear(patch_dim, dim),
            torch.nn.LayerNorm(dim),
        )

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = torch.nn.Dropout(config.embd_pdrop)

        # Note to self, I can probably make this transformer bigger.
        self.transformer = Transformer(dim, config.num_hidden_layers, config.num_heads, config.hidden_state_size,
                                       config.hidden_state_size * 2, 0.1)

        self.pool = 'cls'
        self.to_latent = torch.nn.Identity()

        self.dynamics_network = DynNetwork(input_size=config.hidden_state_size,
                                           num_layers=config.num_rnn_cells,
                                           hidden_size=config.lstm_size,
                                           config=config)
        self.config = config

    def forward(self, sensor_input: torch.Tensor, prev_hidden_state: torch.tensor) -> (torch.Tensor, torch.Tensor):
        sensor_input = torch.unsqueeze(sensor_input, dim=1)
        x = self.to_patch_embedding(sensor_input)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        hidden_state = self.to_latent(x)
        hidden_state, logits = self.dynamics_network(hidden_state, prev_hidden_state)

        return {"prediction": logits, "hidden_state": hidden_state}

    def configure_optimizers(self, train_config):
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 3000, 10000], gamma=0.1)
        return optimizer, scheduler


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

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# Turn normalize into a torch layer
class Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return normalize(x)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = torch.nn.LayerNorm(dim)

        self.attend = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
