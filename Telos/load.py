"""Load the published Telos-Small checkpoint."""

import json
import os
import urllib.request

import torch
from safetensors.torch import load_file

from Telos.telos_src.configuration_telos import TelosConfig
from Telos.telos_src.modeling_telos import TelosModel


ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(ROOT, "weights", "model.safetensors")
CONFIG_PATH = os.path.join(ROOT, "telos_src", "config.json")
WEIGHTS_URL = "https://huggingface.co/magilabsco/Telos-Small/resolve/main/model.safetensors"


def ensure_weights():
    """Download the 50.1M checkpoint the first time it is needed."""
    if os.path.isfile(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 1_000_000:
        return WEIGHTS_PATH
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    print("Downloading Telos-Small weights from {}".format(WEIGHTS_URL), flush=True)
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    return WEIGHTS_PATH


def load_telos(device):
    """Telos-Small on `device`, weights loaded, set to eval."""
    ensure_weights()
    with open(CONFIG_PATH) as handle:
        config = TelosConfig(**json.load(handle))
    model = TelosModel(config)
    missing, unexpected = model.load_state_dict(load_file(WEIGHTS_PATH), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Telos checkpoint did not match the architecture "
            "(missing {}, unexpected {})".format(len(missing), len(unexpected))
        )
    model.to(device)
    model.eval()
    return model


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def cuda_device():
    """First visible CUDA device. Select the physical card with CUDA_VISIBLE_DEVICES."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    print("Using {}".format(torch.cuda.get_device_name(device)), flush=True)
    return device
