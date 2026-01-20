import torch
import subprocess
import re


def get_best_device(logger):
    if torch.cuda.is_available():
        device = torch.device(get_free_gpu())
        logger.info(f"Using gpu: {torch.cuda.get_device_name(device)}")
    # elif (
    #     getattr(torch.backends, "mps", None) is not None
    #     and torch.backends.mps.is_available()
    # ):
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using mps")
    else:
        device = torch.device("cpu")
        logger.info("Using cpu")
    return device


def get_free_gpu():
    command = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode("utf-8")

    cuda = []
    for line in command.splitlines():
        match = re.match(r"(\d+), (\d+) MiB", line)
        if match:
            gpu = int(match[1])
            free = int(match[2])
            cuda.append((gpu, free))
    if cuda:
        return f"cuda:{max(cuda, key=lambda x: x[1])[0]}"
    else:
        return "cpu"
