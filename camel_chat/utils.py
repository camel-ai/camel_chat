import json
import logging
import logging.handlers
import os
import platform
import sys
import warnings
from asyncio import AbstractEventLoop
from typing import AsyncGenerator, Generator

import requests
import torch

from camel_chat.constants import LOGDIR

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        if sys.version_info[1] >= 9:
            # This is for windows
            logging.basicConfig(level=logging.INFO, encoding="utf-8")
        else:
            if platform.system() == "Windows":
                warnings.warn(
                    "If you are running on Windows, "
                    "we recommend you use Python >= 3.9 for UTF-8 encoding."
                )
            logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True, encoding="utf-8"
        )
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                encoded_message = line.encode("utf-8", "ignore").decode("utf-8")
                self.logger.log(self.log_level, encoded_message.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            encoded_message = self.linebuf.encode("utf-8", "ignore").decode("utf-8")
            self.logger.log(self.log_level, encoded_message.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
    }
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


# Flan-t5 trained with HF+FSDP saves corrupted  weights for shared embeddings,
# Use this function to make sure it can be correctly loaded.
def clean_flant5_ckpt(ckpt_path):
    index_file = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
    index_json = json.load(open(index_file, "r"))

    weightmap = index_json["weight_map"]

    share_weight_file = weightmap["shared.weight"]
    share_weight = torch.load(os.path.join(ckpt_path, share_weight_file))[
        "shared.weight"
    ]

    for weight_name in ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]:
        weight_file = weightmap[weight_name]
        weight = torch.load(os.path.join(ckpt_path, weight_file))
        weight[weight_name] = share_weight
        torch.save(weight, os.path.join(ckpt_path, weight_file))


def pretty_print_semaphore(semaphore):
    """Print a semaphore in better format."""
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


"""A javascript function to get url parameters for the gradio web server."""
get_window_url_params_js = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log("url_params", url_params);
    return url_params;
    }
"""


def iter_over_async(
    async_gen: AsyncGenerator, event_loop: AbstractEventLoop
) -> Generator:
    """
    Convert async generator to sync generator

    :param async_gen: the AsyncGenerator to convert
    :param event_loop: the event loop to run on
    :returns: Sync generator
    """
    ait = async_gen.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    while True:
        done, obj = event_loop.run_until_complete(get_next())
        if done:
            break
        yield obj


def detect_language(text: str) -> str:
    """Detect the langauge of a string."""
    import polyglot  # pip3 install polyglot pyicu pycld2
    import pycld2
    from polyglot.detect import Detector
    from polyglot.detect.base import logger as polyglot_logger

    polyglot_logger.setLevel("ERROR")

    try:
        lang_code = Detector(text).language.name
    except (pycld2.error, polyglot.detect.base.UnknownLanguage):
        lang_code = "unknown"
    return lang_code


def parse_gradio_auth_creds(filename):
    """Parse a username:password file for gradio authorization."""
    gradio_auth_creds = []
    with open(filename, "r", encoding="utf8") as file:
        for line in file.readlines():
            gradio_auth_creds += [x.strip() for x in line.split(",") if x.strip()]
    if gradio_auth_creds:
        auth = [tuple(cred.split(":")) for cred in gradio_auth_creds]
    else:
        auth = None
    return auth
