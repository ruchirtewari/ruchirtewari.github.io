"""Shared utilities for the interpretability book programs.

Every program imports from here:
  - get_arg_parser / init      : uniform CLI (--seed, --outdir, --quick, --show)
  - set_seed, savefig          : reproducibility and figure output
  - load_shakespeare           : cached public-domain corpus for ch1/ch3
  - LLM                        : modular text-generation backend
                                 (ollama | openai_compatible | hf_pipeline),
                                 selected in config.json — swap the backend for
                                 ALL programs by editing one file
  - load_causal_lm, get_blocks : HuggingFace model loading with a uniform view
                                 of the transformer block list (gpt2 / llama /
                                 qwen / gemma layouts)
  - ResidualCapture            : context manager caching residual-stream
                                 activations per layer
  - add_vector_hook,
    projection_ablation_hooks  : steering interventions

Programs that only need text generations (judging, labeling, chat) go through
LLM and therefore run against a local ollama server or any remote
OpenAI-compatible endpoint. Programs that need *internal access* (activations,
hooks, steering) must run a local HuggingFace model regardless of the LLM
backend — internals are not available over an API.
"""

import argparse
import json
import os
import random
import re
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CONFIG_PATH = ROOT / "config.json"


# ----------------------------------------------------------------- config ---

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


# -------------------------------------------------------------------- CLI ---

def get_arg_parser(description=""):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--outdir", type=str, default=str(ROOT / "out"),
                   help="base directory for figures/artifacts")
    p.add_argument("--quick", action="store_true",
                   help="reduced steps/samples for a fast smoke run")
    p.add_argument("--show", action="store_true",
                   help="also display figures interactively")
    return p


def init(args, program_name):
    """Set seeds, create and return this program's output directory."""
    set_seed(args.seed)
    outdir = Path(args.outdir) / program_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def savefig(fig, outdir, fname, show=False):
    path = Path(outdir) / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(fig)


# ------------------------------------------------------------------- data ---

SHAKESPEARE_URL = "https://www.gutenberg.org/files/100/100-0.txt"


def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_shakespeare(letters_only=True):
    """Complete works of Shakespeare, cached. Optionally reduced to A-Z + space."""
    raw_path = download(SHAKESPEARE_URL, CACHE / "shakespeare.txt")
    text = raw_path.read_text(encoding="utf-8", errors="ignore")
    start = text.find("*** START")
    end = text.find("*** END")
    if start != -1 and end != -1:
        text = text[text.find("\n", start): end]
    if letters_only:
        text = text.upper()
        text = re.sub(r"[^A-Z]+", " ", text)
        text = re.sub(r" +", " ", text).strip()
    return text


# ---------------------------------------------------------- LLM (modular) ---

class LLM:
    """Text-generation backend chosen by config.json: 'ollama' (default,
    local), 'openai_compatible' (any remote /v1/chat/completions endpoint),
    or 'hf_pipeline' (in-process transformers).

    Usage:  llm = LLM();  reply = llm.chat("label this feature: ...")
    """

    def __init__(self, config=None):
        self.cfg = config or load_config()
        self.backend = self.cfg.get("llm_backend", "ollama")
        self._pipe = None  # lazy hf pipeline

    def chat(self, prompt, system=None, max_tokens=512, temperature=0.7):
        if self.backend == "ollama":
            return self._ollama(prompt, system, max_tokens, temperature)
        if self.backend == "openai_compatible":
            return self._openai(prompt, system, max_tokens, temperature)
        if self.backend == "hf_pipeline":
            return self._hf(prompt, system, max_tokens, temperature)
        raise ValueError(f"unknown llm_backend: {self.backend}")

    # -- backends ---------------------------------------------------------

    def _post_json(self, url, payload, headers=None):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json",
                                              **(headers or {})})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())

    def _messages(self, prompt, system):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _ollama(self, prompt, system, max_tokens, temperature):
        c = self.cfg["ollama"]
        out = self._post_json(
            f"{c['base_url']}/api/chat",
            {"model": c["model"], "messages": self._messages(prompt, system),
             "stream": False,
             "options": {"num_predict": max_tokens, "temperature": temperature}})
        return out["message"]["content"]

    def _openai(self, prompt, system, max_tokens, temperature):
        c = self.cfg["openai_compatible"]
        key = os.environ.get(c.get("api_key_env", "OPENAI_API_KEY"), "")
        out = self._post_json(
            f"{c['base_url']}/v1/chat/completions",
            {"model": c["model"], "messages": self._messages(prompt, system),
             "max_tokens": max_tokens, "temperature": temperature},
            headers={"Authorization": f"Bearer {key}"})
        return out["choices"][0]["message"]["content"]

    def _hf(self, prompt, system, max_tokens, temperature):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-generation",
                                  model=self.cfg["hf_pipeline"]["model"],
                                  device_map=self.cfg.get("device", "auto"))
        msgs = self._messages(prompt, system)
        out = self._pipe(msgs, max_new_tokens=max_tokens,
                         do_sample=temperature > 0, temperature=max(temperature, 1e-5))
        return out[0]["generated_text"][-1]["content"]

    def available(self):
        """True if the configured backend responds (used to skip LLM extras)."""
        try:
            self.chat("Say OK.", max_tokens=5)
            return True
        except Exception as e:
            print(f"[llm] backend '{self.backend}' unavailable: {e}")
            return False


# ------------------------------------------------- HF models + internals ---

def resolve_device(cfg=None):
    import torch
    pref = (cfg or load_config()).get("device", "auto")
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_causal_lm(name=None, device=None, attn_eager=False):
    """Load a HuggingFace causal LM for internal access. Returns
    (model, tokenizer, device). name defaults to config 'hf_base_model'.

    Set attn_eager=True when you need attention weights (output_attentions):
    the default sdpa/flash backends in recent transformers do not return them.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = load_config()
    name = name or cfg["hf_base_model"]
    device = device or resolve_device(cfg)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=torch.float32 if device == "cpu" else torch.float16)
    if attn_eager:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(device).eval()
    return model, tok, device


def get_blocks(model):
    """Uniform view of the transformer block list across architectures."""
    for path in ("transformer.h",      # gpt2
                 "model.layers",       # llama / qwen / mistral / gemma
                 "gpt_neox.layers"):   # pythia
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"unknown architecture: {type(model).__name__}")


def _block_output_hidden(out):
    return out[0] if isinstance(out, tuple) else out


class ResidualCapture:
    """Cache residual-stream activations (block outputs) for chosen layers.

    with ResidualCapture(model, layers=[6]) as cap:
        model(**inputs)
    acts = cap.acts[6]   # tensor (batch, seq, d_model)
    """

    def __init__(self, model, layers=None):
        self.blocks = get_blocks(model)
        self.layers = list(range(len(self.blocks))) if layers is None else layers
        self.acts = {}
        self._handles = []

    def __enter__(self):
        for i in self.layers:
            def hook(mod, inp, out, i=i):
                self.acts[i] = _block_output_hidden(out).detach()
            self._handles.append(self.blocks[i].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def add_vector_hook(model, layer, vec, alpha=1.0):
    """h <- h + alpha*vec at one layer's output, every position.
    Returns the handle; call handle.remove() to stop steering."""
    import torch
    blocks = get_blocks(model)

    def hook(mod, inp, out):
        h = _block_output_hidden(out)
        h = h + alpha * vec.to(h.dtype).to(h.device)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    return blocks[layer].register_forward_hook(hook)


def projection_ablation_hooks(model, vec):
    """h <- h - (h.v̂)v̂ at EVERY layer output. Returns list of handles."""
    import torch
    v = vec / vec.norm()
    handles = []

    def hook(mod, inp, out):
        h = _block_output_hidden(out)
        vv = v.to(h.dtype).to(h.device)
        h = h - (h @ vv).unsqueeze(-1) * vv
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    for b in get_blocks(model):
        handles.append(b.register_forward_hook(hook))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()
