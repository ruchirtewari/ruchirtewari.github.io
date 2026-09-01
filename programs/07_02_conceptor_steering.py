"""Chapter 7, §7.2 — conceptor_steering.py (catalog entry 18).

Purpose: conceptors on a reservoir (the classic Jaeger demo), then
conceptor-AND vs additive vector-sum steering on GPT-2 for two-attribute
control. Part 1 drives a 500-unit echo state network with four periodic
signals, builds a conceptor C = R(R + a^-2 I)^-1 per pattern, regenerates by
a gated loop, and morphs by interpolating conceptors. Part 2 builds
conceptors for positive sentiment and formal register from GPT-2 activation
contrast sets, then steers continuations with AND(C_pos, C_formal) vs the
additive sum, scoring both attributes with embedded lexicons.

Expected result (acceptance): patterns regenerate and morph smoothly;
conceptor-AND lands in the both-attributes quadrant more reliably than the
additive vector-sum.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (get_arg_parser, init, load_causal_lm, resolve_device,
                    ResidualCapture, savefig)

# ------------------------------------------------------------- Part 1 --------
N_RES = 500


def make_reservoir(n=N_RES, spectral_radius=1.4, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (spectral_radius / eig)
    W_in = rng.standard_normal((n, 1)) * 1.2
    return W, W_in


def drive(W, W_in, signal, washout=100):
    """Run the reservoir driven by a 1-D signal; return collected states."""
    n = W.shape[0]
    x = np.zeros(n)
    states = []
    for t in range(len(signal)):
        x = np.tanh(W @ x + (W_in[:, 0] * signal[t]))
        if t >= washout:
            states.append(x.copy())
    return np.array(states)  # (T, n)


def conceptor(states, alpha):
    R = (states.T @ states) / states.shape[0]
    n = R.shape[0]
    return R @ np.linalg.inv(R + (alpha ** -2) * np.eye(n))


def fit_output(states, targets):
    """Ridge readout from states to the driving signal (for regeneration)."""
    lam = 1e-4
    n = states.shape[1]
    W_out = np.linalg.solve(states.T @ states + lam * np.eye(n),
                            states.T @ targets)
    return W_out  # (n,)


def regenerate(W, C, W_out, steps=400, seed=1):
    """Autonomous gated loop: x <- C * tanh(W x); read out with W_out."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(W.shape[0]) * 0.1
    out = []
    for _ in range(steps):
        x = C @ np.tanh(W @ x)
        out.append(W_out @ x)
    return np.array(out)


def part1(outdir, args):
    W, W_in = make_reservoir(seed=args.seed)
    L = 600 if not args.quick else 300
    t = np.arange(L)
    signals = {
        "sine-P10": np.sin(2 * np.pi * t / 10.0),
        "sine-P15": np.sin(2 * np.pi * t / 15.0),
        "sine-P7": np.sin(2 * np.pi * t / 7.0),
        "square": np.sign(np.sin(2 * np.pi * t / 12.0)),
    }
    alpha = 8.0
    conceptors, readouts = {}, {}
    for name, sig in signals.items():
        states = drive(W, W_in, sig)
        conceptors[name] = conceptor(states, alpha)
        readouts[name] = fit_output(states, sig[100:])

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    names = list(signals.keys())
    for ax, name in zip(axes.flat[:4], names):
        y = regenerate(W, conceptors[name], readouts[name])
        ax.plot(y[-80:])
        ax.set_title(f"regenerated: {name}")

    # morph between two conceptors by linear interpolation
    a_name, b_name = names[0], names[1]
    Wout_morph = 0.5 * (readouts[a_name] + readouts[b_name])
    for ax, lam in zip(axes.flat[4:6], [0.25, 0.75]):
        C = (1 - lam) * conceptors[a_name] + lam * conceptors[b_name]
        y = regenerate(W, C, Wout_morph)
        ax.plot(y[-80:])
        ax.set_title(f"morph {a_name}->{b_name} @ lam={lam}")
    fig.suptitle("Part 1: conceptor regeneration and morphing")
    savefig(fig, outdir, "conceptor_reservoir.png", show=args.show)
    print("[part1] regenerated 4 patterns + 2 morphs")


# ------------------------------------------------------------- Part 2 --------
POS_LEX = {"good", "great", "wonderful", "love", "excellent", "happy",
           "beautiful", "amazing", "delightful", "joy", "best", "fantastic",
           "brilliant", "perfect", "lovely", "enjoy", "pleased", "glad",
           "superb", "cheerful"}
NEG_LEX = {"bad", "terrible", "awful", "hate", "horrible", "sad", "ugly",
           "worst", "disappointing", "miserable", "poor", "dreadful",
           "annoying", "unpleasant", "gloomy", "angry", "boring", "painful",
           "sorrow", "grim"}
FORMAL_LEX = {"therefore", "moreover", "furthermore", "consequently",
              "regarding", "hereby", "shall", "pursuant", "aforementioned",
              "notwithstanding", "accordingly", "whom", "thus", "hence",
              "endeavor", "commence", "utilize", "regards", "sincerely",
              "kindly"}
INFORMAL_LEX = {"gonna", "wanna", "yeah", "cool", "stuff", "kinda", "lol",
                "hey", "guys", "awesome", "totally", "dude", "nope", "yep",
                "ok", "okay", "kids", "lots", "big", "fun"}

POS_SENTS = [
    "This is a wonderful and happy day full of joy.",
    "I love how great and beautiful everything looks.",
    "What an amazing, delightful, and excellent experience.",
    "The team did a fantastic and brilliant job today.",
    "I am so pleased and glad with this lovely result.",
    "Everything about it is perfect and truly superb.",
    "A cheerful morning brought the best kind of joy.",
    "The food was excellent and the service delightful.",
]
NEG_SENTS = [
    "This is a terrible and sad day full of sorrow.",
    "I hate how bad and ugly everything looks.",
    "What an awful, disappointing, and horrible experience.",
    "The team did a dreadful and miserable job today.",
    "I am so annoyed and angry with this poor result.",
    "Everything about it is worst and truly unpleasant.",
    "A gloomy morning brought the grim kind of pain.",
    "The food was awful and the service dreadful.",
]
FORMAL_SENTS = [
    "I hereby submit the aforementioned report for your review.",
    "Pursuant to our agreement, we shall commence the work.",
    "Furthermore, the committee shall convene accordingly.",
    "Regarding your inquiry, kindly find the details herein.",
    "Consequently, we endeavor to utilize the proper procedure.",
    "Moreover, the parties shall proceed as stipulated therein.",
    "Therefore, I sincerely request your prompt consideration.",
    "Notwithstanding the delay, we shall thus proceed.",
]
INFORMAL_SENTS = [
    "Hey guys, this stuff is totally awesome and fun.",
    "Yeah I'm gonna grab some cool snacks, wanna come?",
    "Nope, that's kinda boring, let's do something fun.",
    "Lol okay dude, that's totally the best idea ever.",
    "We're gonna have lots of fun with the kids today.",
    "Yep, that's pretty cool stuff, big fan honestly.",
    "Hey, wanna hang out later? It'll be awesome.",
    "Kinda tired but yeah, let's go do something fun.",
]

GEN_PROMPTS = [
    "The weather today", "My favorite meal is", "The meeting was",
    "When I got home", "The new movie", "Our vacation plans",
    "The report shows", "This morning I", "The customer said",
    "After the game",
]


def score_attr(text, pos_lex, neg_lex):
    words = [w.strip(".,!?;:").lower() for w in text.split()]
    p = sum(w in pos_lex for w in words)
    n = sum(w in neg_lex for w in words)
    return (p - n) / max(len(words), 1)


@torch.no_grad()
def layer_acts(model, tok, sents, layer, device):
    """Mean-pooled residual activation per sentence at one layer."""
    out = []
    for s in sents:
        inp = tok(s, return_tensors="pt").to(device)
        with ResidualCapture(model, layers=[layer]) as cap:
            model(**inp)
        out.append(cap.acts[layer][0].float().mean(0).cpu())
    return torch.stack(out)


def build_conceptor(acts_pos, acts_neg, alpha):
    """Conceptor from the positive class covariance (torch)."""
    R = (acts_pos.T @ acts_pos) / acts_pos.shape[0]
    n = R.shape[0]
    C = R @ torch.linalg.inv(R + (alpha ** -2) * torch.eye(n))
    return C


def conceptor_and(C1, C2):
    n = C1.shape[0]
    eye = torch.eye(n)
    inv = torch.linalg.inv
    return inv(inv(C1 + 1e-4 * eye) + inv(C2 + 1e-4 * eye) - eye)


def make_and_hook(model, layer, C, device):
    from common import get_blocks, _block_output_hidden
    blocks = get_blocks(model)
    Cd = C.to(device)

    def hook(mod, inp, out):
        h = _block_output_hidden(out)
        shp = h.shape
        flat = h.reshape(-1, shp[-1]).float()
        steered = flat @ Cd.T.to(flat.dtype)
        steered = steered.reshape(shp).to(h.dtype)
        return (steered,) + tuple(out[1:]) if isinstance(out, tuple) else steered

    return blocks[layer].register_forward_hook(hook)


@torch.no_grad()
def gen(model, tok, prompt, device, hooks=None, max_new_tokens=30):
    inp = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, inp["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def part2(outdir, args):
    try:
        model, tok, device = load_causal_lm(name="gpt2")
    except Exception as e:
        print(f"[skip part2] could not load gpt2: {e}")
        return
    n_blocks = len(model.transformer.h)
    layer = n_blocks // 2
    alpha = 4.0
    print(f"[part2] gpt2 layer={layer} device={device}")

    a_pos = layer_acts(model, tok, POS_SENTS, layer, device)
    a_neg = layer_acts(model, tok, NEG_SENTS, layer, device)
    a_form = layer_acts(model, tok, FORMAL_SENTS, layer, device)
    a_inf = layer_acts(model, tok, INFORMAL_SENTS, layer, device)

    C_pos = build_conceptor(a_pos, a_neg, alpha)
    C_form = build_conceptor(a_form, a_inf, alpha)
    C_and = conceptor_and(C_pos, C_form)

    v_pos = (a_pos.mean(0) - a_neg.mean(0))
    v_form = (a_form.mean(0) - a_inf.mean(0))
    v_sum = 5.0 * (v_pos / v_pos.norm() + v_form / v_form.norm())

    from common import add_vector_hook, remove_hooks

    prompts = GEN_PROMPTS[:5] if args.quick else GEN_PROMPTS
    methods = {"baseline": [], "vector-sum": [], "conceptor-AND": []}
    for p in prompts:
        methods["baseline"].append(gen(model, tok, p, device))
        h = add_vector_hook(model, layer, v_sum.to(device), alpha=1.0)
        methods["vector-sum"].append(gen(model, tok, p, device))
        remove_hooks([h])
        h = make_and_hook(model, layer, C_and, device)
        methods["conceptor-AND"].append(gen(model, tok, p, device))
        remove_hooks([h])

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"baseline": "gray", "vector-sum": "C1", "conceptor-AND": "C0"}
    print("\n=== two-attribute steering (sentiment x formality) ===")
    for name, texts in methods.items():
        sent = [score_attr(t, POS_LEX, NEG_LEX) for t in texts]
        form = [score_attr(t, FORMAL_LEX, INFORMAL_LEX) for t in texts]
        ax.scatter(sent, form, c=colors[name], label=name, s=60, alpha=0.7)
        both = np.mean([(s > 0) and (f > 0) for s, f in zip(sent, form)])
        print(f"  {name:<14} mean-sent={np.mean(sent):+.3f} "
              f"mean-formal={np.mean(form):+.3f} both-positive-frac={both:.2f}")
    ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
    ax.set_xlabel("sentiment score (pos - neg)")
    ax.set_ylabel("formality score (formal - informal)")
    ax.set_title("Part 2: conceptor-AND vs vector-sum steering")
    ax.legend()
    savefig(fig, outdir, "conceptor_two_attribute.png", show=args.show)


def main():
    args = get_arg_parser(description=__doc__).parse_args()
    outdir = init(args, "07_02_conceptor_steering")
    part1(outdir, args)
    part2(outdir, args)


if __name__ == "__main__":
    main()
