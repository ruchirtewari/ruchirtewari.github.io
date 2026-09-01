"""Chapter 1, §1.2-1.3 — Shannon's ladder of text approximations.

Build order-0..4 character Markov models and order-1/2 word models on
Shakespeare by counting; generate text from each rung and measure held-out
cross-entropy (add-one smoothing).

Expected result: order-2 samples show pronounceable fragments, order-4 is
mostly word-like, and bits/char falls strictly with model order from the
uniform baseline log2(27) ~ 4.75.
"""

import math
import random

from common import get_arg_parser, init, load_shakespeare

ALPHABET = " " + "".join(chr(ord("A") + i) for i in range(26))  # 27 symbols


def count_ngrams(seq, k):
    """seq: string or list of words. Returns {context: {next: count}}."""
    counts = {}
    for i in range(k, len(seq)):
        ctx = tuple(seq[i - k:i]) if isinstance(seq, list) else seq[i - k:i]
        d = counts.setdefault(ctx, {})
        d[seq[i]] = d.get(seq[i], 0) + 1
    return counts


def cross_entropy(counts, heldout, k, vocab_size):
    """Bits per symbol on heldout with add-one smoothing over vocab_size."""
    totals = {ctx: sum(d.values()) for ctx, d in counts.items()}
    bits = 0.0
    n = 0
    for i in range(k, len(heldout)):
        ctx = (tuple(heldout[i - k:i]) if isinstance(heldout, list)
               else heldout[i - k:i])
        d = counts.get(ctx, {})
        p = (d.get(heldout[i], 0) + 1) / (totals.get(ctx, 0) + vocab_size)
        bits -= math.log2(p)
        n += 1
    return bits / n


def generate(counts, k, length, seed_ctx):
    """Sample from the empirical conditional distributions."""
    contexts = list(counts.keys())
    out = list(seed_ctx)
    ctx = seed_ctx
    for _ in range(length):
        d = counts.get(ctx)
        if not d:  # unseen context: restart from a random observed one
            ctx = random.choice(contexts)
            d = counts[ctx]
        syms, wts = zip(*d.items())
        nxt = random.choices(syms, weights=wts)[0]
        out.append(nxt)
        ctx = (ctx + (nxt,))[-k:] if isinstance(ctx, tuple) else (ctx + nxt)[-k:] if k else ""
    return out


def main():
    args = get_arg_parser("Shannon's ladder: monkey vs Markov text").parse_args()
    init(args, "01_02_monkey_vs_markov_text")

    text = load_shakespeare()
    if args.quick:
        text = text[:500_000]
    split = int(0.9 * len(text))
    train, held = text[:split], text[split:]
    print(f"corpus: {len(text):,} chars ({len(train):,} train / {len(held):,} held out)\n")

    results = [("uniform (monkey)", math.log2(27))]
    n_sample = 400

    print("=" * 70)
    print("UNIFORM RANDOM (order 0, no statistics)")
    print("=" * 70)
    print("".join(random.choices(ALPHABET, k=n_sample)) + "\n")

    # ---- character models, order 0..4 -----------------------------------
    for k in range(5):
        counts = count_ngrams(train, k)
        h = cross_entropy(counts, held, k, 27)
        results.append((f"char order-{k}", h))
        seed = train[:k] if k else ""
        sample = "".join(generate(counts, k, n_sample, seed))
        print("=" * 70)
        print(f"CHARACTER MODEL, ORDER {k}   ({h:.3f} bits/char)")
        print("=" * 70)
        print(sample + "\n")

    # ---- word models, order 1 and 2 -------------------------------------
    words = train.split()
    held_words = held.split()
    vocab = len(set(words)) + 1  # +1 for unseen held-out words
    for k in (1, 2):
        counts = count_ngrams(words, k)
        # bits/word on held-out, converted to bits/char (word + one space)
        totals = {c: sum(d.values()) for c, d in counts.items()}
        bits = chars = 0.0
        for i in range(k, len(held_words)):
            ctx = tuple(held_words[i - k:i])
            d = counts.get(ctx, {})
            p = (d.get(held_words[i], 0) + 1) / (totals.get(ctx, 0) + vocab)
            bits -= math.log2(p)
            chars += len(held_words[i]) + 1
        h = bits / chars
        results.append((f"word order-{k}", h))
        sample = " ".join(generate(counts, k, 90, tuple(words[:k])))[:n_sample]
        print("=" * 70)
        print(f"WORD MODEL, ORDER {k}   ({h:.3f} bits/char)")
        print("=" * 70)
        print(sample + "\n")

    print("=" * 70)
    print(f"{'model':<20} {'bits/char':>10}")
    print("-" * 32)
    for name, h in results:
        print(f"{name:<20} {h:>10.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
