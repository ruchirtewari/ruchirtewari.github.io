# Programs — Introduction to Neural Network Interpretability

Runnable companion programs for both books (Part I chapters 1–4, Part II
chapters 5–8). Full specifications live in `../programs.md`.

Files are named `CC_SS_title.py` (chapter, section), so `ls -l` lists them
in book order:

```
01_02_uniformrandom_vs_markov.py       ch1  Shannon's ladder of text approximations
01_03_entropy_vs_context_length.py     ch1  conditional entropy vs context; the wall
02_01_kl_bowl_and_fisher.py            ch2  KL bowl curvature = Fisher information
02_02_fisher_rao_distance_bernoulli.py ch2  honest ruler vs Euclidean, verified by flips
03_01_ngram_vs_tiny_transformer.py     ch3  entropy ladder + one learned rung
03_02_flow_warp_2d.py                  ch3  Gaussian warped into two moons
03_03_expert_specialization_toy.py     ch3  mixture-of-experts specialization + collapse
04_01_natural_vs_plain_gradient.py     ch4  zigzag vs geodesic; reparametrization test
04_02_preconditioning_comparison.py    ch4  SGD vs Adam vs diagonal/exact Fisher
04_03_flat_minima_and_noise.py         ch4  noise selects wide basins
05_01_intrinsic_dimension_profile.py   ch5  TwoNN hunchback across GPT-2 layers
05_02_concept_direction_probe_and_steer.py ch5 probe vs steer, per layer
05_03_superposition_phase_transition.py ch5  Elhage toy model phase transition
05_04_entropy_lens.py                  ch5  entropy + KL-to-final across layers (lens)
06_01_sparse_autoencoder_features.py   ch6  TopK SAE on GPT-2 acts + feature browser
06_02_induction_head_finder.py         ch6  find + ablate induction heads
06_03_activation_patching_lab.py       ch6  IOI patching heatmap; ablation flavors
07_01_refusal_direction_steering.py    ch7  refusal direction: ablate/add + dose-response
07_02_conceptor_steering.py            ch7  conceptors: reservoir demo + AND-steering
07_03_prompt_finetune_steer_comparison.py ch7 three levers, one behavior, measured
08_01_chain_of_thought_faithfulness.py ch8  hint injection + truncation faithfulness
08_02_grokking_phase_transition_for_modular_arithmetic.py     ch8  grokking with Fourier leading indicator
08_03_interpretability_case_study.py   ch8  capstone scaffold (quantify→...→report)
08_04_fisher_spectrum_inspector.py     ch8  Fisher eigenvalues tracked through grokking
```

## Prerequisites

1. **Python 3.10+** with a virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (`torch` CPU wheels are fine; a GPU/Apple-Silicon MPS speeds up
   chapters 6–8.)

2. **Ollama** (only for programs that use an LLM for generation/judging —
   06_01's auto-labeling, 08_01's hint check, 08_03):
   ```bash
   brew install ollama          # macOS; see ollama.com for Linux
   ollama serve                 # if not already running as a service
   ollama pull llama3.2         # the default model in config.json
   ```
   Programs degrade gracefully (skip the LLM step with a notice) if the
   backend is unreachable.

3. **HuggingFace models** download automatically on first use into the HF
   cache: `gpt2` (~500 MB) for chapters 5–6, `Qwen/Qwen2.5-1.5B-Instruct`
   (~3 GB) for chapters 7–8. No account/token needed for these.

4. **Data**: the Shakespeare corpus (ch1/ch3) and OpenWebText sample
   (06_01) download automatically on first run into `cache/`.

## Remote vs local LLM (modular configuration)

All LLM text generation goes through `common.LLM`, configured in
**`config.json`** — one edit switches every program:

- `"llm_backend": "ollama"` (default) — local, private, free.
- `"llm_backend": "openai_compatible"` — any remote endpoint speaking
  `/v1/chat/completions` (OpenAI, Together, vLLM, LM Studio…). Set
  `base_url`, `model`, and export the API key named by `api_key_env`.
- `"llm_backend": "hf_pipeline"` — in-process transformers pipeline.

**Scope note:** the backend abstraction covers *text generation only*.
Programs that read or edit activations (05_x, 06_x, 07_x) necessarily run
a local HuggingFace model — internals are not available over any API. The
local model names are also in `config.json` (`hf_base_model`,
`hf_chat_model`).

## Running

```bash
./run.sh list            # all programs in book order
./run.sh 02_01           # one program
./run.sh 04              # all of chapter 4
./run.sh all --quick     # everything in smoke-test mode
./run_grok.sh            # flagship pair: grokking (08_02) + Fisher spectrum (08_04), logged
```

`run_grok.sh` replicates the book's headline result end to end: the 08_02
grokking transition (test accuracy stuck at ~0.30, jumping to ~1.0 near step
5000–5500) and the 08_04 Fisher-spectrum run on the same regime. Reference
outcome (seed 0): the top Fisher eigenvalue climbs through the plateau,
spikes ~10× at step ~4800 while train accuracy briefly dips, and collapses
as the grok lands — the largest spectral movement (step 5200) **leads** the
behavioral jump (step 5600) by ~400 steps, with ~9000× dynamic range across
the run. `./run_grok.sh -h` prints runtimes and the full expected-results
block; logs stream to `logs/grok_*_latest.log`.

Every program supports `--seed N`, `--outdir DIR`, `--show` (interactive
figures), and `--quick` (reduced steps/samples — minutes instead of
hours). Figures and artifacts land in `out/<program_name>/`.

## Expected runtimes (full mode)

| Fast (< 5 min) | Medium (5–30 min) | Long (≥ 1 h) |
|---|---|---|
| 01_x, 02_x, 03_02, 03_03, 04_x, 05_03 | 05_01, 05_02, 05_04, 06_02, 06_03, 07_02, 08_02 (GPU), 08_04 (quick) | 03_01 (CPU), 06_01, 07_01, 07_03, 08_01, 08_02 (CPU), 08_04 (full) |
