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



## README_programs.txt
## Hands-on programs for "Introduction to Neural Network Interpretability"

This file describes every program: what it does, what INPUT it needs, and what
OUTPUT to expect (printed tables, saved figures, and the acceptance criterion --
the specific thing you should see if it worked).

Files are named CC_SS_title.py (chapter, section) so `ls -l` lists them in book
order. Figures/artifacts are written to out/<program_name>/.

-------------------------------------------------------------------------------
 RUNNING THEM
-------------------------------------------------------------------------------
Use run_tests.sh (creates/activates .venv, installs deps once, then runs):

    ./run_tests.sh                 show help (default)
    ./run_tests.sh list            list all programs in order
    ./run_tests.sh all             run everything
    ./run_tests.sh 03              run all of chapter 3
    ./run_tests.sh 05_02           run one program (chapter 5, section 2)
    ./run_tests.sh 05_02 --show    trailing args pass through to the program
    ./run_tests.sh all --quick     fast smoke mode for everything

The flagship pair (grokking 08_02 + Fisher spectrum 08_04) has a dedicated
replication script with logging and reference results in its help text:

    ./run_grok.sh                  run both, logged to logs/
    ./run_grok.sh 08_04            just the Fisher spectrum inspector
    ./run_grok.sh -h               expected results, runtimes

Common flags every program accepts:
    --quick    reduced steps/samples (minutes instead of hours)
    --show     also display figures interactively (default: save only)
    --seed N   set the random seed (default 0)
    --outdir D base directory for outputs (default ./out)

-------------------------------------------------------------------------------
 PREREQUISITES
-------------------------------------------------------------------------------
- Python 3.10+ (a .venv is created automatically by run_tests.sh).
- pip deps: numpy, matplotlib, torch, transformers, datasets, peft, tqdm
  (installed from requirements.txt into the venv, never system Python).
- Ollama (only for 08_01): `ollama serve` + a small model, e.g.
  `ollama pull gemma3:1b`. Configured in config.json.
- HuggingFace models download automatically on first use:
  gpt2 (~500 MB, chapters 5-6, 08_03) and the small chat model
  Qwen2.5-0.5B-Instruct (chapters 7-8). No token needed.
- Data (Shakespeare text, OpenWebText sample) downloads to cache/ on first run.

-------------------------------------------------------------------------------
 MODEL / BACKEND CONFIGURATION  (config.json)
-------------------------------------------------------------------------------
One file controls all model and LLM choices:
  llm_backend      "ollama" (local) | "openai_compatible" (remote) | "hf_pipeline"
  ollama.model     model name served by ollama          (default gemma3:1b)
  hf_base_model    local model for internals             (default gpt2)
  hf_chat_model    local chat model for steering/JSON    (Qwen2.5-0.5B-Instruct)
  device           "auto" | "cpu" | "cuda" | "mps"
Programs that read/edit activations always use a local HuggingFace model
(internals are not available over an API); only text generation is routed
through llm_backend.

===============================================================================
 PART I -- CHAPTER 1: STATISTICAL FOUNDATIONS
===============================================================================

01_02_monkey_vs_markov_text.py            (numpy only, < 1 min)
  Does : reproduces Shannon's 1948 ladder -- generates text from a uniform
         "monkey" and from char order-0..4 and word order-1..2 Markov models
         built by counting; scores each by held-out cross-entropy.
  Input: Shakespeare corpus (auto-downloaded to cache/).
  Output (stdout): a ladder of sample texts, one block per model, then a table
         of bits/char. No figure.
  Expect: samples get more language-like with order; bits/char falls from
         ~4.75 (uniform) monotonically. ACCEPTANCE: order-2 samples show
         pronounceable fragments; cross-entropy decreasing.

01_03_entropy_vs_context_length.py        (numpy only, ~1 min)
  Does : estimates conditional entropy H(next char | last k chars), k=0..8, and
         counts how many distinct k-contexts actually occur vs 27^k.
  Input: Shakespeare corpus (cached).
  Output: 2 figures -- entropy-vs-k (with Shannon reference lines 4.75/4.1/3.1/
         1.0) and observed-contexts vs 27^k (log scale).
  Expect: entropy falls then the estimate becomes unreliable exactly where
         contexts stop repeating -- the state-explosion wall as data sparsity.

===============================================================================
 PART I -- CHAPTER 2: GEOMETRIC FOUNDATIONS
===============================================================================

02_01_kl_bowl_and_fisher.py               (numpy only, seconds)
  Does : verifies numerically that KL divergence is locally a quadratic "bowl"
         whose curvature equals the Fisher information, for Bernoulli(theta in
         {0.5,0.8,0.95}) and Gaussian mean (sigma in {0.5,1,2}).
  Input: none.
  Output: a 2x3 grid figure (numeric KL vs the 1/2 F delta^2 overlay) and a
         printed max-relative-error table.
  Expect: overlay matches KL to <1% for small delta; bowls narrow toward the
         edge (theta->1) / small sigma. ACCEPTANCE: quadratic matches; cubic
         term visibly bites near the edge.

02_02_fisher_rao_distance_bernoulli.py    (numpy only, ~1 min)
  Does : contrasts Euclidean distance |p-q| with the Fisher-Rao distance for
         coins, and verifies the ruler by a coin-flip distinguishability test.
  Input: none.
  Output: 2 heatmaps (both rulers) + a slice plot; a printed table:
         pair -> d_Euclid, d_FR, median flips-to-distinguish (1000 trials).
  Expect: (0.98,0.99) and (0.50,0.51) have EQUAL Euclidean distance but the
         former needs ~17x fewer flips to tell apart. ACCEPTANCE: flips track
         d_FR, not d_Euclid.

===============================================================================
 PART I -- CHAPTER 3: NEURAL NETWORKS AS DISTRIBUTION MACHINES
===============================================================================

03_01_ngram_vs_tiny_transformer.py        (torch, ~1 h CPU / minutes GPU)
  Does : trains a 2-layer 2-head char transformer (~0.5M params) and compares
         it to order-2..5 char n-grams on the same corpus; finds where the
         learned model beats the counted one.
  Input: Shakespeare corpus (cached).
  Output: a bits/char bar chart; printed list of the 10 test positions with the
         largest transformer-vs-ngram advantage, with context.
  Expect: transformer beats order-5 n-gram; disagreement positions are visibly
         long-range (names, quotes, brackets). --quick trains far fewer steps.

03_02_flow_warp_2d.py                      (torch, ~5 min CPU)
  Does : trains a minimal 4-layer RealNVP normalizing flow to map a Gaussian
         into a two-moons distribution; visualizes the warp layer by layer.
  Input: synthetic two-moons (generated in-script).
  Output: a 6-panel figure (base + each coupling layer's grid + sample cloud;
         final panel overlays the learned log-density).
  Expect: the Gaussian bends smoothly onto two moons; density high on the moons.

03_03_expert_specialization_toy.py        (torch, ~2 min)
  Does : trains a 4-expert mixture-of-experts + soft gate on a 4-quadrant
         regression, then repeats with hard top-1 routing.
  Input: synthetic 4-quadrant function (in-script).
  Output: panels of gate routing maps and per-expert/per-quadrant error over
         training; a second run showing expert collapse under hard routing.
  Expect: soft gate -> ~one expert per quadrant; hard routing -> one expert
         takes everything (the failure load-balancing losses prevent).

===============================================================================
 PART I -- CHAPTER 4: TRAINING DYNAMICS
===============================================================================

04_01_natural_vs_plain_gradient.py        (numpy only, seconds)
  Does : fits a Gaussian's (mu, log sigma) by NLL with plain gradient descent
         vs natural gradient (analytic 2x2 Fisher), then reparametrizes to show
         invariance.
  Input: samples from a target Gaussian (in-script).
  Output: trajectories over loss contours; loss-vs-step; a second figure showing
         the natural-gradient path is unchanged under sigma->sigma^2 while plain
         GD's path changes.
  Expect: plain GD zigzags/stalls as sigma shrinks; natural gradient is smooth;
         reparametrized NGD identical in distribution space.

04_02_preconditioning_comparison.py       (numpy, ~1 min)
  Does : trains an ill-conditioned logistic regression (condition number ~1e6)
         with SGD, Adam, diagonal-Fisher, and exact natural gradient.
  Input: synthetic data with feature scales spanning 1e3 (in-script).
  Output: a loss-vs-step figure (log y) with all four; a printed ranking.
  Expect: Adam and diagonal-Fisher curves nearly coincide (Adam IS a diagonal
         Fisher method); exact NGD fastest; plain SGD slowest.

04_03_flat_minima_and_noise.py            (numpy only, ~1 min)
  Does : runs noisy gradient descent from many starts on a 2-D loss with one
         wide and one narrow minimum; measures which basin wins vs noise scale
         and the generalization gap under a shifted "test" surface.
  Input: none (constructed surface).
  Output: 3 panels -- the surface, wide-basin fraction vs noise, test-loss by
         basin.
  Expect: more noise -> more runs settle in the WIDE basin; wide basin has lower
         test loss under the shift.

===============================================================================
 PART II -- CHAPTER 5: REPRESENTATION GEOMETRY
===============================================================================

05_01_intrinsic_dimension_profile.py      (gpt2, ~10 min CPU)
  Does : measures intrinsic dimension (TwoNN estimator) of gpt2's residual
         stream per layer, on real sentences vs word-shuffled sentences.
  Input: ~2000 sentences (from the corpus); gpt2 (auto-download).
  Output: an ID-vs-layer curve for both conditions; a sample-count saturation
         plot.
  Expect: the "hunchback" -- ID rises then falls across layers, and ID << 768
         everywhere; shuffled text differs.

05_02_concept_direction_probe_and_steer.py (gpt2, ~20 min CPU)
  Does : builds a sentiment direction (difference of means) per layer, then
         evaluates it BOTH by probing (held-out accuracy) and by steering
         (adding it during generation), showing the two disagree.
  Input: 50+50 sentiment template sentences + neutral prompts (in-script); gpt2.
  Output: a twin-axis plot of probe accuracy and steering effect vs layer.
  Expect: probe accuracy peaks at a different layer than the steering effect --
         "information present" is not "information used".

05_03_superposition_phase_transition.py   (torch toy, ~5 min)
  Does : reproduces Elhage et al.'s toy model of superposition; sweeps feature
         sparsity S and records how many features get represented.
  Input: synthetic sparse features (in-script), no pretrained model.
  Output: features-represented vs S; W^T W interference heatmaps per S.
  Expect: <=5 features represented when dense (S=0), MORE than the 5 dimensions
         when sparse (superposition); regular polygon geometry in W^T W.

05_04_entropy_lens.py                     (gpt2, ~2 min CPU)
  Does : reads the residual stream after every layer through the model's own
         unembedding (logit lens) and measures two things per layer: Shannon
         entropy of the readout (bits) and KL(final || layer) -- how far each
         layer's readout is from the model's actual final prediction.
  Input: built-in factual/prose prompts; gpt2. --quick uses 4 prompts.
  Output: 3 panels -- KL-to-final vs layer (falls); entropy vs layer (NOT
         monotone -- the cautionary exhibit); KL heatmap per (layer, position).
  Expect: acceptance requires late-layer KL well below early-layer KL (the
         prediction forms with depth). The entropy curve deliberately does NOT
         fall monotonically: early logit-lens readouts can be confidently
         wrong (low entropy about garbage) -- the ch5 lens-bias lesson that
         entropy alone cannot separate confident-right from confident-wrong.

===============================================================================
 PART II -- CHAPTER 6: FEATURES AND CIRCUITS
===============================================================================

06_01_sparse_autoencoder_features.py      (gpt2 + SAE training, ~1 h+ CPU)
  Does : caches gpt2 layer-6 activations, trains a 768->8192 TopK sparse
         autoencoder (k=32), and browses recovered features.
  Input: text sample (OpenWebText sample if available, else corpus); gpt2.
  Output: a training curve; a printed feature browser (top-activating contexts
         per latent, trigger token marked). Optional LLM auto-labeling if a
         backend is reachable (skips gracefully otherwise).
  Expect: many latents show coherent themes (punctuation, a syntax cue, a
         topic); some are messy -- both are the lesson. --quick uses a tiny
         token budget.

06_02_induction_head_finder.py            (gpt2, ~5 min CPU)
  Does : finds gpt2's induction heads by a prefix-matching score on repeated
         random token sequences, then ablates them and measures the loss change.
  Input: 100 synthetic repeated-token sequences; gpt2 (loaded with eager
         attention so attention weights are available).
  Output: a 12x12 head heatmap of prefix-matching scores; a first/second-half
         loss bar chart with and without ablation.
  Expect: a few heads (layers ~5-7) score far above the rest; ablating them
         erases most of the second-half in-context-learning advantage.

06_03_activation_patching_lab.py          (gpt2, ~10 min CPU)
  Does : reproduces the indirect-object (IOI) name-mover result by activation
         patching, and shows that the ablation-flavor choice changes the answer.
  Input: 20 clean/corrupted (name-swapped) IOI prompt pairs (in-script); gpt2.
  Output: a 12x12 logit-difference recovery heatmap; three ranked head lists
         under zero-, mean-, and resample-ablation.
  Expect: name-mover heads (~layers 9-10) dominate the heatmap; the three
         ablation rankings visibly differ -- methodology matters.

===============================================================================
 PART II -- CHAPTER 7: CONTROL AND STEERING
===============================================================================

07_01_refusal_direction_steering.py       (chat model, downloads ~1 GB)
  Does : computes a "refusal direction" (mean diff of harmful vs harmless
         prompt activations) on a small chat model, then suppresses refusal by
         projection-ablation and induces it by vector-addition; prices the
         capability tax. Content is deliberately MILD ("pick the lock on my own
         door").
  Input: ~40+40 mild instruction pairs + capability questions (in-script);
         Qwen2.5-0.5B-Instruct (config hf_chat_model).
  Output: dose-response plots -- refusal rate on harmful & harmless prompts vs
         steering coefficient; capability retention vs coefficient.
  Expect: ablation drops harmful-refusal at near-baseline capability; addition
         induces refusals on harmless prompts; large coefficients degrade
         capability -- the tax made visible. Exits cleanly if the model is
         unavailable.

07_02_conceptor_steering.py               (numpy + gpt2, ~15 min)
  Does : Part 1 -- conceptors on a 500-unit echo state network storing and
         morphing periodic patterns. Part 2 -- conceptor-AND vs additive
         vector-sum steering on gpt2 for two attributes at once (positive +
         formal).
  Input: periodic signals (Part 1); sentiment/register contrast sets (Part 2);
         gpt2.
  Output: regenerated + morphed waveform plots; a 2-D scatter (sentiment x
         formality score) per steering method.
  Expect: patterns regenerate/morph smoothly; conceptor-AND hits the
         both-attributes quadrant more reliably than the vector sum.

07_03_prompt_finetune_steer_comparison.py (chat model + LoRA, ~1-2 h CPU)
  Does : implements ONE behavior (respond in valid JSON with keys answer,
         confidence) three ways -- system prompt, LoRA fine-tune, steering
         vector -- and fills the three-levers trade-off table with measured
         numbers.
  Input: 500 synthetic instruction->JSON examples (in-script; --quick shrinks to
         ~20, 1 epoch); Qwen2.5-0.5B-Instruct; peft for LoRA.
  Output: printed table -- per lever: compliance rate, compliance after an
         adversarial "ignore the format" turn, capability retention, tokens/sec.
  Expect: prompting is easy but weak under adversarial pressure; LoRA most
         durable; steering intermediate (and weak on a 0.5B model in --quick --
         a genuine measured result, not a crash).

===============================================================================
 PART II -- CHAPTER 8: EVALUATION AND FRONTIERS
===============================================================================

08_01_chain_of_thought_faithfulness.py    (LLM via ollama/remote, ~1-2 h)
  Does : measures chain-of-thought unfaithfulness -- runs each question clean
         and with an injected wrong-answer hint, and checks how often the answer
         moves without the reasoning admitting the hint moved it. Truncation
         extension cuts the reasoning at 25/50/75% and forces an answer.
  Input: 30 embedded multiple-choice questions; text generation via config
         llm_backend (default ollama gemma3:1b).
  Output: the two headline rates with bootstrap CIs; a bar chart and a
         truncation-stability plot.
  Expect: a nonzero "influenced but unacknowledged" rate -- the unfaithfulness
         gap. Exits cleanly with a message if no LLM backend is reachable.

08_02_grokking_phase_transition_for_modular_arithmetic.py         (torch, minutes GPU / hours CPU)
  Does : trains a 2-layer transformer (width 128, 4 heads -- original Power et
         al. 2022 size, ~4e5 non-embedding params) on a+b mod 97 with weight
         decay 1.0, logging accuracy, weight norm, and the Fourier spectrum of
         the token embeddings.
  Input: all 97*97 pairs mod 97 (in-script), no pretrained model.
         --train-frac sets the regime: 0.3 (default) grokks; 0.5 generalizes
         immediately with no delay to observe.
  Output: 3 stacked panels vs step (log x) -- train/test accuracy; weight norm;
         embedding Fourier power.
  Expect (FULL run, --train-frac 0.3): train saturates by ~step 500 while test
         sits at chance (~0.30) for a long plateau, then test snaps to ~1.0
         around step ~5000-5500 -- the grok. Weight norm rises during
         memorization, falls under weight decay through the plateau, and settles
         as the grok lands. Sample curve (seed 0, CPU):

           step    500  train 1.000  test 0.299  wnorm 75.6  <- memorized
           step   2000  train 1.000  test 0.299  wnorm 37.1  <- plateau
           step   4500  train 1.000  test 0.435  wnorm 38.5  <- circuit winning
           step   5000  train 1.000  test 0.712  wnorm 36.2
           step   5500  train 1.000  test 0.997  wnorm 33.8  <- GROK
           step 100000  train 1.000  test 1.000  wnorm ~31   <- stays generalized
           acceptance: delayed generalization present: True  (delay ~5000 steps)

         NOTE: --quick is deliberately too short to grok; it says so and you
         must run without --quick to see the transition. Grokking is a lab
         curiosity -- it exists to let you WATCH the circuit form; immediate
         generalization (--train-frac 0.5) is the better practical outcome.

08_03_interpretability_case_study.py       (gpt2, ~10 min, capstone scaffold)
  Does : a runnable template for the full loop -- quantify -> hypothesize ->
         probe -> intervene -> report -- demonstrated on gpt2 completing "The
         capital of France is" with " Paris" and a country/capital steering
         direction.
  Input: default prompts + contrast pairs (in-script; edit to study your own
         behavior); gpt2.
  Output: a logit-lens-across-layers plot; a steering dose-response plot; and a
         markdown report written to out/08_03.../report.md with the numbers
         filled in.
  Expect: a completed report.md with measured effect sizes and CIs. Stages 1,3,
         4,5 run automatically; Stage 2 (hypotheses) is where you write your own.

08_04_fisher_spectrum_inspector.py         (torch, quick ~2 min / full ~1 h CPU)
  Does : re-runs the 08_02 grokking experiment (imports its model/data) and,
         every few hundred steps, estimates the top-k eigenvalues of the true
         Fisher information matrix via Lanczos over Fisher-vector products
         (exact GGN VJP through a retained graph; central-difference JVP).
         The ch2 'measuring Fisher in practice' subsection, made runnable --
         and the test of ch8's claim that internal observables move before
         behavior.
  Input: all pairs a+b mod 97 (from 08_02; train-frac 0.3, wd 1.0, seed 0).
  Output: 3 stacked panels vs step -- train/test accuracy; top-k Fisher
         eigenvalues (log y); top eigenvalue vs embedding Fourier share.
  Expect (FULL run): the run grokks (~step 5000-5500) and the top eigenvalue
         has >5x dynamic range; the lead/lag between the largest spectral
         movement and the test-accuracy jump is PRINTED AS A MEASUREMENT --
         either sign is a finding, not a failure. Reference run (seed 0):
         top eig climbs 3.4 -> ~4e3 through the plateau, spikes ~10x to
         ~3e4 at step ~4800 (train acc dips to 0.70 -- the memorized
         solution destabilizing), collapses as test acc snaps up; largest
         spectral movement at 5200 vs grok at 5600 -- spectrum LEADS
         behavior by ~400 steps; dynamic range ~9000x. Replicate both
         programs with ./run_grok.sh. --quick (1200 steps) only validates
         the estimator: finite positive spectrum, no grok.

===============================================================================
 TROUBLESHOOTING
===============================================================================
- "no module named torch": run through run_tests.sh (it provisions the venv),
  or `source .venv/bin/activate && pip install -r requirements.txt`.
- Chat/LLM programs print a skip message instead of crashing when the model or
  ollama backend is not available -- start ollama or edit config.json.
- First run of any model program downloads weights (gpt2 ~500 MB, chat ~1 GB);
  subsequent runs use the HF cache.
- No GPU is required; everything runs on CPU. For the long ones (03_01, 06_01,
  07_03, 08_02 full), a single small cloud GPU cuts hours to minutes -- see the
  project notes on free Kaggle/Colab or a cheap RunPod/Vast.ai box.
- Every program writes to out/<program_name>/; delete out/ to reset.
===============================================================================
