"""
Visual program specs for Brin-Stuck "Introduction to Dynamical Systems".
Each spec is a dict with: id, chapter, title, concept, what_it_shows,
parameters, algorithm, output, dependencies.
"""

CHAPTER_SUMMARIES = {
    1: """Chapter 1 — Examples and Basic Concepts
Introduces dynamical systems through 13 canonical examples that serve as
reference points for the entire book. A dynamical system is a set X with a
self-map f (or flow); the central objects of study are orbits O(x) = {f^n(x)}.
The chapter showcases: circle rotations (Rα), expanding endomorphisms (Em),
full and sub-shifts (symbolic dynamics), the logistic/quadratic family (Qc),
the Gauss/continued-fraction map, hyperbolic toral automorphisms (Arnold's
cat map), Smale's horseshoe, the solenoid, ODE flows, and strange attractors
(Lorenz, Hénon). Lyapunov exponents are introduced as a measure of sensitive
dependence.""",

    2: """Chapter 2 — Topological Dynamics
Studies dynamical systems on topological spaces without measure structure.
Key concepts: ω-limit sets (long-run accumulation points of orbits),
non-wandering sets, topological transitivity (a single dense orbit),
topological mixing (orbits eventually visit every open pair), and
expansiveness (distinct orbits eventually separate). Topological entropy
h(f) is defined via separating or spanning sets — it counts the exponential
growth rate of distinguishable orbit segments, and is a conjugacy invariant.
Concludes with equicontinuity, distality, and Ramsey-theoretic applications.""",

    3: """Chapter 3 — Symbolic Dynamics
Encodes dynamical systems as sequences over a finite alphabet. Any orbit
gets an "itinerary" via a partition, yielding a map to the full shift Σ_m.
Subshifts of finite type (SFTs) are defined by a transition matrix A;
the Perron-Frobenius theorem gives their entropy as log(spectral radius of A).
The zeta function ζ_A(z) = 1/det(I-zA) counts periodic orbits. Substitution
systems (e.g., Fibonacci, Thue-Morse) generate deterministic sequences with
exotic spectral properties. The chapter ends with a data-storage application
(run-length and constrained codes).""",

    4: """Chapter 4 — Ergodic Theory
Studies statistical/measure-theoretic behavior of orbits. Poincaré recurrence
shows measure-preserving maps return to every positive-measure set.
Ergodicity means time averages equal space averages (Birkhoff ergodic theorem).
Mixing is a stronger stochastic property. The chapter covers invariant measures,
unique ergodicity (Weyl's equidistribution theorem for irrational rotations),
the Gauss measure for continued fractions, spectral theory (discrete spectrum,
weak mixing), and applications: Furstenberg's proof of Szemerédi's theorem
(number theory) and Google's PageRank (internet search).""",

    5: """Chapter 5 — Hyperbolic Dynamics
Develops the theory of diffeomorphisms with invariant splitting TM = E^s ⊕ E^u:
vectors in E^s contract, vectors in E^u expand under iteration. The shadowing
lemma shows every pseudo-orbit (ε-orbit) is shadowed by a true orbit — making
numerical simulations trustworthy. Invariant cone fields give a practical
criterion for hyperbolicity. Stable and unstable manifolds (Hadamard-Perron)
are the geometric backbone. The horseshoe, Anosov diffeomorphisms, Axiom A
systems, and Markov partitions (conjugating hyperbolic maps to SFTs) are all
treated. Structural stability: small perturbations of Anosov systems are
topologically conjugate.""",

    6: """Chapter 6 — Ergodicity of Anosov Diffeomorphisms
Proves the landmark theorem: every C² volume-preserving Anosov diffeomorphism
is ergodic. The proof uses Hopf's argument: any f-invariant L² function must
be constant along stable manifolds (by time averages) and along unstable
manifolds (by backward averages). Since E^s and E^u together span TM,
absolute continuity of the foliations (the key technical lemma) lets Fubini's
theorem conclude the function is a.e. constant. The chapter establishes Hölder
regularity of the stable/unstable distributions and absolute continuity of the
corresponding foliations.""",

    7: """Chapter 7 — Low-Dimensional Dynamics
One-dimensional maps admit complete classification. For circle homeomorphisms,
the rotation number ρ(f) ∈ [0,1) is a complete topological invariant:
ρ rational ↔ periodic orbits exist; ρ irrational ↔ f is semi-conjugate to Rρ
(Poincaré classification). Denjoy's theorem: a C² diffeomorphism with
irrational ρ is conjugate to Rρ. Sharkovsky's theorem orders periods:
3≻5≻7≻…≻2·3≻…≻4≻2≻1; period-3 implies all periods. For the real
quadratic family f_c(x) = x²+c, the bifurcation diagram and Feigenbaum's
universal period-doubling route to chaos are analyzed via the kneading
invariant and Schwarzian derivative.""",

    8: """Chapter 8 — Complex Dynamics
Rational maps R: ℂ̄ → ℂ̄ have two invariant sets: the Fatou set F(R)
(stable/equicontinuous dynamics) and the Julia set J(R) = complement of F(R)
(chaotic dynamics). J is closed, perfect, and fully invariant; repelling periodic
points are dense in J. For quadratic family f_c(z) = z²+c, the Mandelbrot set
M = {c : orbit of 0 is bounded} is the connectedness locus of Julia sets.
c ∈ M ↔ J(f_c) is connected; c ∉ M ↔ J(f_c) is a Cantor set. Montel's theorem
(normal families) is the key tool throughout.""",

    9: """Chapter 9 — Measure-Theoretic Entropy
The Kolmogorov-Sinai entropy h_µ(T) measures information production per time
step of a measure-preserving transformation. It is computed as the supremum
of h(T, ξ) over all finite partitions ξ; the Kolmogorov-Sinai theorem reduces
this to any generating partition. The Shannon-McMillan-Breiman theorem gives an
almost-sure convergence: (1/n) log µ(ξ^n(x)) → h_µ(T). The variational
principle (ch. 9.5) unifies topological and measure-theoretic entropy:
h(f) = sup_µ h_µ(f), supremum over all f-invariant measures. Bernoulli
shifts are isomorphic iff they have equal entropy (Ornstein, not proved here).""",
}

PROGRAM_SPECS = [

    # ── CHAPTER 1 ────────────────────────────────────────────────────────────

    {
        "id": "ch1_cobweb",
        "chapter": 1,
        "title": "Cobweb Diagram — Quadratic Family & Gauss Map",
        "concept": "Orbits, fixed points, periodic orbits, iteration",
        "sections": ["1.5", "1.6"],
        "what_it_shows": (
            "Interactive cobweb plot for f(x) = x²+c (or any user map). "
            "Starting from x₀ the cobweb zig-zags between y=f(x) and y=x, "
            "making convergence to fixed/periodic orbits or chaos visually obvious. "
            "Include Gauss map G(x) = {1/x} (fractional part) as a preset."
        ),
        "parameters": {
            "map": "quadratic | gauss | logistic | user-defined",
            "c": "parameter for f(x)=x²+c, default -1.0",
            "x0": "initial condition, default 0.3",
            "n_iter": "number of iterations, default 80",
        },
        "algorithm": (
            "1. Build (x, y=f(x)) curve on [0,1] or [-2,2].\n"
            "2. From (x0, 0) draw vertical to (x0, f(x0)), then horizontal to "
            "   (f(x0), f(x0)), repeat n_iter times.\n"
            "3. Color orbit segments by iteration index (early=blue, late=red)."
        ),
        "output": "Single matplotlib figure: y=x line, y=f(x) curve, cobweb path.",
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch1_cobweb.py",
    },

    {
        "id": "ch1_henon_lorenz",
        "chapter": 1,
        "title": "Strange Attractors — Hénon & Lorenz",
        "concept": "Attractors, sensitive dependence, Lyapunov exponents",
        "sections": ["1.12", "1.13"],
        "what_it_shows": (
            "Side-by-side plots: Hénon map attractor (2D) and Lorenz flow "
            "attractor (3D). Overlays two nearby initial conditions diverging "
            "exponentially to illustrate sensitivity. Computes and displays the "
            "leading Lyapunov exponent numerically."
        ),
        "parameters": {
            "henon_a": "1.4 (classic)",
            "henon_b": "-0.3 (classic)",
            "lorenz_sigma": "10",
            "lorenz_rho": "28",
            "lorenz_beta": "8/3",
            "n_points": "100000 for Hénon, 50000 steps for Lorenz",
            "delta_x0": "1e-8 (perturbation for Lyapunov)",
        },
        "algorithm": (
            "Hénon: iterate (x,y)→(a-x²+by, x) from (0,0).\n"
            "Lorenz: RK4 on dx/dt=σ(y-x), dy/dt=x(ρ-z)-y, dz/dt=xy-βz.\n"
            "Lyapunov: run two orbits; λ = (1/N)Σ log(|δ_n|/|δ_0|) with "
            "renormalization every k steps."
        ),
        "output": (
            "2-panel figure: Hénon attractor scatter (2D) + Lorenz butterfly (3D). "
            "Inset: divergence-of-nearby-orbits plot with λ annotation."
        ),
        "dependencies": ["numpy", "matplotlib", "scipy"],
        "file": "ch1_henon_lorenz.py",
    },

    {
        "id": "ch1_toral_auto",
        "chapter": 1,
        "title": "Arnold's Cat Map — Hyperbolic Toral Automorphism",
        "concept": "Hyperbolic toral automorphism, mixing, image stretching",
        "sections": ["1.7"],
        "what_it_shows": (
            "Animates the action of A = [[2,1],[1,1]] on T² = [0,1)²: a pixel "
            "image of a cat (or test image) gets stretched, sheared, and wrapped "
            "until it appears random; after exactly as many steps as the period it "
            "returns. Shows exponential mixing visually."
        ),
        "parameters": {
            "matrix": "[[2,1],[1,1]] default (cat map)",
            "n_steps": "animation frames, default 20",
            "image_size": "64×64 to 256×256",
        },
        "algorithm": (
            "For each pixel (i,j) in NxN grid, apply (x,y)→A(x,y) mod 1 "
            "iteratively, sampling pixel colors. Display as animation or "
            "n-panel grid showing t=0,1,2,...,T."
        ),
        "output": "Animated GIF or multi-panel figure of image evolution on torus.",
        "dependencies": ["numpy", "matplotlib", "pillow"],
        "file": "ch1_cat_map.py",
    },

    # ── CHAPTER 2 ────────────────────────────────────────────────────────────

    {
        "id": "ch2_omega_limit",
        "chapter": 2,
        "title": "Omega-Limit Sets & Non-Wandering Set",
        "concept": "ω-limit sets, recurrence, non-wandering set",
        "sections": ["2.1"],
        "what_it_shows": (
            "For a 2D flow (e.g., Duffing or Van der Pol), scatter-plots the "
            "ω-limit set of many initial conditions using color to distinguish "
            "basins. Overlays the non-wandering set NW(f) detected by tracking "
            "recurrence times. Makes the distinction between limit sets and "
            "transient behavior vivid."
        ),
        "parameters": {
            "system": "van_der_pol | duffing | user_ode",
            "mu": "Van der Pol damping, default 1.0",
            "n_orbits": "50 random initial conditions",
            "T": "integration time, default 200",
        },
        "algorithm": (
            "Integrate ODE with scipy.integrate.solve_ivp. "
            "After transient (first half of T), collect remaining orbit points. "
            "Cluster by color using starting basin. Highlight fixed/periodic pts."
        ),
        "output": "Phase portrait with colored ω-limit sets, limit cycles marked.",
        "dependencies": ["numpy", "matplotlib", "scipy"],
        "file": "ch2_omega_limit.py",
    },

    {
        "id": "ch2_entropy",
        "chapter": 2,
        "title": "Topological Entropy via Spanning/Separating Sets",
        "concept": "Topological entropy, exponential orbit complexity",
        "sections": ["2.5", "2.6"],
        "what_it_shows": (
            "Computes h(f) numerically for the logistic map f_r(x)=rx(1-x) "
            "across r ∈ [0,4]. At each r, counts minimal (n,ε)-spanning sets "
            "for increasing n and fits log(S(n,ε))/n to estimate h(f_r). "
            "Plots h(f_r) vs r alongside the bifurcation diagram so the "
            "entropy-chaos connection is visually direct."
        ),
        "parameters": {
            "r_range": "[0, 4], 400 values",
            "epsilon": "0.01",
            "n_max": "15 (orbit length)",
            "n_sample": "2000 initial conditions",
        },
        "algorithm": (
            "For each r and n, generate all n-step orbits from a grid. "
            "Use greedy set cover to estimate spanning set size S(n,ε). "
            "Fit line to log(S(n,ε)) vs n; slope ≈ h(f).\n"
            "Bifurcation diagram: discard transient, plot last 100 iterates."
        ),
        "output": (
            "2-panel figure: top=bifurcation diagram, bottom=h(f_r) vs r "
            "with log(2) marked at r=4 (full chaos)."
        ),
        "dependencies": ["numpy", "matplotlib", "scipy"],
        "file": "ch2_entropy.py",
    },

    # ── CHAPTER 3 ────────────────────────────────────────────────────────────

    {
        "id": "ch3_sftp",
        "chapter": 3,
        "title": "Subshift of Finite Type — Transition Graph & Perron-Frobenius",
        "concept": "SFT, transition matrix, Perron-Frobenius, periodic orbit counting",
        "sections": ["3.2", "3.3", "3.4"],
        "what_it_shows": (
            "Interactive: user enters a 0/1 transition matrix A. Program draws "
            "the transition graph (digraph), computes spectral radius ρ(A) = e^h, "
            "shows the Perron eigenvector as stationary probabilities on edges, "
            "and plots the number of periodic orbits of period n vs n "
            "(should grow like ρ^n)."
        ),
        "parameters": {
            "A": "transition matrix (default: [[1,1],[1,0]] Fibonacci shift)",
            "n_max": "max period for orbit count, default 20",
        },
        "algorithm": (
            "1. Compute eigenvalues of A; ρ = max |λ|.\n"
            "2. #fixed pts of σ^n = Tr(A^n); plot vs n on semi-log axes.\n"
            "3. Draw graph with networkx; node size ∝ stationary prob.\n"
            "4. Display Perron eigenvector as bar chart."
        ),
        "output": (
            "3-panel: transition graph | periodic orbit counts (log scale) | "
            "Perron eigenvector bars."
        ),
        "dependencies": ["numpy", "matplotlib", "networkx"],
        "file": "ch3_sft.py",
    },

    {
        "id": "ch3_substitution",
        "chapter": 3,
        "title": "Substitution Systems — Fibonacci & Thue-Morse",
        "concept": "Substitution dynamics, self-similarity, power spectrum",
        "sections": ["3.6"],
        "what_it_shows": (
            "Iterates substitution rules (e.g., Fibonacci: a→ab, b→a; "
            "Thue-Morse: a→ab, b→ba) to large length. Plots the sequence as "
            "a 1D color strip and its Fourier power spectrum. Fibonacci "
            "spectrum has discrete lines at golden-ratio-spaced frequencies "
            "(quasicrystal); Thue-Morse has singular continuous spectrum."
        ),
        "parameters": {
            "rule": "fibonacci | thue_morse | user",
            "n_levels": "number of substitution steps, default 14",
        },
        "algorithm": (
            "Apply substitution n_levels times starting from 'a'. "
            "Map letters to {0,1} or {-1,+1}. "
            "Compute FFT; plot |FFT|² on log-log axes."
        ),
        "output": "2-panel: sequence strip + power spectrum (linear and log-log).",
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch3_substitution.py",
    },

    # ── CHAPTER 4 ────────────────────────────────────────────────────────────

    {
        "id": "ch4_birkhoff",
        "chapter": 4,
        "title": "Birkhoff Ergodic Theorem — Time Average vs Space Average",
        "concept": "Ergodicity, time averages, equidistribution",
        "sections": ["4.3", "4.5", "4.7"],
        "what_it_shows": (
            "For irrational rotation Rα (ergodic) and rational rotation Rp/q "
            "(non-ergodic), plots (1/N)Σf(R^n_α(x)) vs N for a test function f. "
            "Ergodic case converges to ∫f dθ regardless of x; non-ergodic case "
            "converges to orbit average, which depends on x. Side panel shows "
            "equidistribution: scatter of {nα mod 1} for 1000 steps."
        ),
        "parameters": {
            "alpha": "irrational rotation angle (default: (√5-1)/2 = golden ratio)",
            "alpha2": "rational angle p/q for comparison",
            "f": "observable: sin(2πx) | indicator([0,0.5]) | x²",
            "N_max": "10000 iterations",
        },
        "algorithm": (
            "Orbit: x_n = x_0 + nα mod 1. "
            "Running average A_N = (1/N)Σ_{n=0}^{N-1} f(x_n). "
            "Space average: numerical integration of f over [0,1]."
        ),
        "output": (
            "2-panel: left=running average converging to space mean (2 curves: "
            "ergodic vs non-ergodic); right=equidistribution scatter."
        ),
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch4_birkhoff.py",
    },

    {
        "id": "ch4_pagerank",
        "chapter": 4,
        "title": "PageRank as Ergodic Theory — Stationary Measure of a Random Walk",
        "concept": "Invariant measures, unique ergodicity, PageRank",
        "sections": ["4.12", "4.6"],
        "what_it_shows": (
            "Builds a small random web graph, constructs the Google matrix "
            "G = (1-d)·(1/n)·11ᵀ + d·A (d=0.85), and computes the stationary "
            "distribution by power iteration. Animates rank convergence and "
            "displays the final ranked graph with node size ∝ PageRank."
        ),
        "parameters": {
            "n_pages": "20 (default)",
            "edge_prob": "0.3",
            "damping": "0.85",
            "n_iter": "100 power iterations",
        },
        "algorithm": (
            "Build random directed graph. Form column-stochastic A. "
            "G = d·A + (1-d)/n·1. "
            "Power iteration: v_{k+1} = G·v_k until ||Δv||<1e-8. "
            "Draw with networkx spring layout, node size ∝ rank."
        ),
        "output": "Animated convergence plot + final ranked graph visualization.",
        "dependencies": ["numpy", "matplotlib", "networkx"],
        "file": "ch4_pagerank.py",
    },

    {
        "id": "ch4_poincare_recurrence",
        "chapter": 4,
        "title": "Poincaré Recurrence — Return Times Distribution",
        "concept": "Recurrence, Kac's lemma, return-time distribution",
        "sections": ["4.2"],
        "what_it_shows": (
            "For the irrational rotation Rα and the doubling map E₂(x)=2x mod 1, "
            "picks a small interval A of measure µ(A)=ε and histograms the return "
            "times τ_A(x) for many x₀∈A. Kac's lemma predicts ⟨τ_A⟩=1/µ(A). "
            "For E₂ (mixing) the distribution is approximately geometric; for Rα "
            "(equidistributed) it is concentrated near 1/ε."
        ),
        "parameters": {
            "alpha": "(√5-1)/2",
            "epsilon": "0.02 (interval length = measure)",
            "n_points": "5000 starting points",
            "max_return": "10000 iterations",
        },
        "algorithm": (
            "For each x₀ ∈ A=[0,ε], iterate until x_n ∈ A; record n. "
            "Histogram return times. Draw vertical line at 1/ε (Kac prediction)."
        ),
        "output": "2-panel histogram: Rα return times | E₂ return times, both with Kac mean line.",
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch4_poincare.py",
    },

    # ── CHAPTER 5 ────────────────────────────────────────────────────────────

    {
        "id": "ch5_stable_unstable",
        "chapter": 5,
        "title": "Stable & Unstable Manifolds — Hyperbolic Fixed Point",
        "concept": "Hyperbolic sets, stable/unstable manifolds, homoclinic points",
        "sections": ["5.6", "5.8"],
        "what_it_shows": (
            "For the Hénon map (or Arnold's cat map on ℝ²), computes the stable "
            "manifold W^s(p) and unstable manifold W^u(p) of a hyperbolic fixed "
            "point p by iterating a small segment of the linearized (un)stable "
            "direction forward and backward respectively. Displays the "
            "transverse homoclinic intersection, the hallmark of chaos."
        ),
        "parameters": {
            "map": "henon | catmap",
            "a": "1.4", "b": "-0.3",
            "n_manifold_pts": "5000",
            "n_iter": "20 (folding iterations)",
        },
        "algorithm": (
            "Find fixed point p by Newton iteration on (f(x)-x=0). "
            "Linearize: compute Df(p); eigenvectors give E^s, E^u directions. "
            "W^u: iterate small segment along E^u direction forward n times. "
            "W^s: iterate small segment along E^s direction backward n times."
        ),
        "output": (
            "Phase-plane plot: W^s in blue, W^u in red, fixed point marked, "
            "homoclinic tangle visible after enough iterations."
        ),
        "dependencies": ["numpy", "matplotlib", "scipy"],
        "file": "ch5_manifolds.py",
    },

    {
        "id": "ch5_shadowing",
        "chapter": 5,
        "title": "Shadowing Lemma — Pseudo-orbit vs True Orbit",
        "concept": "ε-orbits, pseudo-orbits, shadowing, numerical reliability",
        "sections": ["5.3"],
        "what_it_shows": (
            "Generates a δ-pseudo-orbit of the cat map (apply exact map, then "
            "add noise δ at each step). Finds the true orbit that shadows it "
            "(using the theoretical shadowing constant). Plots both orbits on "
            "T² to show they stay close. Also shows what happens when δ is too "
            "large (no shadow exists)."
        ),
        "parameters": {
            "delta": "perturbation per step: [1e-4, 1e-3, 1e-2, 0.1]",
            "n_steps": "50",
            "n_trials": "5 pseudo-orbits",
        },
        "algorithm": (
            "Cat map A=[[2,1],[1,1]] on [0,1)². "
            "Pseudo-orbit: x_{n+1} = A·x_n mod 1 + η (||η||≤δ). "
            "Shadow: solve least-squares system for x_0 such that true orbit "
            "stays within C·δ of pseudo-orbit. Display on torus."
        ),
        "output": "2-panel: small δ (orbits indistinguishable) vs large δ (diverge).",
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch5_shadowing.py",
    },

    # ── CHAPTER 6 ────────────────────────────────────────────────────────────

    {
        "id": "ch6_ergodicity_anosov",
        "chapter": 6,
        "title": "Ergodicity of Cat Map — Equidistribution Along Foliations",
        "concept": "Ergodicity, Hopf argument, stable/unstable foliations",
        "sections": ["6.3"],
        "what_it_shows": (
            "Illustrates Hopf's argument numerically for Arnold's cat map. "
            "Takes a smooth test function φ on T². Computes forward time "
            "averages A^+_N φ(x) = (1/N)Σ φ(f^k x) and backward averages "
            "A^-_N φ(x). Shows A^+_N is approximately constant along W^s "
            "leaves and A^-_N along W^u leaves; both converge to ∫φ dµ "
            "as N grows, confirming ergodicity."
        ),
        "parameters": {
            "phi": "sin(2πx)cos(2πy) | indicator of quadrant",
            "N": "500 iterations",
            "grid": "64×64 evaluation points on T²",
        },
        "algorithm": (
            "64×64 grid of initial points. For each, iterate cat map N times. "
            "Compute A^+_N and A^-_N. Display as heatmaps. "
            "Also show variance of A^+_N along stable leaf segments → 0."
        ),
        "output": (
            "4-panel: φ | A^+_50 | A^+_500 | A^-_500, all as heatmaps on T². "
            "Should see A^+ flattening to uniform gray = ∫φ."
        ),
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch6_ergodicity.py",
    },

    # ── CHAPTER 7 ────────────────────────────────────────────────────────────

    {
        "id": "ch7_bifurcation",
        "chapter": 7,
        "title": "Bifurcation Diagram & Feigenbaum Universality",
        "concept": "Period doubling, bifurcations, Feigenbaum constant δ≈4.669",
        "sections": ["7.6", "7.7", "7.8"],
        "what_it_shows": (
            "High-resolution bifurcation diagram for f_r(x)=rx(1-x), r∈[0,4]. "
            "Zooms into successive period-doubling bifurcations to measure "
            "δ_n = (r_{n+1}-r_n)/(r_{n+2}-r_{n+1}) → 4.669… "
            "Shows self-similar structure (the diagram is asymptotically "
            "its own rescaling by δ horizontally and 2.5 vertically)."
        ),
        "parameters": {
            "r_range": "[2.4, 4.0], 3000 pts",
            "n_transient": "1000",
            "n_plot": "500",
            "zoom_sequence": "auto-detect bifurcation points",
        },
        "algorithm": (
            "Bifurcation: for each r, iterate 1000 steps (transient), plot 500. "
            "Bifurcation points: r₁≈3.0, r₂≈3.449, r₃≈3.544, r₄≈3.5644, …\n"
            "Feigenbaum: δ_n = (r_n - r_{n-1})/(r_{n+1} - r_n). "
            "Inset: zoom boxes showing self-similarity."
        ),
        "output": (
            "Main bifurcation diagram + 3 zoom insets + table of δ_n converging "
            "to 4.6692… displayed as annotation."
        ),
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch7_bifurcation.py",
    },

    {
        "id": "ch7_rotation_number",
        "chapter": 7,
        "title": "Rotation Number of a Circle Map",
        "concept": "Rotation number, Poincaré classification, Arnold tongues",
        "sections": ["7.1", "7.2"],
        "what_it_shows": (
            "For the Arnold standard family f_{K,Ω}(θ) = θ + Ω - (K/2π)sin(2πθ), "
            "plots ρ(K, Ω) as a color map over (K,Ω)∈[0,1]×[0,1]. The Arnold "
            "tongues (resonance regions where ρ is rational) appear as colored "
            "wedges emanating from rational points on the Ω-axis. For K=0 "
            "the tongue width is 0 (pure rotation); at K=1 they overlap (chaos)."
        ),
        "parameters": {
            "K_range": "[0, 1], 200 pts",
            "Omega_range": "[0, 1], 200 pts",
            "N": "1000 iterations for rotation number",
        },
        "algorithm": (
            "ρ(f) = lim_{n→∞} F^n(x)/n where F is a lift of f. "
            "For each (K,Ω), iterate 1000 steps, ρ ≈ (F^1000(x₀)-x₀)/1000. "
            "Color by rational value (nearest p/q with q≤10) or irrational."
        ),
        "output": "Arnold tongue diagram: 200×200 color heatmap with tongue labels (0/1, 1/2, 1/3, …).",
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch7_rotation.py",
    },

    {
        "id": "ch7_sharkovsky",
        "chapter": 7,
        "title": "Sharkovsky's Theorem — Period Implication Cascade",
        "concept": "Period forcing, Sharkovsky ordering, period-3 implies chaos",
        "sections": ["7.3"],
        "what_it_shows": (
            "For the tent map T_r(x) = r·min(x, 1-x), sweeps r∈[1,2] and "
            "detects which periods exist at each r. Displays as a 'period "
            "existence' heat-strip: once period 3 appears at r≈1.9, all periods "
            "are present (Sharkovsky). Also shows explicit period-3 orbit and "
            "proves its existence via the intermediate value theorem argument."
        ),
        "parameters": {
            "r_range": "[1.0, 2.0], 500 pts",
            "max_period": "16",
            "n_detect": "10000 transient + 2000 orbit per r",
        },
        "algorithm": (
            "For each r: iterate tent map, collect unique orbit points (rounded). "
            "Detect periods by checking when x_n ≈ x_0. "
            "Build period-existence matrix, display as imshow."
        ),
        "output": "Heat strip: x-axis=r, y-axis=period p∈{1..16}, color=exists.",
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch7_sharkovsky.py",
    },

    # ── CHAPTER 8 ────────────────────────────────────────────────────────────

    {
        "id": "ch8_julia",
        "chapter": 8,
        "title": "Julia Set & Mandelbrot Set — Complex Quadratic Family",
        "concept": "Julia set, Fatou set, Mandelbrot set, connectedness",
        "sections": ["8.5", "8.6"],
        "what_it_shows": (
            "GPU-friendly (pure numpy vectorized) renderer for f_c(z) = z²+c. "
            "Mandelbrot set: color each c by escape time of orbit of 0. "
            "Julia set: for fixed c, color each z₀ by escape time. "
            "Clicking on a point in the Mandelbrot set updates the Julia set "
            "in real time (matplotlib event loop). Demonstrates: c∈M ↔ J(f_c) "
            "connected; c∉M ↔ J(f_c) Cantor dust."
        ),
        "parameters": {
            "resolution": "800×800 default",
            "max_iter": "256",
            "escape_radius": "2.0",
            "colormap": "inferno | hot | custom smooth bands",
        },
        "algorithm": (
            "Vectorized: Z = grid of z₀, C = constant. "
            "Iterate Z ← Z²+C up to max_iter; record escape iteration n. "
            "Smooth coloring: n + 1 - log(log|Z|)/log2 for anti-banding. "
            "Mandelbrot: C = grid, Z starts at 0."
        ),
        "output": (
            "Interactive 2-panel matplotlib figure: Mandelbrot (left), "
            "Julia for clicked c (right). Click anywhere in Mandelbrot to update."
        ),
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch8_julia_mandelbrot.py",
    },

    {
        "id": "ch8_newton",
        "chapter": 8,
        "title": "Newton's Method Basins — Rational Map on ℂ̄",
        "concept": "Fatou/Julia decomposition, immediate basins, rational maps",
        "sections": ["8.4", "8.5"],
        "what_it_shows": (
            "Newton's method for p(z)=zⁿ-1 (roots are n-th roots of unity) "
            "gives a rational map R(z)=z - p(z)/p'(z) with n Fatou components "
            "(basins of attracting fixed points). The Julia set J(R) is the "
            "boundary between basins — a fractal. Color each z₀ by which root "
            "it converges to. For n=3 the fractal is especially striking."
        ),
        "parameters": {
            "degree": "3 (default), 4, 5",
            "resolution": "1000×1000",
            "max_iter": "100",
            "tol": "1e-6",
        },
        "algorithm": (
            "For each z₀ in grid: iterate z ← z - p(z)/p'(z) until |p(z)|<tol "
            "or max_iter. Identify which root (arg(z) mod 2π/n → index 0..n-1). "
            "Color by (root_index, convergence_speed)."
        ),
        "output": "Full-screen fractal: n-colored basins with Julia boundary visible.",
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch8_newton.py",
    },

    # ── CHAPTER 9 ────────────────────────────────────────────────────────────

    {
        "id": "ch9_entropy_partition",
        "chapter": 9,
        "title": "Measure-Theoretic Entropy — Partition Refinement",
        "concept": "Entropy of partitions, Kolmogorov-Sinai theorem, information",
        "sections": ["9.1", "9.3", "9.4"],
        "what_it_shows": (
            "For the doubling map T(x)=2x mod 1 with the natural partition "
            "ξ={[0,0.5),[0.5,1)}, computes H(ξ^n) = H(ξ ∨ T⁻¹ξ ∨ … ∨ T^{-(n-1)}ξ) "
            "for n=1..15. This grows linearly: H(ξ^n) = n·log2 (binary entropy). "
            "The slope is the metric entropy h(T,ξ) = log2. Compares with the "
            "irrational rotation (h=0): H(ξ^n) grows like log(n) not linearly."
        ),
        "parameters": {
            "map": "doubling | rotation_alpha",
            "alpha": "(√5-1)/2",
            "n_max": "20 (partition levels)",
            "n_points": "100000 (Monte Carlo for measure)",
        },
        "algorithm": (
            "Generate n_points uniform samples. For each n, compute the "
            "joined partition ξ^n (2^n intervals for doubling map). "
            "Estimate µ(C_i) by counting samples. H = -Σ µ_i log µ_i. "
            "Plot H(ξ^n) vs n; fit line to extract slope = h."
        ),
        "output": (
            "Plot of H(ξ^n) vs n for both maps: doubling (linear, slope=log2≈0.693) "
            "and rotation (sublinear, slope→0). Annotated with h values."
        ),
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch9_entropy.py",
    },

    {
        "id": "ch9_variational",
        "chapter": 9,
        "title": "Variational Principle — h(f) = sup_µ h_µ(f)",
        "concept": "Variational principle, measure-theoretic vs topological entropy",
        "sections": ["9.5"],
        "what_it_shows": (
            "For the logistic map f_r at various r, compares: "
            "(a) topological entropy h(f_r) estimated via spanning sets (ch2), "
            "(b) metric entropy h_µ(f_r) for the natural SRB measure µ, "
            "estimated via the Shannon-McMillan-Breiman theorem: "
            "-(1/n)log µ(ξ^n(x)) → h_µ(T) a.s. "
            "Variational principle: the two should agree at the measure of "
            "maximal entropy. Plots both curves on the same axes."
        ),
        "parameters": {
            "r_values": "np.linspace(3.5, 4.0, 100)",
            "n_smb": "500 (SMB averaging length)",
            "n_orbit": "50000",
        },
        "algorithm": (
            "SMB: generate long orbit, track partition element at each step, "
            "accumulate log-measure, divide by n. "
            "Topological entropy: spanning-set estimate from ch2 program. "
            "Both plotted vs r with h=log2 reference line at r=4."
        ),
        "output": "Single plot: h_top(r) and h_µ(r) vs r, with h=log2 horizontal ref.",
        "dependencies": ["numpy", "matplotlib"],
        "file": "ch9_variational.py",
    },
]

if __name__ == "__main__":
    # Print summary
    from collections import defaultdict
    by_ch = defaultdict(list)
    for s in PROGRAM_SPECS:
        by_ch[s["chapter"]].append(s)

    for ch in sorted(by_ch.keys()):
        print(f"\n{'='*60}")
        print(CHAPTER_SUMMARIES[ch])
        print(f"\nPrograms ({len(by_ch[ch])}):")
        for s in by_ch[ch]:
            print(f"  [{s['id']}]  {s['title']}")
            print(f"    Concept: {s['concept']}")
            print(f"    Sections: {', '.join(s['sections'])}")
            print(f"    Output: {s['output'][:80]}...")
            print()
