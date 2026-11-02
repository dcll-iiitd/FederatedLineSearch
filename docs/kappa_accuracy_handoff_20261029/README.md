# Agent handout: conditional kappa_f-accuracy experiments

Prepared 29 October 2026 (Asia/Kolkata). Status: all five conditional studies below are complete. This is an evidence handoff, not a claim that the paper's stochastic assumption has been verified globally. No new training was launched to prepare it.

## 1. Assignment for the receiving agent

Address this reviewer concern honestly and concisely:

> Could the authors further clarify the practical applicability of the kappa_f-accuracy assumption, particularly when its sufficient condition is not empirically verified?

The user has a **1,500-character total response budget across the reviewer's concerns**, not necessarily 1,500 characters for this point alone. Use the compact draft in section 12 if other concerns need space. Do not cherry-pick favorable states, call a pooled average verification, or claim manuscript revisions have already been made.

Main conclusion: the condition is practically restrictive in the tested stochastic settings. Exhaustive near-full-batch evaluation supports it at some frozen client states, but **no tested stochastic configuration was supported across every tested state** (batch-50 classifications are Monte Carlo estimates; leave-one-out classifications are exhaustive numerical averages). Full-local-batch controls satisfy the function-estimation requirement mathematically. This is useful scope clarification, not uniform stochastic verification.

The latest fixed-state studies are all **FedSLS, reset option 2**. Earlier exploratory runs included option 1, but they do not constitute conditional verification of option 1 or of the main opt=1 experiments. FedExpSLS was not studied by these conditional runs.

## 2. Mathematical target and correct interpretation

Let s=(training seed, client i, global round t, local step k, pre-step history) denote a fixed conditioning state. Write w for its pre-step model. Draw the current minibatch xi independently of the past under the sampling law specified below, then run the existing local Armijo solver:

- g(xi) = grad f_i(w,xi);
- eta(xi) = the actual accepted Armijo step;
- w_plus(xi) = w - eta(xi) g(xi).

Define

```text
A(xi) = |f_i(w,xi) - f_i(w)|
B(xi) = |f_i(w_plus(xi),xi) - f_i(w_plus(xi))|
D(xi) = eta(xi)^2 * ||g(xi)||^2
```

The assumption asks for both

```text
E[A | F] <= kappa_i * E[D | F]
E[B | F] <= kappa_i * E[D | F]
```

with a client constant kappa_i applicable across the required histories/states and

```text
kappa_f = max_i kappa_i < c / (2 * eta_lmax).
```

When E[D|F]>0, the diagnostic ratios at the fixed state are

```text
R_a(s) = sum_over_trials(A) / sum_over_trials(D)
R_b(s) = sum_over_trials(B) / sum_over_trials(D).
```

Trial counts cancel, so these are ratios of estimated or exhaustively computed conditional means. **Never average A/D or B/D. Never pool different clients, changing iterates, local steps, or global rounds to estimate this conditional expectation.** Taking maxima across states happens only AFTER the separate conditional ratios have been formed.

At a finite collection S of tested states, a common admissible constant exists only if

```text
max_{s in S} max(R_a(s), R_b(s)) < c / (2 * eta_lmax),
```

subject to numerical/estimation qualifications. A high passing percentage is not enough. Even a pass at every finite tested state would not establish the assumption for all unobserved histories.

The SAME xi is used to compute g, select eta, produce w_plus, and evaluate both minibatch losses. There is no fresh evaluation batch or held-out split. The post-step estimate intentionally retains dependence between xi and w_plus(xi).

A distinction important for wording: B is an **absolute estimation error**, not a signed bias estimator. R_b-R_a is not an identified, pure measurement of statistical bias; the evaluation iterate and its distribution also change. R_b need not exceed R_a at every state.

## 3. Shared experimental protocol actually executed

These are a reduced, completed study, not the much larger proposed protocol in the code worktree.

| Item | Executed setting |
|---|---|
| Algorithm | FedSLS: local SGD with existing Armijo solver; server step eta_g=1 |
| Reset policy | Option 2, restart at eta_lmax at each local step |
| Armijo coefficient | c=0.1 |
| Caps | 0.1 for all four datasets; additional CIFAR-10 study at 0.01 |
| Threshold | 0.5 at cap 0.1; 5 at cap 0.01 |
| Training / initialization seed | 0 for every completed conditional study |
| Partition seed | 0 where partitioning is random |
| Communication rounds | 100, indexed t=0,...,99 |
| Selected clients per round | 20, uniformly sampled without replacement |
| Local steps per selected client | K=20 |
| Local optimizer | Fresh SLS optimizer for each client invocation |
| Backtracking | Existing factor 0.9, up to 100 trial evaluations; existing fallback 1e-6 |
| Server aggregation | Client deltas weighted by local sample count, normalized over selected clients |
| Augmentation / clipping / local weight decay | All disabled |
| Training precision | Existing float32 solver; no float64 retraining |
| Statistical accumulation | Python math.fsum over recorded scalar terms |
| Diagnostic reference chunks | 256 examples |
| Frozen global broadcasts | Before rounds t=0 and t=99 |
| Frozen local positions | Immediately before steps k=1 and k=20 |
| Model selection / passing-state selection | All four predeclared positions retained, irrespective of results |

### 3.1 Sampling law

Each local step draws a fresh uniform subset of exactly b distinct local examples from the client's entire training set. Draws are independent across local steps; no example repeats within a batch. Subset indices are sorted before evaluation.

This is a **standalone theory-aligned sampling study**, implemented in a separate driver. Historical main.py experiments use a shuffled-epoch loader, whose batches are not independent across all local steps. The new driver deliberately specifies a different sampling law without changing the existing main algorithm paths. Do not claim the new training trajectories are bit-identical to historical shuffled-epoch runs.

Training, client selection, diagnostic batches and closure seeds use separated seeded streams. The study does not force deterministic GPU algorithms or CUBLAS_WORKSPACE_CONFIG. Cross-GPU bit-identical replay is not claimed.

### 3.2 Fixed-state diagnostic branches

For each saved broadcast and each client:

1. Start a separate diagnostic local rollout from that broadcast.
2. Freeze immediately before local steps 1 and 20.
3. Copy the frozen model/buffers and optimizer into a scratch diagnostic instance.
4. Restore that exact starting state before EVERY trial; diagnostic updates are not chained.
5. Compute that trial's own gradient, Armijo search, accepted eta, and post-step model.
6. Measure A, B and D using its own same minibatch.
7. Restore the frozen state before the next trial.

The shadow rollouts cover **all study clients**, including clients not selected in that training round. They do not send updates to the training server. Their k=20 states are diagnostic local histories, not falsely labeled as actual selected-client histories.

Each batch setting has its **own complete training trajectory and its own late broadcasts/local states**. Batch-setting differences therefore combine sampling effects with changed optimization trajectories; this is not a matched-iterate causal batch-size comparison at late checkpoints.

### 3.3 Reference objective

Here, f_i is the finite local empirical training objective:

```text
f_i(w) = (1/n_i) * sum_{j=1}^{n_i} cross_entropy(model_w(x_j), y_j).
```

Every client in these conditional studies has at most 500 examples, so **all local examples are used**. There is no 1,024-example approximation, moving subset, held-out set, test-set reference, or probe minibatch.

Evaluation uses eval mode, torch.no_grad(), chunks of 256, sum-reduced losses weighted by example count, then division by n_i. The original module modes are restored. The pre-step reference is cached once per frozen state; the post-step reference is evaluated at each trial's own w_plus.

The chosen CNN, logistic-regression and LSTM models have no dropout or BatchNorm, so the eval/train distinction does not define different mathematical functions here. Floating-point kernels and reduction order can still differ.

## 4. Datasets and models

| Dataset | Model | Study clients | Local n_i | Total local training examples | Batch settings |
|---|---|---:|---:|---:|---|
| CIFAR-10 | Existing CNN, 10 outputs | 100 | 500 each | 50,000 | 50, 499, 500 |
| CIFAR-100 | Existing CNN, 100 outputs | 100 | 500 each | 50,000 | 50, 499, 500 |
| FEMNIST | Logistic regression, 62 outputs | First 100 natural writers | 69-392 | 24,931 | 50, n_i-1, n_i |
| Shakespeare | LSTM, 68 outputs | 43 natural writers | 52-492 | 9,514 sequences | 50, n_i-1, n_i |

CIFAR uses the existing equal-size Dirichlet label partition with concentration 0.3, seed 0. Existing normalization is retained: channel means [0.491,0.482,0.447], standard deviations [0.247,0.243,0.262], for both CIFAR datasets.

The CNN has two 5x5 convolutional layers (32,64 channels) with ReLU/max pooling, then fully connected layers 384,192,number_of_classes. FEMNIST uses the existing flattened 784-feature linear classifier. These conditional CNN results must not be relabeled as ResNet-18 results.

FEMNIST retains natural writer partitions; passing alpha=0.3 does not re-Dirichlet-partition them. Its 100 clients are a study subset, not the total number of clients in the complete FEMNIST dataset.

Shakespeare includes all 43 natural users in the available split with **50 < n_i <= 500**. No examples within these users were truncated. This is a selected small-writer cohort, not representative coverage of all 143 users in that source split. The LSTM uses embedding dimension 32, hidden dimension 64, one recurrent layer, no dropout, sequence length 80 and the existing global 68-character mapping. Test evaluation uses the same writers' corresponding test sets.

## 5. Exact enumeration versus Monte Carlo; study size

At each frozen state:

- Batch 50: M=128 independent Monte Carlo subset draws. Report point estimates only; no simultaneous confidence certification was performed.
- Leave-one-out, b=n_i-1: enumerate every one of the n_i equally likely omitted-example choices. This averages the entire finite minibatch distribution, up to numerical implementation effects.
- Full local batch, b=n_i: the only possible subset is the whole local dataset, so one trial suffices.

| Study | States per batch setting | Batch-50 trials | Leave-one-out trials | Full trials | Total trials |
|---|---:|---:|---:|---:|---:|
| CIFAR-10, cap 0.01 | 400 | 51,200 | 200,000 | 400 | 251,600 |
| CIFAR-10, cap 0.1 | 400 | 51,200 | 200,000 | 400 | 251,600 |
| FEMNIST, cap 0.1 | 400 | 51,200 | 99,724 | 400 | 151,324 |
| CIFAR-100, cap 0.1 | 400 | 51,200 | 200,000 | 400 | 251,600 |
| Shakespeare, cap 0.1 | 172 | 22,016 | 38,056 | 172 | 60,244 |

Total across all five studies: **15 training trajectories, 5,316 configuration-specific frozen states and 966,368 diagnostic trials**. All 285 queued tasks completed. These are not 966,368 independent training runs or training seeds.

## 6. Main cross-dataset results: cap 0.1, threshold 0.5

A state counts below the threshold only when BOTH conditional ratios are strictly below it. Maxima of R_a and R_b are separate and may occur at different states.

| Dataset | Batch | max R_a | max R_b | States with both below 0.5 | Clients below at all four states |
|---|---|---:|---:|---:|---:|
| CIFAR-10 / CNN | 50, MC | 15.703453 | 14.822489 | 7/400 (1.75%) | 0/100 |
| CIFAR-10 / CNN | 499, exhaustive | 6.266708 | 5.635057 | 216/400 (54.0%) | 0/100 |
| CIFAR-10 / CNN | 500, full | 0.00098228 | 0.00050177 | 400/400 (100%) | 100/100 |
| CIFAR-100 / CNN | 50, MC | 3.813480 | 8.722232 | 0/400 (0%) | 0/100 |
| CIFAR-100 / CNN | 499, exhaustive | 14.071355 | 13.210560 | 286/400 (71.5%) | 0/100 |
| CIFAR-100 / CNN | 500, full | 0.00253108 | 0.00256372 | 400/400 (100%) | 100/100 |
| FEMNIST / logistic regression | 50, MC | 2.747264 | 7.511846 | 0/400 (0%) | 0/100 |
| FEMNIST / logistic regression | n_i-1, exhaustive | 8.788303 | 8.750694 | 201/400 (50.25%) | 0/100 |
| FEMNIST / logistic regression | n_i, full | 0.00127215 | 0.00082090 | 400/400 (100%) | 100/100 |
| Shakespeare / LSTM | 50, MC | 18.819791 | 19.273099 | 0/172 (0%) | 0/43 |
| Shakespeare / LSTM | n_i-1, exhaustive | 9.088819 | 9.099989 | 36/172 (20.93%) | 0/43 |
| Shakespeare / LSTM | n_i, full | 0.01231469 | 0.00908149 | 172/172 (100%) | 43/43 |

The Monte Carlo rows are **estimated classifications**, not established expectation violations or confidence-certified passes. The exhaustive rows are numerical expectations over the full specified finite minibatch law at those frozen states. Clear exhaustive violations provide direct evidence against a uniform condition over those states.

Passing-state percentages are not probabilities that the assumption holds, percentages of the complete training trajectory, or fractions of individual minibatches that satisfy a pointwise inequality. The theoretical requirement is on conditional expectations and a common client constant, not on most individual draws.

### 6.1 Where leave-one-out passes

Numbers are clients for which both conditional ratios are below 0.5.

| Frozen position | CIFAR-10 | CIFAR-100 | FEMNIST | Shakespeare |
|---|---:|---:|---:|---:|
| t=0, k=1 | 100/100 | 100/100 | 100/100 | 24/43 |
| t=0, k=20 | 16/100 | 86/100 | 1/100 | 8/43 |
| t=99, k=1 | 100/100 | 100/100 | 100/100 | 4/43 |
| t=99, k=20 | **0/100** | **0/100** | **0/100** | **0/43** |

Thus there are non-full stochastic regimes/states with numerical support for the condition. For CIFAR-10, CIFAR-100 and FEMNIST, every tested client passes at k=1 of the late broadcast. However, no client passes at k=20 there. Restricting attention to k=1 would describe a restricted finite set, not verify the original K=20 assumption.

### 6.2 Additional CIFAR-10 cap-0.01 study

Here c=0.1, reset 2, eta_lmax=0.01, threshold **5**, not 0.5.

| Batch | max R_a | max R_b | States with both below 5 | Clients below at all four |
|---|---:|---:|---:|---:|
| 50, MC | 306.907995 | 319.119991 | 2/400 (0.5%) | 0/100 |
| 499, exhaustive | 76.056221 | 76.956146 | 296/400 (74.0%) | 0/100 |
| 500, full | 0.01800253 | 0.01467907 | 400/400 (100%) | 100/100 |

Leave-one-out passing counts by (t,k) were 99,97,100,0 at (0,1),(0,20),(99,1),(99,20). The bigger threshold did not establish a uniform stochastic condition.

Do not compare raw R values across caps without their corresponding thresholds. Reducing the cap increases the threshold, but also changes accepted steps, D, and the training trajectory. If eta tracks the cap and errors/gradient scale are held fixed, D decreases quadratically while the threshold increases only inversely; a smaller cap can therefore make the normalized test HARDER. This is a scaling explanation, not a proof of monotonicity along changed training trajectories.

## 7. Mechanism illustrated by recorded states

These are evidence-based examples, not a complete causal decomposition of every failure.

### 7.1 CIFAR-100, leave-one-out, client 87, late broadcast

| Quantity | Before k=1 | Before k=20 |
|---|---:|---:|
| E[A] | 0.0022726602 | 0.0024800702 |
| E[B] | 0.0036541329 | 0.0023283554 |
| E[D] | 0.0092616978 | 0.0001762496 |
| Mean accepted eta | 0.0052334763 | 0.0025031555 |
| R_a | 0.245383 | 14.071355 |
| R_b | 0.394542 | 13.210560 |

The denominator falls roughly 53-fold while function errors remain comparable. At k=20 the permitted error is 0.5*E[D]=0.0000881248, versus observed errors around 0.0023-0.0025. Both inequalities fail, well away from the threshold.

The accepted eta is much less than cap 0.1. Substituting 0.1 into D would manufacture a different, overly favorable diagnostic.

### 7.2 Shakespeare, leave-one-out, client 31, late broadcast

| Quantity | Before k=1 | Before k=20 |
|---|---:|---:|
| E[A] | 0.0314214795 | 0.0228151092 |
| E[B] | 0.0302135918 | 0.0228431509 |
| E[D] | 0.0215514696 | 0.0025102393 |
| Mean accepted eta | 0.1 | 0.1 |
| R_a | 1.457974 | 9.088819 |
| R_b | 1.401927 | 9.099989 |

Here eta remains at the cap, so shrinking accepted steps cannot be the sole explanation. The gradient-related denominator falls about 8.6-fold; errors decrease much less. The condition fails despite nearly full batches.

### 7.3 Earlier CIFAR-10 example at cap 0.01

Client 31, t=99,k=20, b499:

```text
E[A] = 0.0010507819
E[B] = 0.0010632152
E[D] = 0.00001381586
eta  = 0.01
bound * E[D] = 0.00006907929
R_a = 76.056221; R_b = 76.956146; threshold = 5
```

This is not a line-search failure or a zero denominator. Even tiny leave-one-out loss discrepancies are too large relative to the required squared-update scale.

### 7.4 The second condition is genuinely separate

For CIFAR-100 b50 at t=99,k=1, every client's estimated R_a is below 0.5 (largest 0.365907), while none passes R_b (largest 3.308614). The post-update requirement cannot be inferred merely from a favorable pre-update diagnostic. This pattern is consistent with the relevance of same-minibatch adaptation; it does not isolate its causal bias magnitude.

## 8. Full-batch controls and denominator/failure policy

### 8.1 Why full-batch ratios should be zero

For a finite local empirical objective, f_i(w,xi_full)=f_i(w) for all w, and the same identity holds at w_plus. Hence A=B=0 mathematically. If E[D]>0, both ratios are zero; any positive kappa below the bound satisfies these function-accuracy inequalities.

Small recorded full-batch residuals are not stochastic estimation error. Whole-batch versus chunked sum reductions and batch-dependent floating-point kernels produce slightly different numerical values. For example, the Shakespeare full-batch state with the largest R_a has absolute pre/post discrepancies approximately 9.13e-6 and 5.42e-6. We did not subtract these residuals, force them to zero, or use them to correct stochastic measurements.

Full-batch satisfaction is a sanity check and an exact-function regime. It does not validate a genuinely stochastic minibatch assumption, other hypotheses of the theorem, or the quality/efficiency of the trained model.

### 8.2 Policy in the conditional studies

- Keep every trial, including zero/tiny D and failed-search fallbacks.
- No epsilon, clipping, omission, or redraw-until-success.
- If a search fails, explicitly evaluate the actual fallback batch loss, retain its A/B/D, and flag that the step was not Armijo-accepted.
- Nonfinite values invalidate normal finite-ratio certification; they are not silently dropped.
- If mean D=0 and A=B=0, inequalities are trivial but ratios undefined: label degeneracy.
- If mean D=0 and either exact numerator is positive, no finite kappa satisfies that state.
- Observing zeros in Monte Carlo alone does not establish a zero true expectation.
- Individual zero D contributions can remain in a ratio of sums as long as total D is positive; there is no per-step division.

In **all five completed conditional studies**, recorded training and diagnostic searches had zero failures; all diagnostic states were complete and finite; there were zero trials with D=0 or 0<D<1e-8. Thus the reported violations are not artifacts of failure filtering or numerical division by a near-zero denominator.

The statistical summaries use Python scalar accumulation, but the underlying model/gradient evaluations remain float32. Exhaustive enumeration removes minibatch Monte Carlo error, not floating-point error or uncertainty about unobserved histories. Numerically marginal classifications should not be promoted to a rigorous proof without a precision audit.

## 9. Training outcomes: context, not assumption verification

Final metrics are after 100 training updates, row t=99. They come from the conditional-study trajectories, not the main paper's longer baseline comparisons.

| Dataset | Cap | Batch | Final training loss | Final test accuracy (%) |
|---|---:|---|---:|---:|
| CIFAR-10 | 0.01 | 50 | 1.522397 | 44.22 |
| CIFAR-10 | 0.01 | 499 | 1.587992 | 41.45 |
| CIFAR-10 | 0.01 | 500 | 1.588691 | 41.47 |
| CIFAR-10 | 0.1 | 50 | 0.855804 | 61.72 |
| CIFAR-10 | 0.1 | 499 | 1.306112 | 52.05 |
| CIFAR-10 | 0.1 | 500 | 1.306998 | 52.02 |
| CIFAR-100 | 0.1 | 50 | 2.360422 | 28.58 |
| CIFAR-100 | 0.1 | 499 | 3.519382 | 15.78 |
| CIFAR-100 | 0.1 | 500 | 3.521810 | 15.74 |
| FEMNIST | 0.1 | 50 | 1.770205 | 57.6691 |
| FEMNIST | 0.1 | n_i-1 | 1.842191 | 58.9090 |
| FEMNIST | 0.1 | n_i | 1.843417 | 58.8027 |
| Shakespeare | 0.1 | 50 | 2.312977 | 35.1250 |
| Shakespeare | 0.1 | n_i-1 | 2.311910 | 34.9167 |
| Shakespeare | 0.1 | n_i | 2.311297 | 35.1250 |

These results show measured training progress can coexist with nonpassing diagnostics. They do not establish asymptotic convergence outside the theorem, prove that the condition is necessary, or compare FedSLS favorably against unmeasured baselines.

## 10. Exploratory history: do not conflate with the conditional evidence

### 10.1 Earlier 500-round pooled trajectory diagnostics

Initial instrumentation measured one actual batch per changing training step, then formed a ratio of sums over clients/local steps in each communication round. That is a denominator-weighted aggregate over different states, NOT a conditional expectation at one frozen state.

At round 490, reset 2, c=0.1, cap 0.1, bound 0.5, archived reports give:

| Dataset/model | Batch/seed | Pooled R_a | Pooled R_b |
|---|---|---:|---:|
| CIFAR-10 / ResNet-18 | b50, seed0 | 1.0267113 | 1.2929753 |
| CIFAR-10 / ResNet-18 | b128, seed0 | 1.5808621 | 1.1980374 |
| CIFAR-10 / ResNet-18 | b256, seed0 | 53.241212 | 35.359991 |
| FEMNIST / logistic regression | b50, seed0 | 1.1098339 | 2.3183924 |
| FEMNIST / logistic regression | b50, seed1 | 1.1494120 | 2.4683787 |

These numbers are context only. The older plotter excludes failed-search rows; the new conditional protocol retains every outcome. Older kappa_main.pdf and kappa_batchsize_appendix.pdf show pooled diagnostics, not the fixed-state evidence above.

Larger batches did not monotonically improve those trajectory-level ratios. A heuristic about decreasing sample error alone does not control accepted steps, gradient magnitudes, or changed model trajectories.

### 10.2 Subsequent opt1/opt2 and large-batch screens

Folders kappa_screen_20261027_v2 and kappa_largebatch_20261028 contain additional pooled screens, including multiple reset policies and some second seeds. Some CIFAR-10 CNN near-full settings had pooled ratios below threshold at every measured round. This motivated the proper conditional experiment; it did NOT verify the original client/step-conditional assumption.

Do not use exploratory second seeds as if the conditional studies were replicated across seeds. They were not.

### 10.3 Reset option 1 caveat

Existing Armijo opt1 has no explicit fixed cap on trial eta. It multiplies the previous step by gamma^(1/n_batches_per_epoch), here 2^(1/500); backtracking may shrink it. eta_max=10 belongs to a different Goldstein branch, not an Armijo cap.

For a fresh 20-step local optimizer, an upper envelope is max(initial_eta,1e-6)*2^(20/500), about 1.0281138 times initial_eta for the tested initial values. This requires the stated local horizon and fresh initialization, not arbitrary unbounded training. Prior screen metadata used the envelope, yielding threshold about 0.486327 at initial 0.1. It is not correct to declare the opt1 initial value a hard cap.

The completed fixed-state studies are opt2 only. **Do not assert conditional validation or conditional failure measurements for opt1 from these studies.**

### 10.4 Superseded proposed protocol

The file FederatedLineSearch-kappa/docs/kappa_conditional_experiment_protocol.md describes an earlier, much larger PROPOSAL: more batch sizes, opt1+opt2, extra checkpoints, confirmation seeds and confidence intervals. That full proposal was not executed. The RUN.md and manifest.json in each completed root are authoritative for what actually ran.

## 11. Interpretation for the paper and reviewer

Supported statements:

1. We directly examined both conditional inequalities, retaining intentional same-minibatch reuse, rather than relying solely on pooled round-level curves.
2. Full-local-batch function estimates satisfy the empirical function-accuracy requirement by construction.
3. Exhaustive near-full sampling supports the inequalities at some frozen states, but clear failures occur at others.
4. In the four cap-0.1 studies, no near-full client passed at the late broadcast's local step 20, although many earlier local states passed.
5. The condition is restrictive relative to the tested stochastic trajectories. The squared update scale can decrease while loss-estimation errors persist.
6. Convex logistic regression on FEMNIST did not by itself ensure the condition.
7. These experiments delimit the sufficient theoretical regime; they do not turn a conditional theorem into an unconditional guarantee.

Unsupported statements to avoid:

- “We verified kappa_f-accuracy empirically across our datasets.”
- “It holds with probability 71.5%,” “for 71.5% of clients,” or “through 71.5% of training.”
- “Full-batch success validates ordinary stochastic minibatching.”
- “A favorable pooled average verifies each client's conditional inequality.”
- “All observed training performance is covered by our theorem.”
- “Failure of this assumption invalidates the conditional theorem or proves FedSLS diverges.”
- “Convexity alone guarantees kappa_f-accuracy.”
- “R_b must exceed R_a; their difference is exactly the bias.”
- “Increasing batch size or reducing eta necessarily fixes the condition.”
- “We tested every local/global step or replicated these conditional results over several training seeds.”
- “The conditional experiment was performed on ResNet-18, FedExpSLS, or reset 1.”

The manuscript's stated logical implications (including global kappa_f accuracy implying strong-growth/interpolation properties under their hypotheses) must be discussed in the correct direction. We did not measure SGC or interpolation here. Failure to verify kappa_f does not prove SGC fails; observing SGC would not automatically prove kappa_f. Finite region-restricted evidence must not be substituted for a global-in-w condition.

Whether a theorem covers a particular model also depends on its OTHER hypotheses. These measurements alone do not check those.

Practical recommendation: make the theorem's sufficient-condition scope explicit; acknowledge that the experiments do not generally verify it for stochastic minibatches; use the exact-function and restricted-state results as limited regime evidence, not a rescue claim. Do not keep searching until a favorable cherry-picked configuration can be presented as general applicability.

## 12. Draft language the receiving agent may adapt

### 12.1 Compact rebuttal draft (393 characters, excluding this heading)

The analysis uses kappa_f-accuracy as a sufficient condition. Conditional tests on CIFAR-10/100, FEMNIST and Shakespeare found that leave-one-out batches satisfy both inequalities at some states, but not uniformly; full batches satisfy them by construction. We will clarify this practical restrictiveness and distinguish the theorem's conditional guarantee from empirical training performance.

### 12.2 Detailed rebuttal draft (855 characters, excluding this heading)

Our analysis uses kappa_f-accuracy as a sufficient condition, not an automatically verified property of every experiment. We evaluated both conditional inequalities at frozen client states on CIFAR-10/100, FEMNIST and Shakespeare, using the same minibatch for gradient, Armijo search, update and loss estimates. Exhaustive leave-one-out evaluation satisfied both inequalities at 54.0%, 71.5%, 50.25% and 20.9% of tested states, respectively, but not at all states; full batches satisfy them by construction. These findings show regime dependence and practical restrictiveness, not uniform stochastic validity. The runs still made training progress, which does not establish the theorem's hypotheses or make this sufficient condition necessary for progress. We will clarify these limits and distinguish the conditional guarantee from empirical performance.

Both are below 1,500 characters individually, but that limit is shared across concerns. “We will clarify” is prospective: no manuscript was edited in preparing this handout. If the revision has actually been made, the receiving agent may change that wording truthfully.

### 12.3 Expanded experimental paragraph

We assessed the two conditional function-accuracy inequalities using FedSLS with reset option 2, Armijo c=0.1 and maximum trial step 0.1 (threshold 0.5), on CIFAR-10/CNN, CIFAR-100/CNN, FEMNIST/logistic regression and Shakespeare/LSTM. We trained for 100 communication rounds with 20 participating clients and 20 local steps, fixing training seed 0. At the initial and late broadcasts (t=0,99), independent client-local rollouts froze states before steps k=1,20. At each state, independent one-step branches restored the same model and optimizer and retained the same minibatch for the gradient, Armijo selection, update and both batch-loss estimates. Full local empirical losses were evaluated without gradients over all client data. We used 128 Monte Carlo branches for batch 50, exhaustive enumeration of all omitted examples for leave-one-out batches, and a full-batch control. Numerator and denominator expectations were formed separately at each fixed state. Leave-one-out evaluation placed both ratios below 0.5 at 216/400, 286/400, 201/400 and 36/172 states, respectively, but no client passed all four states in any stochastic setting. Full-batch controls passed throughout, as expected from exact empirical function evaluation. The results support a regime-dependent and restrictive sufficient condition rather than uniform stochastic validity.

## 13. Evidence locations and reproducibility handoff

Main results root:

```text
/home/somya/rungeetika/FederatedLineSearch/results
```

Code worktree:

```text
/home/somya/rungeetika/FederatedLineSearch-kappa
branch: kappa-accuracy
```

The main checkout remains on FedSLS_update. Do not switch branches, discard user changes, or overwrite original experiment outputs merely to inspect these results.

| Study ID | Directory under main results root |
|---|---|
| CIFAR-10, cap 0.01 | kappa_conditional_20261028 |
| CIFAR-10, cap 0.1 | kappa_conditional_cifar10_eta0p1_20261028 |
| FEMNIST, cap 0.1 | kappa_conditional_femnist_eta0p1_20261028 |
| CIFAR-100, cap 0.1 | kappa_conditional_cifar100_eta0p1_20261029 |
| Shakespeare, cap 0.1 | kappa_conditional_shakespeare43_eta0p1_20261029 |

Within every study:

- RUN.md: executed protocol.
- manifest.json: actual configuration, source hashes, queue and expected counts.
- partition.json: client-data fingerprint and cohort sizes (older CIFAR metadata may be less detailed).
- trials/: raw per-branch CSVs, one file per frozen state.
- cells/: JSON summaries from those raw branches.
- conditional_cells.csv: combined per-state conditional results.
- client_maxima.csv: per-client maxima AFTER conditioning.
- batch_summary.csv: worst ratios, count classifications, numerical/failure flags.
- training/b50.csv etc.: all training-round timings and recorded metrics.
- checkpoints/: broadcasts and final trained models.
- anchors/: saved/reconstructible local conditioning states and optimizer states.
- gpu*_status.jsonl and logs/: task completion evidence.
- plots/conditional_maxima.pdf and .png: worst conditional ratios by batch.
- plots/client_state_heatmap.pdf: variation over client and frozen position.
- plots/training_curves.pdf: contextual loss/accuracy curves.

The field candidate_common_kappa_on_tested_states in batch_summary.csv is a bookkeeping midpoint between an observed supported maximum and the bound, NOT a fitted theoretical constant. Do not quote approximately 0.25 as an empirically estimated kappa. For exact full batch the mathematical errors are zero.

CIFAR-100 and Shakespeare also archive source_snapshot/. The earlier three roots record source hashes without a full root-local source snapshot. Preserve that distinction; a current driver may have been extended since those earlier runs.

Relevant implementation:

```text
scripts/run_kappa_conditional.py       # training, shadow anchors, tasks, manifests
scripts/kappa_conditional_core.py      # independent branches, full references, means
scripts/summarize_kappa_conditional.py # per-state/client/batch reports
SLS/sls.py                            # step_with_seed and step_with_kappa
SLS/utils.py                          # reset_step and Armijo condition
tests/test_kappa_conditional.py
tests/test_kappa_conditional_datasets.py
tests/test_kappa_screen.py
```

SLS source SHA-256 recorded for the newest studies:

```text
SLS/sls.py   79443eb42c63c10f6447d8591dee96cb4a6b5a5738244575581ccc918249dde4
SLS/utils.py e2f301f8b1056fa505e51d892934f22d99d41ef08111372b8fb680c96d592c12
```

Recompute derived reports, if needed, WITHOUT training or altering raw CSVs:

```bash
cd /home/somya/rungeetika/FederatedLineSearch-kappa
/home/somya/miniconda3/envs/fls/bin/python scripts/summarize_kappa_conditional.py \
  --root /home/somya/rungeetika/FederatedLineSearch/results/kappa_conditional_cifar100_eta0p1_20261029
```

Replace only the study-directory suffix for another root. This rewrites derived reports/plots, so it is not a purely read-only inspection command. Read existing reports first.

Two raw files independently audited while preparing the preceding result discussion:

```text
kappa_conditional_cifar100_eta0p1_20261029/trials/b499_t99_i087_k20.csv
  500 trials -> R_a=14.071355254677771, R_b=13.210560297358118
kappa_conditional_shakespeare43_eta0p1_20261029/trials/bloo_t99_i031_k20.csv
  55 trials -> R_a=9.088818592153249, R_b=9.099989499624535
```

Companion files in this handout folder:

- conditional_results.csv: 15 configuration rows, exact saved metrics and ratios.
- conditional_state_breakdown.csv: 60 rows, per-position passing counts and maxima.
- rebuttal_drafts.txt: copyable drafts with explicit character counts.

These companion tables were reconstructed from per-state JSON summaries and training CSVs, not by changing any measurement file.

## 14. Verification status and remaining limitations

Measurement-on versus measurement-off was previously checked on CPU: identical model parameters, BatchNorm buffers and preserved RNG in the tested case. The user accepted that verification; it is DONE and was not repeated for this handout. Existing test_kappa_screen.py additionally covers 20 steps with both reset policies on a model containing BatchNorm and dropout. Do not inflate a tested invariance property into a formal proof for every possible model or GPU kernel.

The conditional core has focused tests for manual/exhaustive branch agreement, frozen model/optimizer/RNG restoration, retained failure fallbacks, full-batch/zero-denominator handling, and reproducible varying subset draws. Dataset-specific tests cover variable client sizes, classifier sizes, the Shakespeare cohort/LSTM, and reporting groups/bounds.

Remaining limits:

- One training/partition seed per completed conditional configuration.
- Four predeclared states per client; no exhaustive history or full-trajectory coverage.
- Shadow local states rather than all actual selected-client histories.
- Batch-50 MC point estimates without confidence bands or multiplicity correction.
- Exact expectations apply to finite local empirical data and the specified IID subset law.
- Smaller natural-writer cohorts for FEMNIST/Shakespeare.
- Float32 function/gradient evaluations and numerical residuals.
- Different batch settings produce different later training trajectories.
- No conditional opt1 or FedExpSLS study in this completed evidence set.
- No direct test of SGC/interpolation or other theorem hypotheses.

Bottom line for the receiving agent: **form a strong response by accurately delimiting the theorem and reporting the measured restriction, not by claiming that the experiments verify an assumption they do not verify.**
