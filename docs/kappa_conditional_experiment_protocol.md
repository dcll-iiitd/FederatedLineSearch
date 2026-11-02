# Proposed conditional kappa-f experiment

Status: proposed for review; no experiments from this protocol have been launched.

## Target and unit of analysis

For a fixed conditioning state s=(seed,t,i,k), freeze the pre-step local model,
all model buffers, the optimizer state, and every item in the past training
history needed by the local solver. Define, for an independent minibatch xi:

    A(xi) = abs(f_i(w,xi) - f_i(w))
    w_plus(xi), eta(xi), g(xi) = the original SLS result from that frozen state
    B(xi) = abs(f_i(w_plus(xi),xi) - f_i(w_plus(xi)))
    D(xi) = eta(xi)**2 * ||g(xi)||**2

The quantities of interest are E[A|s]/E[D|s] and E[B|s]/E[D|s], where E[D|s]>0.
Estimate the two numerator expectations and denominator expectation separately.
Do not average trial-wise A/D or B/D. Do not pool clients, local steps, server
rounds, or conditioning states when estimating these ratios.

For each client, take the maximum over the tested states and both ratios. The
configuration statistic is the maximum over all 100 clients. This corresponds
to the finite set of states being evaluated, not to every possible history.

## Training and data

* FedSLS with the existing CIFAR-10 CNN, cross-entropy loss, and server eta_g=1.
* 100 equal-sized clients, 500 local training examples each.
* Existing Dirichlet partition with concentration 0.3; fix partition seed 0
  across all comparisons. Separate partition, initialization, training, and
  diagnostic random-number streams.
* 20 participating clients per communication round; 20 local steps; 100 rounds.
* Discovery training seed 0. Independent confirmation training seeds 1 and 2.
* c=0.1; initial step 0.01; existing backtracking factor 0.9, trial limit 100,
  gamma=2, and growth scale n_batches_per_epoch=500.
* Reset options 2 and 1, evaluated separately. A new local optimizer is created
  on each client invocation, as in the current implementation.
* No augmentation, gradient clipping, or local weight-decay penalty. The chosen
  CNN has no BatchNorm or dropout, so the reference and training functions agree.
* Independent sampling across local steps: at every step draw a uniform subset
  of exactly b distinct examples from all 500 local examples. Examples do not
  repeat within one batch; batches are independent across steps. This replaces
  the shuffled-epoch loader only in a separate opt-in experimental driver.
  Present each chosen subset in ascending index order, in both training and
  diagnostics, so batch order introduces no additional random variable.
* Record training loss and test accuracy with a fixed evaluation batch size 256
  across configurations; evaluation does not change any optimization state.
* Existing algorithms, default commands, Armijo/reset rules, and old runs are
  preserved. These are new theory-consistent sampling experiments.

The batch sweep is b in {10,50,100,250,450,490,499,500}. Batch 500 is a positive
control; all other settings are stochastic. There are 16 discovery training
configurations: eight batch sizes times two reset options.

## Checkpoints and client coverage

Save the global model broadcast before communication rounds t in {0,50,99}.
For each checkpoint and each configuration, run reproducible diagnostic local
rollouts for all 100 clients, using the same 20-step solver and IID sampler.
These rollouts do not contribute updates to the training server. For clients
actually participating, the actual pre-step local states can also be retained;
clearly label actual-trajectory versus diagnostic-rollout anchors.

Freeze states immediately before local steps k in {1,10,20}. Each configuration
therefore has 100*3*3=900 conditioning states. Every client is covered; local
steps and global rounds are sampled rather than exhaustively covered.

To avoid a snapshot being selected because it already passes, collect all
predeclared states regardless of losses or earlier measurements. Save each
anchor's model/optimizer state or an auditable reconstructible checkpoint plus
its rollout RNG history, and record hashes. At a given anchor, all trial resets
must be exact tensor/state copies from the frozen anchor.

## Conditional trials

For each fixed anchor:

1. Evaluate f_i(w) once over all 500 local examples; cache this value.
2. Draw xi from the prescribed minibatch distribution using a private diagnostic
   generator, independently of the history and other diagnostic trials.
3. Restore the anchor model, buffers, and optimizer state, including previous
   accepted step size for option 1. Reusing only the model weights is insufficient.
4. Run the unmodified Armijo search. Capture its batch loss before/after, its
   accepted step, and its pre-step gradient norm. The trial's same xi is used
   for the gradient, every Armijo evaluation, the update, and both batch losses.
5. Evaluate f_i(w_plus(xi)) over all 500 examples, in inference mode, chunks of 256.
6. Record A, B, D, eta, gradient squared norm, search outcome, trial ID, batch
   indices/hash, and the anchor identity. Restore before the next trial; never
   chain diagnostic updates together.

Advance the private diagnostic sampler between trials. Restoring a model's past
state does not mean resetting the sampler to return the same xi on every trial.

Training and evaluation use separate processes or completely isolated model,
optimizer, and random states. No diagnostic sample becomes the training batch.
Reuse one scratch model and reset its tensors; do not retain M cloned models.

## Exact enumeration and Monte Carlo

For b=499 there are exactly 500 equally likely subsets, one for each omitted
example. Enumerate every subset once. Its average gives the conditional
expectation over the prescribed finite-data minibatch distribution, up to
floating-point implementation effects. No Monte Carlo interval is needed for
this enumeration. It does not remove uncertainty from untested conditioning
states or histories.

For b=500 there is one subset; evaluate it as the full-batch control.

For b in {10,50,100,250,450,490}, use M=128 independent trials per anchor in
discovery. This is an exploratory estimate, not a pass certification. Keep
independent draws even if the same subset happens to recur.

Discovery covers 14,400 anchors in total and approximately 2.2842 million
diagnostic trials, plus the ordinary training/rollouts. A short timing pilot
must benchmark b=50 and b=499 before scheduling the full study; no runtime
promise is made before that benchmark.

## Confirmation and uncertainty

Before confirmation, for each reset policy separately choose the smallest
non-full batch size that has both point-estimated conditional ratios below the
correct bound at every discovery anchor for that policy. Nominate at most one
candidate per policy. If none qualifies for a policy, report this rather than
changing the rule. Seed 0 discovery alone is not the final claim.

Run each nominated candidate with independent training seeds 1 and 2 and the
same fixed data partition. Confirmation covers all 100 clients, global checkpoints
t={0,50,99}, and ALL local steps k=1,...,20, not just the three discovery steps.
For b499, enumerate all 500 subsets at every anchor again. For an intermediate
batch, use M=1024 fresh trials per anchor, independent of discovery.

If both policies nominate b499, full confirmation has 24,000 anchors and 12 million
enumerated one-step trials. This can be expensive; the timing pilot must report
the expected compute cost before full execution. A predeclared futility stop is
allowed if an exact anchor violates a bound, or a simultaneous lower confidence
bound establishes a Monte Carlo violation. Keep the failing anchor and record
that the candidate was stopped, rather than claiming untested states passed.

For Monte Carlo confirmation, use paired resampling of (A,B,D) to form one-sided
approximate simultaneous 95% upper confidence bounds for the ratios across
the entire predeclared confirmation family: clients, checkpoints, local steps,
both inequalities, both reset policies, and both confirmation seeds. Use a
predeclared max-t bootstrap correction, rather than uncorrected 95% intervals
at thousands of cells. Preserve numerator/denominator dependence. Report all
estimates and uncertainty, including inconclusive cells; bootstrap uncertainty
is empirical, not a formal distribution-free guarantee.
Use 5,000 resamples with a separate recorded bootstrap seed; do not tune the
resampling count or family after looking at which intervals pass.

The equivalent slack variables A-bound*D and B-bound*D are also reported, with
one-sided simultaneous bounds, to avoid hiding instability in ratio division.
For an enumerated cell, there is no minibatch sampling uncertainty; report its
numerical value and margin directly.

## Threshold and uniform client constant

For opt2, eta_upper=0.01 and bound=c/(2*eta_upper)=5.
For opt1, eta_upper=0.01*2**(20/500), approximately 0.0102811383, and the bound
is approximately 4.863274737. This upper envelope applies to a fresh 20-step
local optimizer and includes the existing 1e-6 fallback because it is below
the chosen initial step. Check every returned step against the envelope.

At each client i, define U_i as the maximum of the exact ratios or simultaneous
upper confidence bounds over its tested t,k and both inequalities. A supported
configuration requires max_i U_i strictly below the relevant bound; an average
or a high pass percentage is insufficient for that claim. For the finite set of
tested states, a positive common constant strictly between max_i U_i and the
bound then exists (for example their midpoint).

If a confidence interval crosses the threshold, classify the result as
inconclusive. If a one-sided lower confidence bound exceeds the threshold,
classify it as evidence against the condition at that anchor. Do not label a
noisy point-estimate exceedance as a proved violation of the expectation.

## Failed searches, zero denominators, and precision

Include every minibatch trial in the expectation. Do not discard failed searches
or redraw until success. If the implementation returns its fallback update,
evaluate the post-fallback batch loss and full loss, record its actual eta, and
include it. Also flag the fact that fallback is not an Armijo-accepted step;
an accepted-step theory claim needs this issue resolved, not hidden by filtering.

Keep zero and tiny D contributions without clipping or epsilon. If E[D]=0 and
both numerator expectations are zero, the inequalities hold trivially but the
ratio is undefined; label this separately. If E[D]=0 and either numerator is
positive, no finite kappa satisfies that cell. For Monte Carlo, observed zeros
do not prove the true expectation is zero; mark unresolved cases inconclusive.
Nonfinite losses or gradients invalidate a normal finite-moment pass claim and
must be reported.

Use float64 for statistical accumulation, without changing the training solver's
precision. Full-batch controls quantify numerical residuals. Do not subtract
those residuals from stochastic discrepancies or clip discrepancies to zero.
Numerically marginal cells require an explicit higher-precision audit before
being described as passing.

## Verification before expensive execution

* Tiny finite dataset: compare enumerated conditional expectations with a manual
  enumeration, for both reset policies.
* Check that every trial begins with identical parameters, buffers, gradients
  cleared as required, and identical optimizer step-size state.
* Verify independent, fixed-size uniform subset sampling and separate RNG streams.
* Check that offline diagnostics do not affect the training checkpoint sequence.
* Full-batch control: A and B vanish up to reported floating-point residuals.
* Exercise failed-search and zero-denominator cases; verify no observations are
  silently dropped from the expectation.

## Outputs and permissible conclusion

Write separate raw trial files per job or anchor group; a summary row per
(configuration,seed,t,i,k); client maxima; and the overall maximum. Keep exact
enumeration and Monte Carlo outputs clearly distinguished. Record sampling
law, partition, seeds, checkpoint hashes, optimizer state, bounds, trial counts,
numerical diagnostics, and confidence procedure in metadata.

Plots should show batch size against worst tested conditional ratio, the bound,
client/step heatmaps, and Monte Carlo confidence bounds where applicable. A
round-level sum is not the main result of this protocol.

The strongest permissible statement is that the conditional inequalities are
supported at all tested client/iterate states under the specified sampling law.
Even exact enumeration at those states cannot establish the assumption for all
unobserved histories, every training round/step, or arbitrary models/datasets.
