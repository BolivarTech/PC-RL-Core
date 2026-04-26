# Predictive Coding + Backprop — An Intuitive Guide

> **Companion document to** `pc_actor_critic_paper.md`. The paper covers the formal architecture, math, and empirical results. This guide explains the same machinery in conversational prose, with metaphors and concrete numerical examples, for readers who want to build mental models before diving into the formalism.

---

## 1. The big idea, in one paragraph

Most neural networks do `input → forward pass → output` in a single sweep. Predictive Coding (PC) does something different: it has the network **deliberate** by running an iterative loop where higher layers **predict** what the layers below them should look like, lower layers **report** what they actually look like, and everyone **adjusts** until predictions and reports agree. The output is the result of that conversation, not a single feed-forward computation. The training rule (backpropagation) is then applied to the **converged state** as if it were a regular forward pass — the iterative deliberation happens during inference, but the gradient computation does not unroll through it.

That's the whole architecture. The rest of this document explains why this matters, what it looks like in practice, and how the gradient story actually works.

---

## 2. The conversation metaphor

Think of a 3-level hierarchy in a hospital:

- **Resident** (capa baja): observes raw patient symptoms.
- **Attending** (capa media): forms a working hypothesis (an interpretation).
- **Department head** (capa alta): consults broader patterns (the diagnosis).

In a feedforward MLP, the conversation is one-way only: resident → attending → head, with the head making the final call without ever asking back. If the resident misreports a symptom, the head's diagnosis carries the error all the way through.

In PC, the conversation goes both ways:

- The **head** says: "if my diagnosis (D) is right, the attending should be seeing pattern X."
- The **attending** says: "I'm seeing pattern Y, not X. Let me adjust my interpretation."
- The **attending** then tells the **resident**: "if my interpretation is correct, you should be seeing symptoms a, b, c."
- The **resident** says: "I see a, c, and slightly different from b. Let me note the discrepancy."
- They iterate. After a few rounds, everyone agrees on a coherent interpretation that matches the observed symptoms AND the higher-level diagnosis.

The "answer" is not the head's first guess. It's the **equilibrium of the conversation**.

---

## 3. The 6-step loop, with a concrete example

Suppose we have a 3-layer actor: state input → hidden 1 → hidden 2 → output. Hyperparameters: `alpha = 0.03`, `tol = 0.01`, `max_steps = 5`.

### Step 0 — Initialization

A standard feedforward computes initial values for each layer:

```
y[0] = state                  = [0.5, 0.3, 0.1, 0.2]   (fixed during inference)
y[1] = activation(W_1 · y[0]) = [0.42, ...]
y[2] = activation(W_2 · y[1]) = [0.31, ...]
y[3] = activation(W_3 · y[2]) = [0.28]                  (the output)
```

This is what a regular MLP would compute and stop. PC keeps going.

### Iteration 1

**Top-down predictions:** each layer above predicts what its lower neighbor should look like, using the *transposed* weights.

```
ŷ[0] = activation(W_1^T · y[1])    "what y[0] should be, if y[1] is correct"
ŷ[1] = activation(W_2^T · y[2])    "what y[1] should be, if y[2] is correct"
ŷ[2] = activation(W_3^T · y[3])    "what y[2] should be, if y[3] is correct"
```

**Prediction errors:**

```
e[0] = y[0] − ŷ[0] = 0.08    (input level — the state)
e[1] = y[1] − ŷ[1] = 0.05    (hidden 1)
e[2] = y[2] − ŷ[2] = 0.04    (hidden 2)
```

These errors say: *"the layers don't agree with each other yet."*

**State update:** each hidden layer moves itself slightly to reduce its error. The update blends the bottom-up signal (what the layer below sends up) with the top-down correction:

```
y[1] += alpha · (bottom_up_to_1 − y[1] + W_2^T · e[1])
y[2] += alpha · (bottom_up_to_2 − y[2] + W_3^T · e[2])
y[3] = activation(W_3 · y[2])    (output recomputed from updated hidden states)
```

After this iteration:
```
y[1] = 0.45, y[2] = 0.33, y[3] = 0.27
e[0] = 0.04, e[1] = 0.02, e[2] = 0.02
RMS error = 0.029   (still > tol = 0.01, keep iterating)
```

### Iteration 2

The errors shrink as the conversation continues:
```
y[1] = 0.46, y[2] = 0.34, y[3] = 0.27
RMS error = 0.011
```

### Iteration 3

```
y[1] = 0.46, y[2] = 0.34, y[3] = 0.27
RMS error = 0.005   (✓ < tol = 0.01, converged!)
```

The loop stops. The output `y_conv = 0.27` is what the agent will use to choose an action.

### What just happened

The system found a **mutually consistent interpretation** of the input: a configuration of hidden states where each layer's prediction of its neighbor matches the neighbor's actual activation. That is the equilibrium. The output emerges from this equilibrium, not from a single feedforward pass.

---

## 4. Why this isn't just "more compute"

A skeptic might say: "you ran the network 3 times instead of 1. Did you really gain anything?"

Yes, two things:

### 4.1 Robustness to noisy or ambiguous inputs

If the state vector has a noisy component, a feedforward pass propagates the noise straight to the output. The PC loop, instead, **filters** the noise: the equilibrium converges to the coherent interpretation closest to the noisy input, not to the noisy input itself. Higher layers refuse to fully trust an interpretation that conflicts with their priors.

Think of it like reading handwriting: a feedforward network reads each pixel literally. PC reads "this looks like the letter 'a' in someone's specific handwriting style — adjust my interpretation of the ambiguous strokes accordingly." The hidden state doesn't just pass forward; it **adjusts itself** based on what the higher layer thinks should be there.

### 4.2 An explicit signal of how confident the network is

After convergence, the final RMS error tells you how well the layers agreed. This is the **surprise score**:

- Low surprise (e.g. 0.005) → the input fits cleanly into the network's existing model. The interpretation converged easily.
- High surprise (e.g. 0.15) → the input is ambiguous, novel, or out-of-distribution. The layers struggled to agree.

This signal does not exist in feedforward networks. PC produces it for free as a byproduct of the inference process. We use it for:

- **Learning rate modulation:** boost LR when surprise is high (novel state — learn fast); reduce LR when surprise is low (familiar state — don't disturb existing knowledge).
- **Hysteresis triggers:** the FROZEN/PLASTIC state machine consumes surprise to decide when to gate weight updates.
- **Curriculum diagnostics:** a sudden spike in mean surprise signals a distribution shift.

---

## 5. Now the training side — backprop on the equilibrium

Once the loop converges, we have an output `y_conv` and we want to update the weights so that next time, the agent does better. How does the gradient work?

### 5.1 The naive answer (what we don't do)

You might think: "we ran the loop 3 times, so we backprop through all 3 iterations." That's called **backpropagation through time** (BPTT), and it has two problems:

1. **Cost:** if the loop took 5 iterations, BPTT requires 5 backward passes — the gradient computation is now 5× more expensive. Plus you have to keep all intermediate activations in memory.
2. **Stability:** gradients can vanish or explode as they cross the unrolled iterations, just like with recurrent networks. Training becomes unstable.

### 5.2 The actual answer (snapshot the equilibrium)

PC-RL-Core does something simpler: **treat the converged state as if it were a single feedforward output, and backprop normally from there.**

```
After convergence:                    For training:
  y[1] = 0.46                           Treat y[1], y[2], y[3] as static activations.
  y[2] = 0.34                           Backprop the output gradient through them
  y[3] = 0.27                           using regular chain rule, ONE backward pass.
```

This is one backward pass, not 5. Memory: same as a regular MLP. Stability: same as regular backprop.

The trick is that we discard the iterative history. The network "thought hard" during inference (3-5 iterations of conversation), but for the gradient, we pretend it was a single thought.

### 5.3 Why this is mathematically OK

The hand-waving explanation: at the equilibrium, the network is by definition stable — the hidden states aren't moving anymore (or moving very little). So the **last** iteration through the loop is essentially equivalent to what would have happened if the network had reached the same state on a single pass. Backpropagating through that last iteration is an approximation to the "true" gradient at the fixed point, and the approximation is good when the convergence is good.

The technical name for this is **implicit differentiation at a fixed point**. The exact gradient would require inverting a Jacobian (computationally horrifying), but the equilibrium-snapshot version is a low-cost approximation that works empirically — provided the loop actually converges, which is why the `tol` and `max_steps` settings matter.

### 5.4 The connection with surprise (closing the loop)

Putting this together:

- **Surprise low → equilibrium converged tightly → snapshot is a sharp approximation → gradient is trustworthy → use full learning rate.**
- **Surprise high → equilibrium barely converged → snapshot is loose → gradient is noisy → attenuate learning rate.**

The surprise modulation isn't just an exploration cue. It's also protecting the optimizer from updates derived from poorly-converged equilibria. When the network couldn't form a coherent interpretation, the gradient computed from that messy state is correspondingly less reliable, and the LR scaling rule "use small steps when surprise is high" is the safety net.

This dual role is one reason the architecture is so resilient to curriculum transitions: out-of-distribution inputs simultaneously trigger curiosity (high surprise → "I should learn this") AND caution (high surprise → "but trust the gradient less"). The system doesn't blindly chase novelty; it learns from it carefully.

---

## 6. The hybrid learning rule (`local_lambda`)

There's one more piece. PC's iterative loop produces, as a byproduct, **prediction errors at every layer**. These errors carry information about how each layer disagreed with its neighbors during the conversation. They are a **second source of learning signal**, completely separate from the policy-gradient backprop signal.

For each hidden layer, the actual weight update blends both signals:

```
delta = lambda · backprop_gradient + (1 - lambda) · pc_prediction_error
```

- `lambda = 1.0` — pure backprop. Standard policy gradient. Reward signal is the only teacher.
- `lambda = 0.0` — pure PC. The network learns from internal consistency only, ignoring reward. (Doesn't work for RL; the policy never learns to optimize the task.)
- `lambda = 0.99` — 99% backprop + 1% PC error as a regularizer.

That last setting is the empirical sweet spot. The 1% of PC error doesn't carry reward information — it carries **representational coherence** information. It nudges weights toward "the layers should agree with each other" alongside the main "the policy should maximize reward" signal. Empirically, this small dose helps the optimizer escape local minima that pure backprop can't cross. The intuition: pure backprop optimizes a noisy, high-dimensional objective and gets stuck in saddle points; the small PC error term provides a structured perturbation that points in a direction (representational coherence) which is correlated with — but not identical to — improved policy quality. That correlation is enough to escape.

The output layer always uses pure backprop because it has no upper layer to receive PC predictions from — only the policy gradient signal flows in.

---

## 7. Pseudocode of the full step

Putting it all together:

```rust
fn step(state, reward, done) -> action {
    // ─── INFERENCE: PC LOOP CONVERGES TO EQUILIBRIUM ─────────────
    let mut y = initial_feedforward(state);
    for k in 0..max_steps {
        let predictions = top_down_predictions(y);
        let errors = y - predictions;
        let bottom_up = bottom_up_pass(state, y);
        y = update_hidden_states(y, bottom_up, errors, alpha);
        if rms(errors) < tol { break }
    }
    let output = y[N];                  // the converged output
    let surprise = rms(errors);         // how well it converged

    // ─── ACTION: SAMPLE FROM POLICY ──────────────────────────────
    let action = sample_or_argmax(output);    // discrete
    // or: action = output + sigma * randn()    // continuous

    // ─── VALUE: CRITIC ESTIMATES V(s) ────────────────────────────
    let value = critic.forward([state, latent_concat]);
    let td_error = reward + gamma * next_value - value;

    // ─── GRADIENT: BACKPROP ON THE EQUILIBRIUM SNAPSHOT ──────────
    let delta = build_policy_gradient(action, output, td_error);
    // For each hidden layer, blend backprop with PC error:
    //   weights_grad = lambda * backprop_grad + (1-lambda) * pc_error
    actor.update_with_decay(delta, snapshot_y, surprise_to_scale(surprise));

    critic.update([state, latent_concat], target_value);

    return action;
}
```

Three phases: **think** (PC loop), **act** (sample), **learn** (backprop on the snapshot, scaled by surprise).

---

## 8. Why this matters for RL specifically

PC was originally a model of perception (Rao & Ballard 1999). The Free Energy Principle (Friston 2009) extended it to a general theory of cognition. Why is it useful for reinforcement learning?

1. **Deliberation as a primitive.** Most RL agents react in a single feedforward sweep. PC agents deliberate before acting. For tasks where the right action depends on subtle interpretation of the state (board games, robotic control, navigation), deliberation produces measurably better policies — the empirical results in `pc_actor_critic_paper.md` §2 quantify "+1 to +2 minimax depth levels over equivalent MLP."

2. **Latent concatenation.** Because PC produces explicit hidden state activations at every layer, those activations can be fed to the critic as part of its input. The critic sees not just the raw state but also the actor's internal interpretation. This gives the critic richer features for value estimation — the actor's "thoughts" become inputs to the value function.

3. **Surprise as a free intrinsic signal.** Without instrumenting anything, PC produces a per-step uncertainty estimate. This drives learning rate modulation, hysteresis transitions, and curriculum diagnostics, all from a signal that is mechanically inevitable in any PC architecture. Standard RL agents need to instrument this separately (e.g. via prediction networks, RND, or count-based exploration).

4. **The hybrid learning rule.** PC + backprop + the 1% blend produces an optimization dynamic that can escape local minima a pure REINFORCE / actor-critic can't. The empirical depth-9 ceiling on Tic-Tac-Toe (paper §2.5) demonstrates this concretely.

5. **A natural fit with continuous action spaces.** When the actor uses PC inference and the action space is continuous (v4.0.0), the converged output `y_conv` is a vector in continuous space — and the policy gradient operates on the *geometry* of that space, not on enumerated alternatives. PC's deliberation produces a coherent vector, and continuous policy gradient respects that coherence by updating all dimensions simultaneously. The combination is more than the sum of its parts.

---

## 9. Trade-offs in practice

| Property | PC actor | Standard MLP actor |
|---|---|---|
| Forward cost per step | k × 1 forward pass (k = 3–5) | 1 forward pass |
| Backward cost per step | 1 backward pass | 1 backward pass |
| Memory footprint | Same as MLP (snapshot at equilibrium) | Same |
| Stability of gradients | Same as MLP backprop | Standard |
| Robustness to noisy inputs | High (deliberation filters) | Low (passes noise through) |
| Built-in uncertainty estimate | Yes (surprise score) | No |
| Latent representations exposed | Yes (latent_concat for critic) | Possible but ad-hoc |
| Hyperparameter sensitivity | Adds `alpha`, `tol`, `max_steps`, `local_lambda` | Standard set |
| Convergence requirement | Critical — bad convergence breaks gradient | N/A |
| Activation function constraints | Bounded only (Tanh, Softsign) | Unbounded acceptable (ReLU, ELU) |

The right read is: **PC pays roughly 2–3× compute cost during inference for benefits that are structural — not just empirical.** The structural benefits (built-in uncertainty, deliberation, latent exposure to the critic) compound: you don't have to re-engineer them; they are inherent to the architecture.

The cost in compute is real and matters for very large models. The current `pc-rl-core` configurations (~1,900 parameters total) target environments where the deliberation gain dominates the compute cost. Whether PC scales to large language model regimes is an open empirical question (and beyond the scope of this codebase).

---

## 10. Where to look in the codebase

- `src/pc_actor.rs` — the PC inference loop and the actor's `infer()` method.
- `src/pc_actor_critic/mod.rs` — `learn_continuous_inner()` is where backprop on the equilibrium snapshot lives. The StepAction match builds the output gradient (REINFORCE for discrete, Gaussian for continuous), and `update_with_decay` propagates it.
- `src/pc_actor_critic/config.rs` — hyperparameters (`alpha`, `tol`, `max_steps`, `local_lambda`, `surprise_low/high`, etc.).
- `docs/pc_actor_critic_paper.md` — formal architecture spec, mathematical justification, full empirical results.
- `docs/experiment_analysis.md` — statistical validation across 35 seeds × 8 configurations.

---

## 11. Common questions

**Q: Is PC just a fancy way of running a feedforward network k times?**

No. The k iterations are not redundant — they are different. Iteration 1 is the standard feedforward initialization. Iteration 2 onward applies top-down predictions and adjusts hidden states based on prediction errors. Each iteration converges further toward an equilibrium. The output at the final iteration is qualitatively different from the output of iteration 1.

**Q: What if the loop doesn't converge?**

Then `surprise` is high, the gradient computed from that snapshot is noisy, and the LR scaling rule attenuates the update. The agent doesn't crash — it just learns less from that particular step. The architecture is designed to gracefully tolerate occasional non-convergence (it should be rare in distribution).

**Q: Why doesn't PC use BPTT?**

Two reasons: cost (BPTT would multiply the backward cost by k, the number of iterations) and stability (gradients propagated through many recurrent steps tend to vanish or explode). The equilibrium snapshot avoids both, at the cost of being an approximation rather than the exact gradient. The empirical results indicate the approximation is acceptable.

**Q: Why does `local_lambda = 0.99` work but `local_lambda = 0.5` doesn't?**

PC prediction errors and policy gradients optimize different objectives (representational coherence vs reward maximization). At very small lambda mixing (1% PC error), the PC term acts as a structured perturbation that helps the optimizer escape saddle points. At larger mixing, the two objectives compete and the policy fails to converge. The paper §2.5 has the empirical sweep.

**Q: Is PC biologically plausible?**

PC is one of the most prominent computational theories of cortical function (Rao & Ballard 1999; Friston Free Energy Principle). The brain does not appear to do single-pass feedforward computation in cortex; it appears to maintain ongoing predictive loops between hierarchical levels. This codebase is an engineering implementation of the same computational principle — not a biological simulation, but architecturally consistent with the theory. The hybrid `local_lambda` rule combines PC-style local plasticity with backpropagation, which is a deliberate engineering choice (pure local PC fails for RL — see paper §2.4); it is not claimed to be biologically realistic.

---

## 12. Summary

PC turns the actor into a **deliberator**: instead of computing the output in a single forward sweep, it runs an iterative top-down/bottom-up conversation between layers until they reach equilibrium. The output is the result of this deliberation. The training rule then takes a snapshot of the equilibrium and applies regular backprop to it — preserving the cost of standard MLP training while gaining the benefits of iterative deliberation. A 1% blend of PC prediction errors into the gradient acts as a regularizer that helps the optimizer escape local minima.

The key insight is that the iterative inference and the standard backprop are **decoupled**: deliberation happens during the forward pass, gradient computation operates on the converged snapshot. This decoupling is what makes the architecture both computationally tractable and empirically effective.

For the formal version with proofs, equations, and full experimental tables, read `pc_actor_critic_paper.md`. For an even higher-level overview of why this matters for `pc-rl-core` as a whole, read the project README.
