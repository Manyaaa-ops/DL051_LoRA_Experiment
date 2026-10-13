# DL-051 Viva Preparation — 25 Questions with Answers

Paper: *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., ICLR 2022).
This project: from-scratch PyTorch LoRA on DistilBERT + GLUE SST-2, full FT vs
LoRA r ∈ {2, 4, 8}, rank ablation.

Numbers below are either **paper-reported** (marked 📄) or **measured in this
project** (marked 📊, source `results/results.csv`).

---

## Core concepts

**Q1. What problem does LoRA solve?**
Full fine-tuning updates every pretrained parameter, so each downstream task
needs its own full model copy to store/serve, and training needs gradients +
optimizer states for all of them (📄 GPT-3 175B: ~1.2 TB training memory,
350 GB checkpoint). LoRA keeps the backbone frozen and trains a low-rank
update instead, cutting trainable parameters, optimizer state and per-task
storage (📄 up to ~10,000× smaller checkpoints).

**Q2. Derive W′ = W₀ + (α/r)BA.**
Start from the pretrained layer h = W₀x. Instead of learning a full ΔW ∈ ℝ^{d×k},
parameterise ΔW = (α/r)BA with B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r ≪ min(d,k). Then
h = W₀x + (α/r)BAx. The product BA is a sum of r rank-1 matrices, so the update
has rank ≤ r. α is a constant wrt r (paper) and just rescales the update; our
project fixes α = 2r so α/r = 2 for every rank, isolating rank as the variable.

**Q3. Why is BA low rank, and why is that a reasonable assumption?**
rank(BA) ≤ min(rank(B), rank(A)) ≤ r. The paper argues the change caused by
fine-tuning has low *intrinsic* rank — the task-specific information is far
lower-dimensional than d·k. Empirically a rank-4 update already matches full
fine-tuning in the paper's GLUE and GPT-3 experiments (📄 Table 6: Wq,Wv WikiSQL
73.4/73.3/73.7/73.8/73.5 for r = 1/2/4/8/64).

**Q4. Why are the pretrained weights frozen?**
Three reasons: (1) it preserves the pretrained knowledge the model already has;
(2) frozen parameters need no gradients and no Adam moment/velocity buffers —
the dominant training-memory saving; (3) the shared backbone can serve many
tasks while only tiny adapters are stored per task. The backbone is still
forwarded (activations) and stored, so memory does not shrink by the trainable
factor alone.

**Q5. Why is B initialised to zero?**
So that BA = 0 at step 0 and the adapted layer exactly reproduces the
pretrained layer: training starts at the pretrained function and learns a
*change* to it, with no initial disturbance of pretrained behaviour. A side-effect
(tested in our unit tests): ∂L/∂A = (α/r)(GB)ᵀxᵀ = 0 at the first step while
B = 0, while ∂L/∂B is generally non-zero — both factors train once B moves.

**Q6. Why is A random and B zero (not the other way round)?**
Symmetric zero-init would give zero gradients for *both* factors (both
Jacobians vanish), so nothing would ever learn. Only one factor may start at
zero. B is chosen so that the *output* of the adapter starts at zero.

**Q7. What happens when rank increases?**
Adapter parameters grow linearly: r(d+k) per adapted layer (📊 our injection:
6 layers × 2 projections × r × (768+768) = 18,432r → 36,864 / 73,728 / 147,456
for r = 2/4/8, verified by counting the live model). Capacity grows, but the
paper and our runs show saturating accuracy returns for Wq+Wv adaptation.

**Q8. Why adapt the query and value projections specifically?**
The paper's choice (📄 "in most experiments, we attach LoRA to Wq and Wv"),
motivated by its ablation (📄 Table 5, 18M budget): Wq alone 70.4 vs Wv alone
73.0 vs Wq+Wv 73.7 WikiSQL — adapting both beats putting higher rank into one
weight, matching our injection into DistilBERT's q_lin and v_lin.

**Q9. How does LoRA affect gradient and optimizer-state memory?**
Gradients exist only for trainable parameters; AdamW keeps two fp32 buffers per
trainable parameter. Full FT: 2 × 67M × 4 B ≈ 536 MB of optimizer state just
for DistilBERT. LoRA r=4: 2 × 665,858 × 4 B ≈ 5.3 MB total, of which the
adapters alone are 2 × 73,728 × 4 B ≈ 0.6 MB (the rest is the trainable
head). Activations and the frozen weights remain, so peak memory falls much
less than 100× (measured ~4×: 1027 vs 265 MiB — we report the measured peak
honestly).

**Q10. Full fine-tuning vs parameter-efficient fine-tuning (PEFT)?**
Full FT: all parameters trainable, one full model copy per task. PEFT (LoRA is
one family alongside adapters, prompt tuning, prefix tuning): freeze the
backbone and train a small task-specific subset. PEFT reduces trainable params,
optimizer memory, and per-task storage; it can slightly trade off accuracy and
adds an adaptation/merge step.

**Q11. Why can LoRA accuracy differ from full fine-tuning?**
LoRA constrains the update to a rank-r subspace, so it cannot represent
arbitrary changes; with very large intrinsic-rank shifts (new language, new
modality) it can lag. In practice at moderate scale it matches or exceeds full
FT (📄 RoBERTa-large SST-2: 96.2 ± 0.5 vs 96.4; GPT-3 MNLI-m 91.7 vs 89.5).
Our runs: within ~1 point (📊).

**Q12. What does the ablation establish?**
That rank is a real capacity knob: parameters scale exactly linearly in r while
accuracy saturates early for Wq+Wv — so one can pick the smallest rank that
hits target accuracy. What it does NOT establish (single seed, small subset):
precise rank-vs-accuracy differences within noise.

---

## Experimental design and correctness

**Q13. Why DistilBERT + SST-2 when the paper uses RoBERTa/GPT-3?**
Compute: the assignment targets a Colab T4 in a few GPU-hours; DistilBERT (66M)
lets us run baseline + three ranks + tests locally and on CPU/MPS. SST-2 is the
sentiment task in the paper's own GLUE table, so the *task* matches even though
the model does not. We state explicitly that this adapts the paper's principle,
not a specific paper row.

**Q14. What exactly is the protocol, and what did we change vs the paper?**
Change vs paper: model (DistilBERT vs RoBERTa-large/GPT-3), data (seeded subset
of \SubsetSize{} of 67,349 vs full data), epochs (2), seed (single), device
(Apple GPU locally / Colab T4 via notebook), α = 2r rule. Held fixed across
conditions: checkpoint, subset indices (persisted to JSON), batch 16, seq 128,
evaluation (validation accuracy, best epoch), optimizer family.

**Q15. How do you avoid data leakage / unfair comparison?**
Same subset indices for all runs (stored JSON, asserted at load), same
evaluation code path, fresh pretrained checkpoint per run, no test-set usage.
Honest caveat: validation is used both for model selection and reporting
(GLUE test labels withheld) — reported as a limitation.

**Q16. How do you know the backbone is really frozen?**
(a) requires_grad flags inspected and unit-tested; (b) a probe weight
(layer-0 q_lin) is cloned before training and compared bit-exactly after; every
run records frozen_backbone_verified = True in results.csv; (c) optimizer only
receives trainable params.

**Q17. How do you measure peak GPU memory fairly?**
torch.cuda.reset_peak_memory_stats() before training; read
max_memory_allocated() after — allocated *tensor* memory, not total process
memory. Same GPU, batch size, precision, seq length across runs. On Apple MPS
(local machine) there is no peak-stats API in our torch build, so we sample
current_allocated_memory() per batch and take the max — documented as sampled.

**Q18. Why is the LoRA learning rate 1e-3 while full FT uses 2e-5?**
Adapters (and their random-init path) train from scratch, like newly
initialised layers; pretrained weights need tiny updates to avoid destroying
pretrained features. This is standard practice (and the assignment's documented
starting rule); we flag it as "best practice per method", not a controlled sweep.

**Q19. Repducibility — how could someone reproduce this?**
`pytest -q tests/test_lora.py`, `python scripts/smoke_test.py`, then
`bash scripts/run_all.sh`; every run writes config.json + CSV rows; seeds are
fixed; subset indices are persisted; the notebook reruns everything on Colab.

---

## Extensions and critique

**Q20. What are the limitations of this experiment?**
Single seed; subset of the data; 2 epochs; validation-as-test; local device is
Apple GPU not T4; no secondary ablation of α or placement; DistilBERT is small
enough that memory savings are dominated by the frozen backbone + activations.

**Q21. How would you extend to GPT-2?**
Replace the classifier with a causal LM head; inject into c_attn (Conv1D) — our
LoRALinear expects nn.Linear, so either wrap with a Linear-compatible shim
(c_attn weight is (d, 3d): transpose into a Linear) or generalise the wrapper;
evaluate on E2E NLG (📄 paper Table 3: GPT-2 medium LoRA BLEU 70.4) or WikiSQL.

**Q22. How would you extend to AG News?**
Same sequence-classification path: 4 labels, load "ag_news" from the Hub,
num_labels=4; everything else (injection, protocol, metrics) is unchanged —
a good test that the pipeline generalises beyond binary SST-2.

**Q23. What did not work / what failed during the project?**
Early bugs: calling nn.Parameter like a function in the forward pass (fixed with
F.linear); the freezing probe comparing tensors across devices (fixed by cloning
to CPU); a unit test initially asserted a non-zero gradient for A at step 0,
which is mathematically impossible with B=0 — the test was corrected, not the
math. All reported failures were fixed and re-run; nothing was concealed.

**Q24. Could you use PEFT? Why not?**
Technically yes, but the assignment requires implementing the paper's core
contribution ourselves. PEFT (and the official LoRA repo) may be consulted for
reference — we did not consult any third-party LoRA implementation; a test
asserts no peft import exists in src/.

**Q25. If accuracy at r=2 equals r=8, is the ablation pointless?**
No — the point of a rank ablation is exactly to find the smallest sufficient
rank: it quantifies the parameter/accuracy trade-off (r=2 costs half of r=4 and
a quarter of r=8). Matching accuracy at smaller r *is* the finding, consistent
with the paper's observation that small ranks suffice for Wq+Wv.
