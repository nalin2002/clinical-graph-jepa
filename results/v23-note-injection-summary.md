# Fawkes v23 note-injection ablation results

## Protocol

- Dataset: 4,000 admission graphs from
  `wmatbooth/fawkes-training-graph-embedded-260615`
- Split: 3,200 train / 400 validation / 400 test, `DATA_SPLIT_SEED=42`
- Training seeds: 42--51
- Evaluation: 8,283 filtered leave-one-out edge queries per run
- Shared configuration: 60 JEPA epochs, 40 readout epochs, BFS patch masking,
  eight patches, 0.4 node mask, frozen encoder readout and MLP decoder
- Hardware: one A10G per run for every arm
- Source: `1a698a9839860954786a42b7b47a4adfa014da26`
- Note memory: 4,000 x 64 x 768 FP16 Clinical ModernBERT span embeddings;
  maximum note length 1,081 tokens, no truncation

Values are mean ± sample standard deviation over ten paired seeds.

## Leave-one-out results

| Injection | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|
| Global mean | 0.4680 ± 0.0213 | 0.2910 ± 0.0263 | 0.5350 ± 0.0233 | 0.9109 ± 0.0077 |
| Uniform spans | **0.4778 ± 0.0204** | **0.3016 ± 0.0274** | **0.5481 ± 0.0194** | **0.9153 ± 0.0052** |
| Entity attention | 0.4582 ± 0.0262 | 0.2802 ± 0.0300 | 0.5224 ± 0.0328 | 0.9108 ± 0.0113 |

## Paired differences

| Contrast | MRR Δ | 95% CI | Winning seeds |
|---|---:|---:|---:|
| Uniform − mean | +0.0098 ± 0.0316 | [−0.0128, +0.0324] | 6/10 |
| Attention − mean | −0.0098 ± 0.0343 | [−0.0343, +0.0147] | 5/10 |
| Attention − uniform | **−0.0196 ± 0.0211** | **[−0.0348, −0.0045]** | 3/10 |

The attention-versus-mean interval includes zero, so this experiment does not
demonstrate an attention improvement. Uniform span injection is numerically
better than the global mean, but its primary-endpoint interval also includes
zero. Attention is significantly worse than the parameterized uniform control
under the unadjusted paired 95% interval.

## Batch-mask results

| Injection | AUC | AP | MRR |
|---|---:|---:|---:|
| Global mean | 0.8086 ± 0.0119 | 0.7279 ± 0.0226 | 0.3721 ± 0.0041 |
| Uniform spans | **0.8110 ± 0.0089** | **0.7335 ± 0.0183** | **0.3745 ± 0.0024** |
| Entity attention | 0.8095 ± 0.0134 | 0.7308 ± 0.0255 | 0.3733 ± 0.0045 |

Uniform batch-mask MRR improved by `+0.00239`, with an unadjusted 95% CI of
`[+0.00002, +0.00477]`. This is a secondary endpoint and does not survive a
family-wise interpretation of all reported metrics.

## Inferred-relation LOO MRR

| Injection | MANAGED_FOR | INDICATES | COMPLICATED_BY | CONFIRMS |
|---|---:|---:|---:|---:|
| Global mean | 0.4122 ± 0.0222 | 0.6357 ± 0.0306 | 0.5141 ± 0.0295 | 0.5132 ± 0.0414 |
| Uniform spans | **0.4238 ± 0.0146** | **0.6430 ± 0.0504** | **0.5200 ± 0.0276** | **0.5168 ± 0.0660** |
| Entity attention | 0.4046 ± 0.0285 | 0.6218 ± 0.0325 | 0.5001 ± 0.0329 | 0.4958 ± 0.0520 |

No individual relation had a paired mean-versus-attention or
mean-versus-uniform 95% interval excluding zero.

## Interpretation

The token-count-weighted uniform arm reconstructs the original v22 mean at a
minimum cosine of `0.999953` before its learned gated fusion. Its small numerical
gain therefore points more toward the separate gated injection path than toward
retaining local note content.

The attention module is learned only during JEPA because the encoder remains
frozen during the supervised edge readout. The negative result is consistent
with the unsupervised masked-node objective not teaching the entity query which
note spans matter for relation ranking. A follow-up should keep the GNN frozen
but train only the note-attention injector during readout, or use a semantic
entity-text query instead of the hashed entity bucket. Those are new experiments
and should not be folded into this v23 result.

Machine-readable statistics, including every paired per-seed difference and
confidence interval, are in
[`v23-note-injection-summary.json`](v23-note-injection-summary.json).
