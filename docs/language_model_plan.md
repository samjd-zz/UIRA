# UIRA Language-Model Implementation Plan

Date: 2026-10-22

Status: Phase 1 implemented and validated; Phases 2-6 remain pending.

## Implementation Progress

| Phase | Status | Evidence |
| --- | --- | --- |
| P1: Text encoding and compatibility | Complete | `text_encoding.py`, class-owned synthetic example, and 59 passing unit tests |
| P2: Causal context | Pending | No sequence model implemented yet |
| P3: Language training/checkpoints | Pending | Synthetic example only |
| P4: Generation | Pending | No generation loop |
| P5: Conversation training | Pending | Reserved role IDs only; no conversation pipeline |
| P6: Comparative evaluation | Pending | No language-quality results claimed |

### Phase 1 validation (2026-10-22)

- Tests were written first and initially failed collection because the new
  encoding module and application classes did not exist.
- `.venv/bin/python -m pytest tests/test_text_encoding.py tests/test_uira.py -q`:
  **59 passed, 2 skipped**. CUDA/XPU integration checks were skipped because
  those devices were unavailable; CPU forward/backward and float64 propagation
  passed. The editor test runner did not discover pytest tests, so the suite
  was executed explicitly in the repository virtual environment.
- `.venv/bin/python uira.py --device cpu`: finite losses and logits shape
  `[4, 8, 250]`; original CLI and `select_device` compatibility preserved.
- `git diff --check`: passed.
- Tokenizer persistence includes the root-count configuration; encoder buffers
  survive state-dict reload. All 263 token IDs have unique root pairs.
- NumPy vectorizes byte preprocessing; PyTorch performs root lookup on-device.
  No authenticated external API, training corpus, or benchmark is required.
- Next task: T004, followed by T005 and the P2 causality/padding acceptance gate.

## Goal and Scope

Turn the existing representation-learning prototype into a small, trainable,
causal text model, then add controlled generation and supervised conversation
training. The first deliverable is a reproducible research model, not an
OpenAI-scale assistant or a guarantee of factual accuracy.

This plan derives requirements from the requested four missing capabilities and
the current [implementation](../uira.py). There is no separate feature spec or
migration design artifact. Requirements, decisions, tasks, and traceability are
consolidated here rather than spread across additional planning files.

## Requirements

| ID | Requirement |
| --- | --- |
| REQ-001 | Reversible text tokenization and deterministic, collision-free token-to-root encoding with saved configuration |
| REQ-002 | Causal contextual sequence processing with positional information and padding handling |
| REQ-003 | Real next-token training, reproducible splits, and reloadable checkpoints |
| REQ-004 | Autoregressive generation with explicit limits, stopping rules, and validated decoding options |
| REQ-005 | Serialized conversations and supervised assistant-only instruction training |
| REQ-006 | Require object-oriented Python while preserving the root-pair prototype, device support, type safety, and offline tensor-only model computation |
| REQ-007 | Reproducible evaluation and honest capability reporting before any scaling or performance claim |

## Technical Context and Working Decisions

- Keep Python 3.9+ and PyTorch. Use Python UTF-8 encoding plus NumPy vectorized
  byte-array conversion; use the standard library for configuration, JSONL
  records, and the initial command-line tools. Use pytest for unit tests.
- Reuse maintained libraries for implemented needs. Consider scikit-learn for
  conventional evaluation baselines and maintained BM25 implementations if
  retrieval is added; neither is currently a dependency because those features
  are not implemented. Benchmark workload performance before claiming a library
  is the most performant option.
- Require object-oriented Python for all application components, not just neural
  layers. Use focused classes, typed public methods, constructor-injected
  collaborators, typed configuration dataclasses, and composition. Module-level
  entry points may only instantiate/delegate to an application class; small
  stateless helpers are allowed, but procedural application workflows are not.
- Compose focused `nn.Module` components through `UIRAFullArchitecture`. Preserve
  its existing root-pair input path and output keys; make contextual processing
  an explicit constructor option, disabled for the legacy synthetic example.
- Start with UTF-8 byte tokens, not a new tokenizer dependency. Reserve seven
  distinct IDs for PAD, BOS, EOS, SYSTEM, USER, ASSISTANT, and END_TURN, followed
  by 256 byte IDs: vocabulary size 263. Persist the exact ID allocation.
- Encode token ID `t` as `(t // R, t % R)`, with `R = 17` roots by default:
  `R**2 = 289 >= 263`. Store a registered lookup buffer for token-to-root pairs.
  This is an injective code, not a discovery of semantic primitives.
- The text configuration must override legacy defaults of 16 roots in the
  example and vocabulary size 250. Keep both root indices in range and keep
  logits restricted to the actual 263-token vocabulary.
- Start with embedding width 64, two causal Transformer layers, four attention
  heads, feed-forward width 256, dropout 0.1, and context length 128. These are
  initial CPU-oriented settings, not tuned performance claims.
- Use PyTorch's existing Transformer building blocks with an explicit causal
  mask, positional embeddings, and a padding mask. A masked encoder stack can
  implement decoder-only self-attention without adding cross-attention.
- Keep the pairwise composition for the first experiment. Its dense projection
  scales approximately as `D**3` parameters; width 512 means about 134 million
  projection weights alone. Do not scale width blindly. Factorization is a
  later experiment requiring equivalence/quality and resource measurements.
- Keep the isomorphic loss optional and disabled for ordinary text without a
  meaningful external metric. Measure consensus regularization separately;
  view agreement does not verify truth.
- These are proposed initial choices. Change them only with recorded rationale,
  artifact versioning, and updated acceptance criteria.

## Project Constraints and Applied Guidelines

Follow the [project instructions](../.github/copilot-instructions.md): registered
parameters/modules, configurable dimensions, device/dtype propagation, explicit
shape validation, and training/data/CLI code outside `forward`.
No framework migration guidelines apply to this additive PyTorch roadmap.
CPU is the required validation platform; run equivalent checks on CUDA/XPU when
available and report untested devices explicitly. Keep accelerator distributions
in their existing separate requirements files.

Phase 1 adds NumPy runtime and pytest development requirements, text encoding,
and a compatibility-preserving class refactor. Other proposed paths/interfaces
remain future work. Every subsequent phase requires unit tests, targeted
validation, and a progress entry before tasks are marked complete.

### Required Class Ownership

| Proposed file | Class owners and responsibilities |
| --- | --- |
| `text_encoding.py` | `TokenizerConfig` dataclass, `ByteTokenizer` for encoding/decoding, and `TokenToRootEncoder(nn.Module)` for registered root-pair lookup |
| `sequence_model.py` | `SequenceConfig` dataclass and `CausalSequenceProcessor(nn.Module)` for contextual tensor computation |
| `uira.py` | `UIRAFullArchitecture` composes model modules; `DeviceSelector` owns device policy; `SyntheticTrainingExample` owns the existing example workflow |
| `training_data.py` | `CorpusPreparer` owns provenance/splitting; `TextDataset` and `ConversationDataset` subclass `torch.utils.data.Dataset`; `ConversationSerializer` owns templates; `CausalBatchCollator` owns shifting/padding/label masks |
| `train.py` | `TrainingConfig` dataclass, `LanguageModelTrainer`, and `TrainingApplication` own optimization, validation, and CLI orchestration |
| `checkpoints.py` | `CheckpointManager` owns artifact validation, saving, loading, and resume state |
| `generation.py` | `GenerationConfig` dataclass, `TextGenerator`, and `GenerationApplication` own decoding policy and completion/conversation CLI behavior |
| `evaluate.py` | `LanguageModelEvaluator` and `EvaluationApplication` own metrics, ablations, and CLI behavior |

These classes are implementation responsibilities, not a requirement to build
an inheritance framework. Share the trainer between pretraining and supervised
fine-tuning through composition/configuration; do not duplicate training loops.
Keep existing public functions such as `select_device` as thin compatibility
delegates when moving their behavior into a class.

## Intended Pipeline

```text
Text or serialized conversation
  -> UTF-8 bytes + explicit control-token IDs
  -> deterministic root pairs
  -> existing root composition / views / reconciliation
  -> positional embeddings + causal contextual stack
  -> vocabulary logits
  -> shifted next-token training OR iterative generation
```

Put the contextual stack after reconciliation and before the task head.
This preserves the existing pre-context meaning of the view-agreement and
isomorphic losses. All supervised predictions use contextual states when the
new option is enabled.

## Implementation Phases and Tasks

### P1. Text Encoding and Compatibility Foundation

Requirements: REQ-001, REQ-006. Prerequisites: none.

- [x] T001 [Plan:P1] Add `text_encoding.py` with `ByteTokenizer.encode/decode`, explicit special-token handling, and a versioned JSON configuration. Ordinary input strings must never be interpreted as role/control tokens.
- [x] T002 [Plan:P1] Add `TokenToRootEncoder(nn.Module)` in `text_encoding.py` with a registered integer lookup buffer, vocabulary/capacity checks, and `[B, T] -> [B, T, 2]` conversion on the selected device.
- [x] T003 [Plan:P1] Update `uira.py` to validate root-pair shape, integer indices, index range, and supported combination size. Explicitly reject non-pair composition until it is implemented rather than silently accepting an incompatible option. Move the existing device policy and synthetic workflow into `DeviceSelector` and `SyntheticTrainingExample`, preserving `select_device` as a delegate and the current CLI behavior.

Acceptance: empty strings, ASCII, multilingual text, emoji, and literal role-marker
strings round-trip exactly through byte encoding and strict UTF-8 decoding.
All 263 IDs map to unique valid root pairs, saved/reloaded configurations agree,
and invalid IDs/configurations produce explicit errors. Control IDs are handled
explicitly, not treated as bytes.

### P2. Causal Contextual Processing

Requirements: REQ-002, REQ-006. Prerequisites: P1.

- [ ] T004 [Plan:P2] Add `CausalSequenceProcessor(nn.Module)` in `sequence_model.py` with configurable layers, heads, width, dropout, maximum length, learned positions, causal masking, and right-padding support.
- [ ] T005 [Plan:P2] Wire the optional processor and attention mask into `UIRAFullArchitecture` in `uira.py`. Keep the legacy path unchanged by default and preserve existing output keys. Exclude padding from sequence pooling and view-agreement loss on the new path.

Acceptance: logits are `[B, T, V]`; finite CPU forward/backward results reach
roots, projections, contextual layers, and head. In evaluation mode, changing
future tokens must leave earlier logits unchanged within a stated tolerance
(`atol=1e-6`, `rtol=1e-5` for the initial CPU float32 check). Changing a preceding
token must affect a later prediction on a controlled context-dependent task.
Right-padding must not change valid-position outputs or losses. Reject empty,
all-padding, overlength, or incompatible-shape inputs with clear errors.

### P3. Next-Token Training and Checkpoints

Requirements: REQ-003, REQ-006, REQ-007. Prerequisites: P1, P2.

- [ ] T006 [Plan:P3] Add `CorpusPreparer`, `TextDataset`, and `CausalBatchCollator` in `training_data.py` to read licensed local documents, record source/license/content hashes, split and deduplicate by document before windowing, insert BOS/EOS, and produce right-padded shifted input/target batches.
- [ ] T007 [Plan:P3] Add `TrainingConfig`, `LanguageModelTrainer`, and `TrainingApplication` in `train.py` with seeded initialization/data order, AdamW, configurable batch size/steps, gradient clipping, token-weighted validation loss, and separate logging of task/metric/consensus losses. Refuse batches with zero supervised targets.
- [ ] T008 [Plan:P3] Add `CheckpointManager` in `checkpoints.py` to save model/optimizer state, architecture and tokenizer/root configurations, progress, RNG states, corpus/split fingerprints, and format version. Reject incompatible artifacts; load only trusted checkpoints.

For a token stream `s`, use inputs `s[:-1]` and targets `s[1:]`; shift exactly
once in data preparation, not again in the model. Use `-100` for ignored target
positions. Keep token ID 0 valid as PAD in inputs without making it a target.
Construct non-overlapping supervised windows with enough predecessor context;
do not introduce fake BOS tokens at every arbitrary window boundary.

Acceptance: overfit a fixed tiny corpus with dropout disabled to mean task
cross-entropy below 0.1 nats per supervised token within a recorded step budget.
This is a plumbing check, not a language-quality claim. Reloaded evaluation
logits must match within the P2 tolerance, and resuming must restore optimizer
and progress state. Report held-out task loss and byte-token perplexity
separately from auxiliary losses; report bits per byte on byte targets only.
Do not compare perplexities from different tokenizers as if they were equivalent.

### P4. Autoregressive Generation

Requirements: REQ-004, REQ-006. Prerequisites: P1, P2, P3.

- [ ] T009 [Plan:P4] Add `GenerationConfig`, `TextGenerator`, and `GenerationApplication` in `generation.py` with checkpoint-backed prompt encoding, last-valid-position logits, token selection, appending, root re-encoding, termination, and a thin CLI entry point.
- [ ] T010 [Plan:P4] Implement greedy decoding by default and optional positive-temperature/top-k sampling in `TextGenerator` with a local seed. Validate options, run under `torch.inference_mode()`, and restore the caller's previous training/evaluation mode afterward.

Start with single-prompt inference and recompute the full prefix; defer KV
caching and batched generation. Reject prompts that cannot fit the requested
continuation in the context window rather than silently dropping context.
Define `max_new_tokens=0` as an empty continuation; reject negative limits,
non-finite/nonpositive sampling temperatures, and invalid top-k values.
Mask PAD/BOS and role-control IDs during ordinary completion; allow EOS.
Stop on EOS or the hard new-token limit and do not expose EOS as visible text.

Accumulate generated byte IDs before UTF-8 decoding: individual bytes are not
necessarily characters. Exact encode/decode uses strict UTF-8, while generation
must explicitly report invalid/incomplete UTF-8 if replacement display is used.
Never silently replace undecodable bytes and call the output an exact round trip.

Acceptance: deterministic greedy output for a fixed checkpoint, repeatable
seeded sampling on the same supported environment, immediate EOS and exact
token-limit behavior, explicit invalid-option errors, and no gradient creation.
Use a controlled model output to exercise stopping independently of model quality.

### P5. Instruction and Conversation Training

Requirements: REQ-005, REQ-006. Prerequisites: P3, P4.

- [ ] T011 [Plan:P5] Add `ConversationDataset` and `ConversationSerializer` in `training_data.py` with a validated JSONL message schema and reserved role/END_TURN tokens. Accept optional SYSTEM followed by alternating USER/ASSISTANT messages; reject unsupported or malformed records explicitly.
- [ ] T012 [Plan:P5] Extend `CausalBatchCollator` and `LanguageModelTrainer` with instruction training from a language-model checkpoint, masking system/user/padding labels and supervising assistant content and END_TURN with the same one-time next-token shift.
- [ ] T013 [Plan:P5] Extend `TextGenerator` and `GenerationApplication` with conversation mode composed with `ConversationSerializer`, an ASSISTANT prefix, END_TURN/EOS stopping, and a context-budget error rather than silent removal of system instructions or earlier turns.

Start with supervised fine-tuning only. Use licensed human-authored or otherwise
permitted examples with recorded provenance; keep evaluation prompts separate.
Split by conversation and group near-duplicates/template families to reduce
leakage. For records exceeding the context limit, report and exclude them with
counts initially; never truncate labels silently or train on partial turns.
At inference, allow END_TURN but mask generated role-start tokens; user text
containing role-marker spellings must remain ordinary byte tokens.

Acceptance: inspect target masks against serialized multi-turn examples; a
controlled loss calculation must show no task-loss contribution from user,
system, role-prefix, or padding targets. Changing user tokens can still affect
assistant predictions through context. Evaluate on a frozen set of at least
50 held-out instructions spanning format constraints, simple transformations,
and multi-turn reference tasks; report per-category scores and failure examples.
Any usefulness threshold must be chosen before the evaluation, not after seeing
the scores. Passing serialization checks does not establish reliable instruction
following, safety, factuality, or production readiness.

### P6. Comparative Evaluation and Documentation

Requirements: REQ-006, REQ-007. Prerequisites: P3 for language evaluation;
P4 and P5 for completion/conversation evaluation.

- [ ] T014 [Plan:P6] Add `LanguageModelEvaluator` and `EvaluationApplication` in `evaluate.py` for frozen split evaluation and run configurations: standard token embeddings with the same contextual stack, UIRA roots without agreement loss, UIRA with agreement loss, and metric-loss ablation only when a justified metric exists.
- [ ] T015 [Plan:P6] Update `README.md` with implemented commands, artifact formats, hardware/precision, seeds, parameter counts, peak memory, training/inference timings, held-out metrics, and representative successes/failures. Keep proposed capabilities marked as such until their acceptance gates pass.

Use at least three seeds for comparative claims and report mean/spread.
Account for parameter and compute differences, or document why a comparison is
not matched. Measure view diversity alongside prediction quality. A decrease
in agreement loss alone is not a success criterion.

## Proposed File Layout

`text_encoding.py`, `uira.py`, and the Phase 1 unit tests now exist. The remaining
language-model modules below are proposed:

```text
uira.py                 existing core + opt-in contextual composition
text_encoding.py        tokenizer and root-ID mapping
sequence_model.py       causal contextual processor
training_data.py        document/conversation loading and supervision
train.py                language training and supervised fine-tuning
checkpoints.py          versioned save/load and resume
generation.py           completion/conversation decoding and CLI
evaluate.py             held-out evaluation and ablation entry point
docs/language_model_plan.md
```

Keep downloaded datasets and checkpoints out of source control. Keep data
fetching, credentials, and optional Katzilla integration outside model modules.
No retrieval service is required by any phase above.

## Execution Order and Completion Gates

`P1 -> P2 -> P3 -> P4 -> P5`; begin P6 language evaluation after P3, then
extend it after P4/P5. Task numbering is execution order within each phase.
P1-P3 establish a trainable causal text model; P4 enables completion; P5 adds
the conversation training/inference pipeline. Useful instruction following
remains an empirical outcome, not an automatic consequence of implementing P5.

First complete a small CPU run within a recorded parameter/memory budget before
committing to longer training or accelerator-specific tuning. The tiny-corpus
gate verifies mechanics; only held-out results assess generalization.
Do not infer factual grounding from next-token loss or conversational fluency.
Keep experimental results in versioned evaluation records with configs and
checkpoint identifiers; no benchmark result is asserted by this plan.

Every phase review must also confirm the class ownership above, explicit
dependencies, registered model state, and thin entry points. Copilot instructions
are an authoring/review requirement, not an automated architectural linter.

## Requirement Mapping

| Requirement | Plan items | Tasks | Planned completion evidence |
| --- | --- | --- | --- |
| REQ-001 | P1 | T001-T003 | Exact text round trips, unique root pairs, persisted configuration |
| REQ-002 | P2 | T004-T005 | Causality/context/padding checks and gradient evidence |
| REQ-003 | P3 | T006-T008 | Shifted real-data batches, overfit gate, checkpoint reload/resume |
| REQ-004 | P4 | T009-T010 | Bounded deterministic/seeded generation and stopping checks |
| REQ-005 | P5 | T011-T013 | Stable chat template, assistant-only labels, held-out instruction report |
| REQ-006 | P1-P6 | T001-T015 | Class-owned workflows, thin entry points, legacy compatibility, explicit validation, CPU and available-device evidence |
| REQ-007 | P3, P6 | T006-T008, T014-T015 | Frozen splits, multi-seed ablations, resource accounting, qualified README claims |

## Deferred Work

Subword tokenizer training, learned semantic root assignment, explicit hypergraph
grammars, factorized tensor products, KV caches, distributed training,
preference optimization, tool execution, retrieval, and deployment hardening
are outside this first roadmap. Revisit them only after the basic pipeline
works and measurements identify a need.
