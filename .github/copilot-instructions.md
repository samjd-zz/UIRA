# Copilot Instructions

## Project Overview

UIRA is a research prototype of a PyTorch representation-learning architecture. Its current implementation is in `uira.py`; the README describes the theoretical formulation. Treat theoretical goals as hypotheses, not demonstrated guarantees.

## Design and OOP

- Prefer object-oriented design and composition. Reusable model components should be focused `torch.nn.Module` subclasses, assembled by `UIRAFullArchitecture`.
- Register trainable parameters and child modules through PyTorch's standard APIs (`nn.Parameter`, module attributes, `nn.ModuleList`, and `nn.Sequential`). Do not keep trainable tensors in ordinary Python containers.
- Keep `forward` methods focused on tensor computation. Put setup and configuration in constructors; keep training loops, data loading, and command-line behavior outside model components.
- Preserve configurable constructor options and make their behavior match their names. Avoid silently hard-coding dimensions or assumptions that conflict with a public parameter.
- Use descriptive names and type annotations. Document non-obvious tensor dimensions at the operation where they matter.

## PyTorch and Validation

- Follow PyTorch device and dtype propagation; create derived tensors from existing tensors when possible.
- Keep model parameters and all tensors on the selected device. The example should support Intel XPU, CUDA, and CPU fallback without hard-coding tensor placement.
- Validate input shape assumptions when a mismatch could otherwise produce a confusing broadcasting or indexing error.
- When changing model behavior, verify output shapes, finite loss values, and that a backward pass produces gradients on a small CPU example. Run the same smoke check on accelerator hardware when available.
- Keep dependencies in `requirements.txt`. Do not add a dependency for functionality already provided by Python or PyTorch.
- Keep accelerator-specific PyTorch distributions in separate requirements files so CPU users do not download unrelated GPU runtimes.
- Set up a local environment with `python3 -m venv .venv`, activate it, then run `python -m pip install -r requirements.txt`.
- Run the example with `python uira.py` from the repository root.

## Documentation

- Keep the README aligned with implemented behavior and include runnable setup instructions.
- Clearly distinguish implemented behavior from proposed or theoretical capabilities; do not claim the prototype guarantees grounding, truth verification, or hallucination elimination without supporting evidence.