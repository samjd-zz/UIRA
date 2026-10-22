# Copilot Instructions

## Project Overview

UIRA is a research prototype of a PyTorch representation-learning architecture. Its current implementation is in `uira.py`; the README describes the theoretical formulation. Treat theoretical goals as hypotheses, not demonstrated guarantees.

## Design and OOP

- Require object-oriented Python for all new or substantially changed application code, including model components, tokenization, datasets, training, checkpointing, generation, evaluation, and CLI orchestration.
- Give each responsibility a focused class with typed public methods and explicit constructor dependencies. Prefer composition over inheritance; use inheritance where required by PyTorch interfaces. Do not add empty wrappers, static-method-only utility classes, or unnecessary class hierarchies.
- Reusable model components must be `torch.nn.Module` subclasses, assembled by `UIRAFullArchitecture`. Dataset implementations should use PyTorch dataset classes; configuration and structured records should use typed dataclasses where appropriate.
- Put workflows in trainer, generator, evaluator, and application classes rather than procedural scripts or module-level training loops. A module-level `main()` and `if __name__ == "__main__":` block may only construct/delegate to an application class. Small stateless helpers and test functions are allowed; they must not replace class-owned application workflows.
- Preserve existing public APIs when refactoring procedural code into classes, using thin compatibility wrappers where necessary. Do not perform unrelated rewrites merely to satisfy this rule.
- For implementation reviews, verify that application responsibilities have class owners, dependencies are explicit, trainable state is registered, and entry points contain no business logic. Follow the class ownership and phase gates in [the language-model plan](../docs/language_model_plan.md).
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