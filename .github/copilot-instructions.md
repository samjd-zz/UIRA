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
- Keep every Python source and test file below 600 total lines (maximum 599, including comments and blank lines). Check line counts before completing changes; split growing files into cohesive modules/classes rather than compressing code or removing useful documentation to meet the limit.

## PyTorch and Validation

- Follow PyTorch device and dtype propagation; create derived tensors from existing tensors when possible.
- Keep model parameters and all tensors on the selected device. The example should support Intel XPU, CUDA, and CPU fallback without hard-coding tensor placement.
- Validate input shape assumptions when a mismatch could otherwise produce a confusing broadcasting or indexing error.
- When changing model behavior, verify output shapes, finite loss values, and that a backward pass produces gradients on a small CPU example. Run the same smoke check on accelerator hardware when available.
- Add or update proper pytest unit tests for every new capability and bug fix. Cover normal behavior, invalid inputs, boundary cases, serialization, and device/dtype behavior where relevant. Write failing tests before implementation, run the targeted suite, and record results in the implementation plan before marking tasks complete.
- Reuse maintained, performant libraries rather than recreating established tools. Use NumPy for suitable CPU array/byte preprocessing and PyTorch for model/device tensor operations; avoid NumPy conversions inside accelerator forward paths.
- Consider scikit-learn for conventional ML baselines/metrics and maintained BM25 libraries for retrieval when those features are implemented. Do not add unused packages or claim a library is fastest without workload-specific evidence.
- Keep runtime dependencies in `requirements.txt` and test dependencies in `requirements-dev.txt`. Prefer functionality already provided by an existing dependency; add libraries only for a concrete implemented need and verify compatibility with supported Python/PyTorch versions.
- Keep accelerator-specific PyTorch distributions in separate requirements files so CPU users do not download unrelated GPU runtimes.
- Set up a local environment with `python3 -m venv .venv`, activate it, then run `python -m pip install -r requirements.txt`.
- Run the example with `python uira.py` from the repository root.

## Documentation

- Keep the README aligned with implemented behavior and include runnable setup instructions.
- Clearly distinguish implemented behavior from proposed or theoretical capabilities; do not claim the prototype guarantees grounding, truth verification, or hallucination elimination without supporting evidence.