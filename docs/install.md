# Install

`narratoflow` requires Python 3.10+.

## With `pip`

```bash
pip install narratoflow
```

With optional extras:

```bash
pip install "narratoflow[dev]"        # pytest, ruff, mypy
pip install "narratoflow[docs]"       # mkdocs + material
pip install "narratoflow[benchmark]"  # pretty CLI output for benchmark
pip install "narratoflow[nlp]"        # spaCy (planned NLP integrations)
```

## With `uv`

[`uv`](https://docs.astral.sh/uv/) installs and resolves packages much faster than `pip`. `narratoflow` is a fully PEP 621 / hatchling package, so `uv` works out of the box.

### Install into a project

```bash
uv add narratoflow
# or, into your current env:
uv pip install narratoflow
```

### Run the CLI without installing globally

```bash
uvx --from narratoflow narratoflow --version
uvx --from narratoflow narratoflow compress doc.txt --profile rag-en
```

## From source

```bash
git clone https://github.com/Mrrobi/narratoflow.git
cd narratoflow

pip install -e ".[dev]"
# or with uv:
uv pip install -e ".[dev]"

pytest -q
```

## Build distributions

The package builds as a pure-Python wheel.

```bash
python -m build       # produces dist/*.whl + dist/*.tar.gz
# or:
uv build              # same outputs, faster
```

Verify the wheel before publishing:

```bash
twine check --strict dist/*
```

## Provider SDKs

`narratoflow` always pulls in the `anthropic`, `openai`, and `httpx` SDKs as required dependencies. They are small and installed regardless of which providers you actually use; the imports are guarded so missing credentials do not crash the library at import time.

For the [Ollama](https://ollama.com) provider you also need a running Ollama daemon (default: `http://localhost:11434`). No extra Python dep is required.

## Type checking

`narratoflow` ships a `py.typed` marker (PEP 561) and is fully annotated. `mypy`, `pyright`, and IDEs read the inline annotations directly:

```bash
pip install mypy
mypy your_app.py     # picks up narrato's types automatically
```
