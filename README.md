# doc-agent-ai

Auto-generates and keeps API documentation in sync with Python source code,
powered by [Claude](https://www.anthropic.com) (claude-sonnet-4-6).

## Features

- **AST-based extraction** — parses Python files to find every public class,
  method, and function together with their signatures and docstrings.
- **AI-generated docs** — sends the structured API info to Claude, which writes
  complete Markdown or RST documentation with usage examples.
- **Agentic tool use** — the agent can call `read_file`, `list_files`, and
  `write_file` to gather extra context and persist results.
- **Drift detection** — CI-friendly `check` command exits with code 1 when
  documentation is out of sync with source code.

## Installation

```bash
pip install doc-agent-ai
# or, from source:
pip install -e .
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## CLI Usage

### Generate documentation

```bash
doc-agent generate src/mypackage --output docs/api
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | *(required)* | Output directory for doc files |
| `--format`, `-f` | `markdown` | `markdown` or `rst` |
| `--model` | `claude-sonnet-4-6` | Claude model to use |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |

### Check for drift (CI mode)

```bash
doc-agent check src/mypackage --output docs/api
# exits 0 if in sync, 1 if any drift detected
```

Same flags as `generate`. Integrate with GitHub Actions:

```yaml
- name: Check API docs are up to date
  run: doc-agent check src/ --output docs/api
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Python API

```python
from doc_agent import DocAgent, DocExtractor

# Generate docs for a whole directory
agent = DocAgent()  # reads ANTHROPIC_API_KEY from env
written = agent.generate("src/mypackage", "docs/api", fmt="markdown")
print(f"Wrote {len(written)} files")

# Check for drift
drifted = agent.check("src/mypackage", "docs/api")
if drifted:
    for path, reason in drifted:
        print(f"{path}: {reason}")

# Use the extractor standalone
extractor = DocExtractor()
info = extractor.extract("src/mypackage/core.py")
for cls in info.classes:
    print(cls.name, [m.name for m in cls.methods])
for fn in info.functions:
    print(fn.name, fn.signature)
```

## Architecture

```
doc_agent/
  __init__.py     # Public exports: DocAgent, DocExtractor
  extractor.py    # DocExtractor — AST parsing, returns ModuleInfo
  agent.py        # DocAgent — agentic loop, tool dispatch, Claude calls
  cli.py          # Click CLI: generate + check commands
```

### Tools available to the agent

| Tool | Description |
|------|-------------|
| `read_file(path)` | Read a source file for extra context |
| `list_files(dir, pattern)` | Discover Python files in a directory |
| `write_file(path, content)` | Persist generated documentation |

### Agentic loop

1. `DocExtractor` parses the source file with Python's built-in `ast` module.
2. The structured API info (classes, methods, signatures, docstrings) is
   serialised as JSON and sent to Claude in a user message.
3. Claude may call tools to read more source code, then writes the final
   documentation as a text response.
4. The loop runs until `stop_reason == "end_turn"` (max 20 iterations as a
   safety cap).

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check doc_agent/
mypy doc_agent/
```

## License

MIT
