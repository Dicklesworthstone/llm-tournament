# Changelog

All notable changes to the **llm-tournament** project are documented here.

This project has no versioned releases or tags. History is tracked by commit.
Links point to the canonical GitHub repository at
<https://github.com/Dicklesworthstone/llm-tournament>.

---

## 2026-02-22 — License documentation update

[`e5299d9`](https://github.com/Dicklesworthstone/llm-tournament/commit/e5299d9a8b55dd0b5a57fe42a53b1295957c9ede)

### Documentation
- Updated README license section to reference "MIT License (with OpenAI/Anthropic Rider)" consistently.

---

## 2026-02-21 — License rider and social preview

[`63c9a59`](https://github.com/Dicklesworthstone/llm-tournament/commit/63c9a59c06e654199e3a3e256e7ebe8212fb0c3b)
&ensp;
[`49d451b`](https://github.com/Dicklesworthstone/llm-tournament/commit/49d451b66a28e07901b9bc6dcdb9bde0c7693fec)

### Licensing
- Replaced plain MIT license with **MIT + OpenAI/Anthropic Rider**, restricting use by OpenAI, Anthropic, and their affiliates without express written permission from Jeffrey Emanuel.

### Assets
- Added `gh_og_share_image.png` (1280x640) for GitHub social preview / Open Graph link cards.

---

## 2026-02-11 — Codebase cleanup

[`b6fa61d`](https://github.com/Dicklesworthstone/llm-tournament/commit/b6fa61dc057f25ea872957c6213c08a61dc5c98c)

### Code quality (main tournament script)
- Removed unused imports from `llm_tournament.py`: `textwrap`, `threading`, `Union`, `Set` from `typing`.
- Converted f-strings with no interpolation to plain strings.
- Removed dead variable assignment (`class_name` was assigned but never read).

### Code quality (tournament results)
- `analyze_results.py`: removed unused imports.
- `run_tests.py`: removed unused import.
- `test_all_solutions.py`: major simplification removing ~130 lines of redundant test infrastructure (7,952 lines reduced to ~408).
- 12 extracted solution files across rounds 0--5: removed unused imports and fixed spurious f-string prefixes.

---

## 2026-01-21 — MIT License added

[`7662821`](https://github.com/Dicklesworthstone/llm-tournament/commit/7662821e175826f8cb26113a4abb49d92bafa355)

### Licensing
- Added `LICENSE` file with MIT License, copyright 2026 Jeffrey Emanuel.

---

## 2026-01-18 — Dependency upgrade log

[`14edc14`](https://github.com/Dicklesworthstone/llm-tournament/commit/14edc14d8c90d9fbc81b1e084ad4ecbdf32e995d)

### Documentation
- Added `UPGRADE_LOG.md` documenting that `requirements.txt` uses unpinned versions; dependencies are pulled at latest compatible versions at install time.
- Routine maintenance review performed via the library-updater workflow.

---

## 2025-03-23 — Initial release and tournament run

The entire functional codebase was created, iterated, and exercised in a single day across six commits.

### Tournament engine

[`f4ee9d1`](https://github.com/Dicklesworthstone/llm-tournament/commit/f4ee9d125ad6fe9f4a633134de30da90a662a027)
&ensp;
[`c203907`](https://github.com/Dicklesworthstone/llm-tournament/commit/c20390780bb54f2ccdaf5c87896133ecb5e462a3)
&ensp;
[`2bd0c39`](https://github.com/Dicklesworthstone/llm-tournament/commit/2bd0c39bd985e3552db0f6e6430d0f73e967a0ac)

#### Multi-round orchestration
- `LLMTournament` class managing the full lifecycle: prompt creation, concurrent API calls, response collection, code extraction, test generation, and metrics reporting.
- Configurable rounds (default 5), temperature (default 0.7), and concurrency (default 4) via CLI.
- CLI via `argparse`: `--prompt`, `--rounds`, `--output-dir`, `--test-file`, `--temperature`, `--concurrent-requests`, `--skip-tests`, `--verbose`.

#### Dynamic prompt synthesis
- Each round's prompt incorporates all prior solutions so models analyze and improve on peers' work.
- Prompts grow from 65 lines (round 0) to 1,583+ lines (round 5) as context accumulates.

#### Concurrent execution and error handling
- `ThreadPoolExecutor` with configurable parallelism (`--concurrent-requests`).
- Retry logic (`MAX_RETRIES = 3`) with exponential backoff (5s, 15s, 30s).
- Provider-specific API parameter handling: strips unsupported parameters on the fly (e.g., o3-mini does not accept `temperature` or `max_tokens`).

#### AI-powered code extraction
- Raw LLM outputs are sent to Claude 3.7 Sonnet to restructure into self-contained Python classes (`{ModelName}Round{N}Solution` with a static `solve(input_text)` method).
- Extracted code is cached to disk (`extracted_code/` directory) and reused on subsequent runs.
- Fallback: if AI extraction fails, generates a minimal passthrough class.
- Originally used regex-based code-block parsing ([`f4ee9d1`](https://github.com/Dicklesworthstone/llm-tournament/commit/f4ee9d125ad6fe9f4a633134de30da90a662a027)); replaced with AI extraction in [`c203907`](https://github.com/Dicklesworthstone/llm-tournament/commit/c20390780bb54f2ccdaf5c87896133ecb5e462a3).

#### Response caching
- Both raw responses and extracted code are persisted to disk.
- Re-runs detect existing files and skip already-collected data, avoiding redundant API calls.

#### Metrics and complexity analysis
- `ModelResponse` dataclass: stores responses per model/round, computes complexity metrics (function count, class count, import count, cyclomatic complexity estimate via decision-point counting).
- Complexity regex patterns updated to work with class-structured code (improved `from ... import` counting, class-aware function detection).

### Model roster

[`f4ee9d1`](https://github.com/Dicklesworthstone/llm-tournament/commit/f4ee9d125ad6fe9f4a633134de30da90a662a027)
&ensp;
[`c203907`](https://github.com/Dicklesworthstone/llm-tournament/commit/c20390780bb54f2ccdaf5c87896133ecb5e462a3)

#### Original roster (initial commit)
| Key       | Provider  | Model ID                                        | Thinking | Max Tokens |
|-----------|-----------|------------------------------------------------|----------|------------|
| `claude37`| Anthropic | `anthropic:claude-3-7-sonnet-20240219`         | Yes      | 100,000    |
| `gemini`  | Google    | `google:gemini-2.0-flash-thinking-exp-01-21`   | Yes      | 8,192      |
| `o1_pro`  | OpenAI    | `openai:o1-pro`                                | No       | 4,096      |
| `o3_mini` | OpenAI    | `openai:o3-mini-high`                          | No       | 4,096      |

#### Final roster (after refactor)
| Key              | Provider  | Model ID                               | Thinking |
|------------------|-----------|----------------------------------------|----------|
| `o3_mini`        | OpenAI    | `openai:o3-mini`                       | Yes      |
| `gpt4o`          | OpenAI    | `openai:gpt-4o`                        | No       |
| `claude37`       | Anthropic | `anthropic:claude-3-7-sonnet-20250219` | Yes      |
| `mistral_large`  | Mistral   | `mistral:mistral-large-latest`         | Yes      |

**Changes:** Removed Gemini and o1-pro. Added GPT-4o and Mistral Large. Corrected Claude 3.7 Sonnet model ID (`20240219` to `20250219`). Changed o3-mini from `o3-mini-high` to `o3-mini` with thinking enabled.

### Challenge and test data

[`f4ee9d1`](https://github.com/Dicklesworthstone/llm-tournament/commit/f4ee9d125ad6fe9f4a633134de30da90a662a027)

- `challenge_prompt.md`: CSV Normalization and Cleaning Challenge -- a `normalize_csv` function spec covering inconsistent delimiters, quoting, encoding, whitespace, date formats, numeric formats, duplicates, and headers.
- `messy_csv_sample.csv`: 31-line sample with deliberately messy data for solution validation.

### Tournament artifacts

[`2bd0c39`](https://github.com/Dicklesworthstone/llm-tournament/commit/2bd0c39bd985e3552db0f6e6430d0f73e967a0ac)

Over 40,000 lines of tournament output committed after running all 6 rounds:

- **24 raw responses** (`tournament_response__round_{0-5}__{model}.md`) -- full outputs from claude37, gpt4o, mistral_large, o3_mini across 6 rounds.
- **24 extracted code files** (`extracted_code__round_{0-5}__{model}.py`) -- AI-structured solution classes, growing from ~80--320 lines (round 0) to ~360--484 lines (round 5).
- **6 round prompts** (`prompt_round_{0-5}.md`) -- dynamically generated prompts showing how context accumulates.
- **5 comparison tables** (`markdown_table_prompt_response_comparison__round_{1-5}.md`) -- side-by-side prompt vs. response analysis for rounds 1--5.
- **24 per-model output samples** (`output_results_for_each_round_and_model/sample_file_output__{model}_round_{0-5}.md`).
- `metrics/tournament_metrics.json` (378 lines) and `metrics/test_metrics.json` (242 lines): detailed per-model, per-round performance data.
- `metrics/tournament_report.md` and `analysis/tournament_results_report.md`: summary reports.

### Analysis and testing tools

[`2bd0c39`](https://github.com/Dicklesworthstone/llm-tournament/commit/2bd0c39bd985e3552db0f6e6430d0f73e967a0ac)

- `test_all_solutions.py` (7,952 lines): generated test harness exercising all 24 solutions against the messy CSV sample.
- `analyze_results.py` (214 lines): results analysis and optional matplotlib visualization.
- `run_tests.py` (34 lines): convenience test runner wrapper.

### Documentation

[`2bd0c39`](https://github.com/Dicklesworthstone/llm-tournament/commit/2bd0c39bd985e3552db0f6e6430d0f73e967a0ac)
&ensp;
[`14a9d80`](https://github.com/Dicklesworthstone/llm-tournament/commit/14a9d800dc8d9b5dbc49e621eaaf308781cec6d1)

- Complete README rewrite with project overview, "Why this Project Matters" section, core features, implementation details (AI-powered extraction, dynamic prompts, metrics), and expanded usage docs.
- Added hero image (`llm-tournament.webp`).
- Linked to the companion article at `github.com/Dicklesworthstone/llm_multi_round_coding_tournament`.
- Fixed image URL to use `raw.githubusercontent.com` for proper rendering ([`14a9d80`](https://github.com/Dicklesworthstone/llm-tournament/commit/14a9d800dc8d9b5dbc49e621eaaf308781cec6d1)).

### Infrastructure

[`ead8526`](https://github.com/Dicklesworthstone/llm-tournament/commit/ead852664f50ccad9df321390253b7f1afcec5ec)
&ensp;
[`c203907`](https://github.com/Dicklesworthstone/llm-tournament/commit/c20390780bb54f2ccdaf5c87896133ecb5e462a3)
&ensp;
[`d9f832d`](https://github.com/Dicklesworthstone/llm-tournament/commit/d9f832dcc6583321affc728e18982734222e47c3)
&ensp;
[`2bd0c39`](https://github.com/Dicklesworthstone/llm-tournament/commit/2bd0c39bd985e3552db0f6e6430d0f73e967a0ac)

- `.gitignore` (174 lines, comprehensive Python/IDE/OS ignores).
- `.env` template for API keys (Anthropic, OpenAI, Mistral).
- `requirements.txt`: `aisuite[all]`, `pydantic`, `docstring_parser`, `dotenv`, `mistralai`.
- Integrated `python-dotenv` for `.env`-based API key loading (replacing shell-exported environment variables).
- Provider abstraction via `aisuite` unifying OpenAI, Anthropic, and Mistral APIs behind a single interface.
- Indentation fix in generated `test_all_solutions.py` template: test functions and `main()` were incorrectly indented inside the f-string, causing them to be nested in the module-level code ([`2bd0c39`](https://github.com/Dicklesworthstone/llm-tournament/commit/2bd0c39bd985e3552db0f6e6430d0f73e967a0ac)).

---

## Architecture overview

```
llm-tournament/
  llm_tournament.py          # Main tournament orchestrator (~1,200 lines)
  challenge_prompt.md         # Challenge spec (CSV normalization)
  messy_csv_sample.csv        # Test data (31 rows)
  requirements.txt            # aisuite[all], pydantic, docstring_parser, dotenv, mistralai
  tournament_results/
    prompt_round_{0-5}.md                       # Dynamic prompts per round
    tournament_response__round_{0-5}__{model}.md  # Raw LLM outputs
    extracted_code__round_{0-5}__{model}.py      # AI-extracted solution classes
    output_results_for_each_round_and_model/     # Per-solution output samples
    markdown_table_prompt_response_comparison__round_{1-5}.md
    metrics/                                     # JSON metrics + reports
    analysis/                                    # Tournament results report
    test_all_solutions.py                        # Generated test harness
    analyze_results.py                           # Results analysis script
    run_tests.py                                 # Test runner
```

### Key design decisions
- **AI-powered code extraction** (not regex): raw LLM outputs are sent to Claude 3.7 Sonnet to restructure into self-contained classes with a `solve(input_text)` static method.
- **Response caching**: both raw responses and extracted code are persisted to disk; re-runs skip already-collected data.
- **Provider abstraction**: uses `aisuite` to unify OpenAI, Anthropic, and Mistral APIs behind a single interface.
- **Concurrent execution**: `ThreadPoolExecutor` with configurable parallelism and retry/backoff logic.
