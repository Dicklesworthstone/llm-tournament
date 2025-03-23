# LLM Multi-Round Coding Tournament

This project implements an automated tournament where multiple LLMs collaborate and compete to solve a coding challenge across multiple rounds of refinement.

## Prerequisites

- API keys for the LLM providers (Anthropic, OpenAI, Mistral)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Dicklesworthstone/llm-tournament
   cd llm-tournament
   ```

2. Install dependencies (using `uv` for virtual environment management):

   ```bash

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv --python 3.13                                                                                                                                                                                                                   ✔  base   at 04:42:14 AM 
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

3. Set up API keys as environment variables:

   ```bash
   export ANTHROPIC_API_KEY="your_anthropic_key"
   export OPENAI_API_KEY="your_openai_key"
   export MISTRAL_API_KEY="your_mistral_key"
   ```

## File Structure

- `llm_tournament.py`: The main tournament script
- `challenge_prompt.md`: The coding challenge prompt
- `messy_csv_sample.csv`: A test file for evaluating solutions
- `README.md`: This file

## Usage

### Basic Usage

Run a tournament with default settings:

```bash
python llm_tournament.py --prompt challenge_prompt.md --test-file messy_csv_sample.csv
```

Note that, if you already have a response saved in the `tournament_results` directory, the script will skip that round and use the saved response instead (assuming there is a valid code block found within the response).

### Advanced Options

```bash
python llm_tournament.py --prompt challenge_prompt.md --test-file messy_csv_sample.csv --rounds 3 --temperature 0.8 --concurrent-requests 4 --verbose
```

### Options

- `--prompt`: File containing the coding challenge prompt (required)
- `--rounds`: Number of tournament rounds (default: 5)
- `--output-dir`: Directory for tournament results (default: "tournament_results")
- `--test-file`: File to use for testing solutions
- `--temperature`: Temperature for LLM generation (default: 0.7)
- `--concurrent-requests`: Maximum number of concurrent API requests (default: 4)
- `--skip-tests`: Skip running tests on the solutions
- `--verbose`: Enable verbose logging

## Output

The script will create a directory structure with:

- Individual responses from each model for each round
- Combined prompts for each round
- Test results showing how each solution performs
- Metrics and analysis of the tournament

## Example

Here's a sample run:

```bash
python llm_tournament.py --prompt challenge_prompt.md --test-file messy_csv_sample.csv --rounds 3
```

This will:

1. Submit the CSV normalization challenge to 4 different LLMs
2. Collect their initial solutions
3. Have them review and integrate each other's solutions over 3 rounds
4. Test all solutions on the sample messy CSV file
5. Generate metrics and visualizations of the results

## License

MIT
