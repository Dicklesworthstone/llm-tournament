#!/usr/bin/env python3
"""
LLM Multi-Round Coding Tournament Automation

This script automates the process of running a multi-round coding tournament between
different LLMs, where models analyze and improve each other's solutions across
multiple rounds of refinement and integration.

Features:
- Supports multiple LLM providers through the aisuite API
- Creates a structured directory hierarchy for organizing tournament artifacts
- Manages the full lifecycle of prompt creation, response collection, and analysis
- Generates test scripts to evaluate the performance of each solution
- Tracks and reports detailed metrics about each solution
- Supports configurable tournament parameters (rounds, models, etc.)
- Handles error recovery and rate limiting

External dependencies:

Just `pip install aisuite`
"""

import os
import re
import time
import json
import logging
import argparse
import traceback
import statistics
import subprocess
import threading
from typing import List, Dict, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import aisuite as ai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tournament.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llm_tournament")

# Model configurations with appropriate parameters
MODELS = {
    "claude37": {
        "id": "anthropic:claude-3-7-sonnet-20240219",
        "max_tokens": 100000,
        "thinking": True,
        "provider": "anthropic"
    },
    "gemini": {
        "id": "google:gemini-2.0-flash-thinking-exp-01-21",
        "max_tokens": 8192,
        "thinking": True,
        "provider": "google"
    },
    "o1_pro": {
        "id": "openai:o1-pro",
        "max_tokens": 4096,
        "thinking": False,
        "provider": "openai"
    },
    "o3_mini": {
        "id": "openai:o3-mini-high",
        "max_tokens": 4096,
        "thinking": False,
        "provider": "openai"
    }
}

# Maximum retry attempts for API calls
MAX_RETRIES = 3

# Backoff times (in seconds) for retrying API calls
BACKOFF_TIMES = [5, 15, 30]

@dataclass
class ModelResponse:
    """Class for storing and processing a single model's response"""
    model_name: str
    round_num: int
    prompt: str
    response: str
    file_path: Optional[Path] = None
    code: Optional[str] = None
    thinking: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def extract_code(self) -> str:
        """
        Extract the code portion from the model's response.
        Handles various code block formats and edge cases.
        
        Returns:
            Extracted code as a string
        """
        if not self.response:
            return ""
            
        # Extract code blocks (handles both ```python and ``` formats)
        code_blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', self.response, re.DOTALL)
        
        if code_blocks:
            # Join all code blocks together with appropriate spacing
            full_code = "\n\n".join(code_blocks)
            
            # Remove any leading/trailing whitespace
            full_code = full_code.strip()
            
            # Store the extracted code
            self.code = full_code
            return full_code
        
        # Try other formats like <code> tags or indented blocks
        if "<code>" in self.response and "</code>" in self.response:
            code_blocks = re.findall(r'<code>(.*?)</code>', self.response, re.DOTALL)
            if code_blocks:
                self.code = "\n\n".join(code_blocks).strip()
                return self.code
        
        # If all extraction methods fail, use the entire response as a fallback
        # (this is not ideal but provides a graceful fallback)
        logger.warning(f"Could not extract code blocks from {self.model_name}'s response (round {self.round_num})")
        self.code = self.response
        return self.response
    
    def extract_thinking(self) -> Optional[str]:
        """
        Extract the thought process or reasoning from the model's response.
        
        Returns:
            Extracted thinking content if found, None otherwise
        """
        # Look for chain-of-thought sections marked with common patterns
        patterns = [
            r'<thinking>(.*?)</thinking>',
            r'### Thinking\s*\n(.*?)(?:\n###|\Z)',
            r'## Thought Process\s*\n(.*?)(?:\n##|\Z)',
            r'Step-by-step reasoning:\s*\n(.*?)(?:\n#|\Z)',
            r'Let me think through this:\s*\n(.*?)(?:\nFinal answer:|\Z)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, self.response, re.DOTALL)
            if matches:
                self.thinking = matches[0].strip()
                return self.thinking
        
        return None
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate various metrics about the response and code.
        
        Returns:
            Dictionary of calculated metrics
        """
        # Extract code if not already done
        if not self.code:
            self.extract_code()
            
        # Calculate basic metrics
        response_size = len(self.response)
        response_lines = self.response.count('\n') + 1
        
        code_size = len(self.code) if self.code else 0
        code_lines = self.code.count('\n') + 1 if self.code else 0
        
        # Compute code complexity metrics
        complexity_metrics = self._compute_code_complexity()
        
        # Combine metrics
        metrics = {
            "response_size_kb": round(response_size / 1024, 2),
            "response_lines": response_lines,
            "code_size_kb": round(code_size / 1024, 2),
            "code_lines": code_lines,
            "timestamp": self.timestamp.isoformat(),
            **complexity_metrics
        }
        
        self.metrics = metrics
        return metrics
    
    def _compute_code_complexity(self) -> Dict[str, Any]:
        """
        Compute code complexity metrics like function count, class count, etc.
        
        Returns:
            Dictionary of complexity metrics
        """
        if not self.code:
            return {}
            
        # Count functions
        function_count = len(re.findall(r'\bdef\s+([a-zA-Z0-9_]+)\s*\(', self.code))
        
        # Count classes
        class_count = len(re.findall(r'\bclass\s+([a-zA-Z0-9_]+)', self.code))
        
        # Count import statements
        import_count = len(re.findall(r'\bimport\s+([a-zA-Z0-9_., ]+)', self.code))
        
        # Estimate cyclomatic complexity by counting decision points
        decision_patterns = [
            r'\bif\s+', r'\belif\s+', r'\belse\s*:', 
            r'\bfor\s+', r'\bwhile\s+', r'\bexcept\s*',
            r'\btry\s*:'
        ]
        
        decision_count = sum(len(re.findall(pattern, self.code)) for pattern in decision_patterns)
        
        return {
            "function_count": function_count,
            "class_count": class_count,
            "import_count": import_count,
            "decision_points": decision_count,
            "complexity_estimate": decision_count / (function_count if function_count else 1)
        }
    
    def save_to_file(self, output_dir: Path) -> Path:
        """
        Save the response to a file.
        
        Args:
            output_dir: Directory where the file should be saved
            
        Returns:
            Path to the saved file
        """
        # Create the output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a filename from the model name and round number
        filename = f"tournament_response__round_{self.round_num}__{self.model_name}.md"
        file_path = output_dir / filename
        
        # Write the response to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.response)
        
        self.file_path = file_path
        return file_path


class LLMTournament:
    """
    Main class for managing the multi-round LLM tournament.
    
    This class handles the full lifecycle of the tournament, including:
    - Creating and managing the directory structure
    - Querying LLMs for responses
    - Creating prompts for each round
    - Tracking metrics and generating reports
    - Testing solutions
    """
    
    def __init__(
        self, 
        prompt: str, 
        rounds: int = 5, 
        output_dir: str = "tournament_results",
        models: Dict[str, Dict[str, Any]] = None,
        temperature: float = 0.7,
        concurrent_requests: int = 2,
        test_file: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the tournament with the given parameters.
        
        Args:
            prompt: The initial coding challenge prompt
            rounds: Number of rounds to run (not including round 0)
            output_dir: Base directory for tournament artifacts
            models: Dictionary of model configurations (defaults to MODELS)
            temperature: Temperature setting for generation
            concurrent_requests: Maximum number of concurrent API requests
            test_file: Optional path to a file for testing solutions
            verbose: Whether to enable verbose logging
        """
        self.prompt = prompt
        self.rounds = rounds
        self.output_dir = Path(output_dir)
        self.models = models or MODELS
        self.temperature = temperature
        self.concurrent_requests = concurrent_requests
        self.test_file = test_file
        self.verbose = verbose
        
        # Set logging level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize the AI client
        self.client = ai.Client()
        
        # Store responses for each round and model
        self.responses: Dict[int, Dict[str, ModelResponse]] = {i: {} for i in range(rounds + 1)}
        
        # Initialize metrics tracking
        self.metrics: Dict[str, Any] = {}
        
        # Create directory structure
        self.setup_directories()
        
        # Log initialization
        logger.info(f"Initialized LLM Tournament with {len(self.models)} models for {rounds} rounds")
        logger.info(f"Output directory: {self.output_dir}")
        logger.debug(f"Models: {list(self.models.keys())}")

    def setup_directories(self) -> None:
        """
        Create the necessary directory structure for the tournament.
        
        Directory structure:
        - output_dir/
          - round_0_responses/
          - round_1_responses/
          - ...
          - round_N_responses/
          - output_results_for_each_round_and_model/
          - metrics/
        """
        # Create main output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create round-specific directories
        for i in range(self.rounds + 1):
            round_dir = self.output_dir / f"round_{i}_responses"
            round_dir.mkdir(exist_ok=True)
        
        # Create output results directory
        results_dir = self.output_dir / "output_results_for_each_round_and_model"
        results_dir.mkdir(exist_ok=True)
        
        # Create metrics directory
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        logger.debug(f"Created directory structure in {self.output_dir}")

    def query_model(
        self,
        model_name: str,
        round_num: int,
        prompt_text: str
    ) -> ModelResponse:
        """
        Query an LLM with the given prompt and handle retries and errors.
        
        Args:
            model_name: Name of the model to query
            round_num: Current tournament round number
            prompt_text: The prompt to send to the model
            
        Returns:
            ModelResponse object containing the model's response and metadata
        """
        model_config = self.models[model_name]
        model_id = model_config["id"]
        thinking_enabled = model_config.get("thinking", False)
        max_tokens = model_config.get("max_tokens", 4096)
        
        # Create system message based on model capabilities
        system_message = "You are an expert programmer specializing in writing clean, efficient, and robust code."
        
        if thinking_enabled:
            system_message += (
                " Please think through the problem carefully before providing your solution. "
                "Show your reasoning process and explain key design decisions."
            )
        
        logger.info(f"Querying {model_name} (round {round_num})...")
        
        # Create a ModelResponse object to store the result
        response_obj = ModelResponse(
            model_name=model_name,
            round_num=round_num,
            prompt=prompt_text,
            response=""
        )
        
        # Try the API call with retries
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                
                # Make the API call
                api_response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Extract and store the response text
                response_text = api_response.choices[0].message.content
                response_obj.response = response_text
                response_obj.metrics["response_time"] = round(response_time, 2)
                
                # Extract code and thinking components
                response_obj.extract_code()
                response_obj.extract_thinking()
                
                # Calculate metrics
                response_obj.calculate_metrics()
                
                # Log success
                logger.info(f"Received response from {model_name} ({round(response_time, 2)}s)")
                logger.debug(f"Response metrics: {response_obj.metrics}")
                
                # Store in the responses dictionary
                self.responses[round_num][model_name] = response_obj
                
                return response_obj
                
            except Exception as e:
                # Log the error
                error_msg = f"Error querying {model_name} (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                
                # Store error information
                response_obj.error = error_msg
                
                # Retry with backoff if not the last attempt
                if attempt < MAX_RETRIES - 1:
                    backoff_time = BACKOFF_TIMES[min(attempt, len(BACKOFF_TIMES) - 1)]
                    logger.info(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                else:
                    # If all retries failed, return the error response
                    response_obj.response = f"ERROR: Failed to get response after {MAX_RETRIES} attempts.\n\n{error_msg}"
                    return response_obj
        
        # This should never be reached, but just in case
        return response_obj

    def create_round_prompt(self, round_num: int) -> str:
        """
        Create the prompt for a specific round by combining responses from the previous round.
        
        Args:
            round_num: The round number to create a prompt for
            
        Returns:
            The combined prompt text for the next round
        """
        # For round 0, just use the original prompt
        if round_num == 0:
            return self.prompt
        
        # For subsequent rounds, combine the responses from the previous round
        prev_round = round_num - 1
        
        # Start with the standard combination prompt
        combined_prompt = f"""
I have the following problem which I posed to 4 different LLMs. I want you to carefully read the problem and then each solution. Choose the best ideas and elements from ALL solutions to the extent they are complementary rather than conflicting/inconsistent, and then weave together a true hybrid "best of all worlds" implementation which you are highly confident will not only work, but will outperform any of the individual solutions individually.

Original prompt:

{self.prompt}

Responses from different LLMs:
"""
        
        # Add each model's response
        for model_name, model_response in self.responses[prev_round].items():
            # For round 1, include the full response; for later rounds, just include the code
            if round_num == 1:
                content = model_response.response
            else:
                content = model_response.code or model_response.response
            
            combined_prompt += f"\n\n{model_name}:\n\n```python\n{content}\n```\n"
        
        # Add specific instructions for synthesis
        combined_prompt += """
Analyze each solution carefully, identifying strengths and weaknesses. Consider:
1. Correctness - Does the code handle all cases properly?
2. Efficiency - Is the code optimized for performance?
3. Readability - Is the code clear and maintainable?
4. Robustness - Does the code handle errors gracefully?

Then create a new implementation that combines the best aspects of all solutions.
Your implementation should be complete and ready to use without modification.
"""
        
        return combined_prompt

    def run_round(self, round_num: int) -> Dict[str, ModelResponse]:
        """
        Run a single round of the tournament, querying all models in parallel.
        
        Args:
            round_num: The round number to run
            
        Returns:
            Dictionary mapping model names to their responses
        """
        logger.info(f"Starting Round {round_num}")
        
        # Create the prompt for this round
        round_prompt = self.create_round_prompt(round_num)
        
        # Save the prompt for reference
        prompt_file = self.output_dir / f"prompt_round_{round_num}.md"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(round_prompt)
        
        # Query all models in parallel
        round_responses = {}
        
        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            # Submit all model queries
            future_to_model = {
                executor.submit(self.query_model, model_name, round_num, round_prompt): model_name
                for model_name in self.models.keys()
            }
            
            # Process results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    response = future.result()
                    round_responses[model_name] = response
                    
                    # Save the response to a file
                    response_dir = self.output_dir / f"round_{round_num}_responses"
                    response.save_to_file(response_dir)
                    
                    logger.info(f"Saved response from {model_name} for round {round_num}")
                    
                except Exception as e:
                    logger.error(f"Error processing response from {model_name}: {str(e)}")
                    logger.debug(traceback.format_exc())
        
        # Create comparison file for all responses in this round
        self.create_round_comparison_file(round_num)
        
        # Update metrics
        self.update_metrics(round_num, round_responses)
        
        # Return the responses
        return round_responses

    def create_round_comparison_file(self, round_num: int) -> Path:
        """
        Create a markdown file comparing all responses for a specific round.
        
        Args:
            round_num: The round number to create a comparison for
            
        Returns:
            Path to the created comparison file
        """
        if round_num == 0:
            return None
            
        # Create the comparison content
        comparison_content = f"""# Round {round_num} Response Comparison

## Original Prompt

```
{self.prompt}
```

## Model Responses

"""
        
        # Add each model's response
        for model_name, response in self.responses[round_num].items():
            # Extract metrics
            metrics = response.metrics
            code_lines = metrics.get("code_lines", 0)
            code_size = metrics.get("code_size_kb", 0)
            
            comparison_content += f"### {model_name}\n\n"
            comparison_content += f"**Metrics:** {code_lines} lines, {code_size} KB\n\n"
            comparison_content += "```python\n"
            comparison_content += response.code or "# No code extracted"
            comparison_content += "\n```\n\n"
        
        # Save the comparison file
        comparison_file = self.output_dir / f"markdown_table_prompt_response_comparison__round_{round_num}.md"
        with open(comparison_file, "w", encoding="utf-8") as f:
            f.write(comparison_content)
            
        logger.info(f"Created comparison file for round {round_num}: {comparison_file}")
        return comparison_file

    def update_metrics(self, round_num: int, round_responses: Dict[str, ModelResponse]) -> None:
        """
        Update the tournament metrics with the results from a round.
        
        Args:
            round_num: The round number
            round_responses: Dictionary of model responses from the round
        """
        # Create metrics structure if it doesn't exist
        if "rounds" not in self.metrics:
            self.metrics["rounds"] = {}
        
        # Create metrics for this round
        round_metrics = {
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        # Add metrics for each model
        for model_name, response in round_responses.items():
            round_metrics["models"][model_name] = response.metrics
        
        # Calculate aggregate metrics
        code_sizes = [r.metrics.get("code_size_kb", 0) for r in round_responses.values()]
        code_lines = [r.metrics.get("code_lines", 0) for r in round_responses.values()]
        
        round_metrics["aggregate"] = {
            "avg_code_size_kb": round(statistics.mean(code_sizes), 2) if code_sizes else 0,
            "avg_code_lines": round(statistics.mean(code_lines), 2) if code_lines else 0,
            "max_code_size_kb": round(max(code_sizes), 2) if code_sizes else 0,
            "max_code_lines": max(code_lines) if code_lines else 0,
            "min_code_size_kb": round(min(code_sizes), 2) if code_sizes else 0,
            "min_code_lines": min(code_lines) if code_lines else 0
        }
        
        # Store the metrics
        self.metrics["rounds"][round_num] = round_metrics
        
        # Save metrics to file
        self.save_metrics()

    def save_metrics(self) -> Path:
        """
        Save the tournament metrics to a JSON file.
        
        Returns:
            Path to the saved metrics file
        """
        metrics_file = self.output_dir / "metrics" / "tournament_metrics.json"
        
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
            
        logger.debug(f"Saved metrics to {metrics_file}")
        return metrics_file

    def generate_metrics_report(self) -> Path:
        """
        Generate a detailed metrics report in markdown format.
        
        Returns:
            Path to the generated report
        """
        # Create report content
        report_content = f"""# LLM Tournament Metrics Report

## Tournament Overview

- **Start Time:** {self.metrics.get("start_time", "Unknown")}
- **End Time:** {self.metrics.get("end_time", "Unknown")}
- **Total Rounds:** {self.rounds + 1} (including round 0)
- **Models:** {", ".join(self.models.keys())}

## Round Summaries

"""
        
        # Add metrics for each round
        for round_num in range(self.rounds + 1):
            round_metrics = self.metrics.get("rounds", {}).get(str(round_num), {})
            if not round_metrics:
                continue
                
            report_content += f"### Round {round_num}\n\n"
            
            # Create a table of metrics
            report_content += "| Model | Code Size (KB) | Code Lines | Functions | Complexity |\n"
            report_content += "|-------|---------------|------------|-----------|------------|\n"
            
            for model_name, model_metrics in round_metrics.get("models", {}).items():
                code_size = model_metrics.get("code_size_kb", "N/A")
                code_lines = model_metrics.get("code_lines", "N/A")
                function_count = model_metrics.get("function_count", "N/A")
                complexity = model_metrics.get("complexity_estimate", "N/A")
                
                report_content += f"| {model_name} | {code_size} | {code_lines} | {function_count} | {complexity} |\n"
            
            report_content += "\n"
        
        # Add convergence analysis
        report_content += "## Convergence Analysis\n\n"
        
        # Plot code size and lines over rounds
        report_content += "### Code Size Over Rounds\n\n"
        report_content += "| Round | " + " | ".join(self.models.keys()) + " |\n"
        report_content += "|-------| " + " | ".join(["-" * len(name) for name in self.models.keys()]) + " |\n"
        
        for round_num in range(self.rounds + 1):
            round_metrics = self.metrics.get("rounds", {}).get(str(round_num), {})
            if not round_metrics:
                continue
                
            row = [str(round_num)]
            
            for model_name in self.models.keys():
                model_metrics = round_metrics.get("models", {}).get(model_name, {})
                code_size = model_metrics.get("code_size_kb", "N/A")
                row.append(str(code_size))
                
            report_content += "| " + " | ".join(row) + " |\n"
        
        # Save the report
        report_file = self.output_dir / "metrics" / "tournament_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        logger.info(f"Generated metrics report: {report_file}")
        return report_file

    def create_test_suite(self) -> Tuple[Path, Path]:
        """
        Create a comprehensive test suite for evaluating all solutions.
        
        Returns:
            Tuple containing paths to the test script and test runner
        """
        # Extract all solution code
        solution_classes = []
        
        for round_num in range(self.rounds + 1):
            for model_name, response in self.responses.get(round_num, {}).items():
                if not response.code:
                    continue
                    
                # Clean model name for use as a class name
                clean_model_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
                class_name = f"{clean_model_name.title()}Round{round_num}Solution"
                
                # Create a class definition
                class_code = f"""
class {class_name}:
    \"\"\"Solution from {model_name} at round {round_num}\"\"\"
    
    @staticmethod
    def solve(input_text):
        \"\"\"Apply the solution to the input text\"\"\"
        # Define any necessary helper functions or imports
        import re
        
        # Original code from {model_name} (round {round_num})
{response.code.replace('import ', '        import ').replace('\ndef ', '\n        def ').replace('\nclass ', '\n        class ').replace('\n', '\n        ')}
        
        # Try to find and call the main function
        # This is a heuristic approach - may need adjustment for specific challenges
        possible_functions = [
            'fix_invalid_markdown_tables',
            'fix_markdown_tables',
            'process_markdown',
            'repair_markdown_tables',
            'solve'
        ]
        
        for func_name in possible_functions:
            if func_name in locals():
                try:
                    func = locals()[func_name]
                    if callable(func):
                        return func(input_text)
                except Exception as e:
                    return f"Error executing {func_name}: {str(e)}"
        
        # If no suitable function found, log an error
        return f"Error: Could not find a suitable function to call in {class_name}"
"""
                solution_classes.append(class_code)
        
        # Create the test script
        test_script = f"""#!/usr/bin/env python3
\"\"\"
LLM Tournament Test Suite

This script tests solutions from all rounds and models on a given input file,
and collects metrics on the results.
\"\"\"

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

{os.linesep.join(solution_classes)}

def count_lines(text: str) -> int:
    \"\"\"Count the number of lines in a text\"\"\"
    return len(text.splitlines())

def count_chars(text: str) -> int:
    \"\"\"Count the number of characters in a text\"\"\"
    return len(text)

def main():
    \"\"\"Test all solutions and collect metrics\"\"\"
    parser = argparse.ArgumentParser(description="Test LLM tournament solutions")
    parser.add_argument("--input", type=str, required=True, help="Input file to test on")
    parser.add_argument("--output-dir", type=str, default="output_results_for_each_round_and_model",
                      help="Directory for results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Read the input file
    with open(args.input, "r", encoding="utf-8") as f:
        input_text = f.read()
    
    # Get input file metrics
    input_lines = count_lines(input_text)
    input_chars = count_chars(input_text)
    print(f"Input file: {args.input}")
    print(f"Input lines: {input_lines}")
    print(f"Input chars: {input_chars}")
    
    # List of all solution classes
    solution_classes = [
        {", ".join([s.split()[1].split(":")[0] for s in solution_classes])}
    ]
    
    # Test each solution
    metrics = []
    
    for solution_class in solution_classes:
        class_name = solution_class.__name__
        print(f"\\nTesting {class_name}...")
        
        # Extract model name and round number
        parts = class_name.split("Round")
        model_name = parts[0].replace("_", "-").lower()
        round_num = parts[1].split("Solution")[0]
        
        try:
            # Apply the solution
            start_time = time.time()
            result = solution_class.solve(input_text)
            execution_time = time.time() - start_time
            
            # Save the result
            output_filename = f"sample_10k_reformatted__fixed_tables__{model_name}_round_{round_num}.md"
            output_path = output_dir / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
                
            # Calculate metrics
            output_lines = count_lines(result)
            output_chars = count_chars(result)
            output_size_kb = len(result) / 1024
            
            # Store metrics
            solution_metrics = {
                "model": model_name,
                "round": round_num,
                "execution_time": round(execution_time, 2),
                "output_lines": output_lines,
                "output_chars": output_chars,
                "output_size_kb": round(output_size_kb, 2),
                "lines_ratio": round(output_lines / input_lines, 2),
                "chars_ratio": round(output_chars / input_chars, 2)
            }
            
            metrics.append(solution_metrics)
            
            # Print metrics
            print(f"  Execution time: {solution_metrics['execution_time']}s")
            print(f"  Output lines: {output_lines}")
            print(f"  Output size: {solution_metrics['output_size_kb']} KB")
            print(f"  Output saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error testing {class_name}: {str(e)}")
    
    # Save metrics
    metrics_path = output_dir.parent / "metrics" / "test_metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\\nMetrics saved to: {metrics_path}")
    
if __name__ == "__main__":
    main()
"""
        
        # Create a test runner script
        test_runner = f"""#!/usr/bin/env python3
\"\"\"
LLM Tournament Test Runner

This script runs the test suite on the provided test file.
\"\"\"

import os
import subprocess
import argparse
from pathlib import Path

def main():
    \"\"\"Run the test suite\"\"\"
    parser = argparse.ArgumentParser(description="Run LLM tournament tests")
    parser.add_argument("--test-file", type=str, required=True, help="File to test on")
    args = parser.parse_args()
    
    # Path to the test script
    test_script = Path(__file__).parent / "test_all_solutions.py"
    
    # Run the test script
    cmd = [
        "python",
        str(test_script),
        "--input", args.test_file,
        "--output-dir", "output_results_for_each_round_and_model"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
if __name__ == "__main__":
    main()
"""
        
        # Save the test script
        test_script_path = self.output_dir / "test_all_solutions.py"
        with open(test_script_path, "w", encoding="utf-8") as f:
            f.write(test_script)
        
        # Save the test runner
        test_runner_path = self.output_dir / "run_tests.py"
        with open(test_runner_path, "w", encoding="utf-8") as f:
            f.write(test_runner)
        
        # Make the scripts executable
        os.chmod(test_script_path, 0o755)
        os.chmod(test_runner_path, 0o755)
        
        logger.info(f"Created test suite: {test_script_path}")
        return test_script_path, test_runner_path

    def create_results_analyzer(self) -> Path:
        """
        Create a script to analyze and visualize the tournament results.
        
        Returns:
            Path to the analyzer script
        """
        analyzer_script = """#!/usr/bin/env python3
\"\"\"
LLM Tournament Results Analyzer

This script analyzes and visualizes the results of the LLM tournament.
It creates plots and tables comparing the performance of different models
across rounds.
\"\"\"

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Optional import for visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def load_metrics(metrics_dir: str) -> Dict[str, Any]:
    \"\"\"Load metrics from JSON files\"\"\"
    metrics_dir = Path(metrics_dir)
    
    # Load tournament metrics
    tournament_metrics_path = metrics_dir / "tournament_metrics.json"
    tournament_metrics = {}
    
    if tournament_metrics_path.exists():
        with open(tournament_metrics_path, "r", encoding="utf-8") as f:
            tournament_metrics = json.load(f)
    
    # Load test metrics
    test_metrics_path = metrics_dir / "test_metrics.json"
    test_metrics = []
    
    if test_metrics_path.exists():
        with open(test_metrics_path, "r", encoding="utf-8") as f:
            test_metrics = json.load(f)
    
    return {
        "tournament": tournament_metrics,
        "test": test_metrics
    }

def generate_markdown_report(metrics: Dict[str, Any], output_file: str) -> None:
    \"\"\"Generate a markdown report from the metrics\"\"\"
    tournament_metrics = metrics.get("tournament", {})
    test_metrics = metrics.get("test", [])
    
    # Start building the report
    report = "# LLM Tournament Results\n\n"
    report += "## Overview\n\n"
    report += "This report summarizes the results of the LLM tournament.\n\n"
    
    # Add test metrics table if available
    if test_metrics:
        report += "## Test Results\n\n"
        report += "| Model | Round | Execution Time (s) | Output Lines | Output Size (KB) |\n"
        report += "|-------|-------|-------------------|--------------|------------------|\n"
        
        # Sort by round, then model
        sorted_metrics = sorted(test_metrics, key=lambda x: (int(x.get("round", 0)), x.get("model", "")))
        
        for metric in sorted_metrics:
            model = metric.get("model", "Unknown")
            round_num = metric.get("round", "Unknown")
            time = metric.get("execution_time", "N/A")
            lines = metric.get("output_lines", "N/A")
            size = metric.get("output_size_kb", "N/A")
            
            report += f"| {model} | {round_num} | {time} | {lines} | {size} |\n"
    
    # Add round metrics if available
    rounds_data = tournament_metrics.get("rounds", {})
    if rounds_data:
        report += "\n## Code Metrics by Round\n\n"
        report += "| Round | Model | Code Size (KB) | Code Lines |\n"
        report += "|-------|-------|---------------|------------|\n"
        
        for round_num, round_data in sorted(rounds_data.items(), key=lambda x: int(x[0])):
            models_data = round_data.get("models", {})
            
            for model, model_data in sorted(models_data.items()):
                code_size = model_data.get("code_size_kb", "N/A")
                code_lines = model_data.get("code_lines", "N/A")
                
                report += f"| {round_num} | {model} | {code_size} | {code_lines} |\n"
    
    # Save the report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Report saved to: {output_file}")

def create_visualizations(metrics: Dict[str, Any], output_dir: str) -> None:
    \"\"\"Create visualizations of the metrics\"\"\"
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualizations.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    tournament_metrics = metrics.get("tournament", {})
    test_metrics = metrics.get("test", [])
    
    # Extract data for plotting
    rounds_data = tournament_metrics.get("rounds", {})
    if not rounds_data:
        return
    
    # Create a plot of code size by round and model
    plt.figure(figsize=(10, 6))
    
    # Extract data
    models = set()
    round_nums = []
    data_by_model = {}
    
    for round_num, round_data in sorted(rounds_data.items(), key=lambda x: int(x[0])):
        round_nums.append(int(round_num))
        models_data = round_data.get("models", {})
        
        for model, model_data in models_data.items():
            models.add(model)
            if model not in data_by_model:
                data_by_model[model] = []
            
            code_size = model_data.get("code_size_kb", 0)
            data_by_model[model].append(code_size)
    
    # Plot code size by round for each model
    for model, sizes in data_by_model.items():
        # Ensure all models have data for all rounds
        while len(sizes) < len(round_nums):
            sizes.append(None)
        
        plt.plot(round_nums, sizes, marker='o', label=model)
    
    plt.xlabel('Round')
    plt.ylabel('Code Size (KB)')
    plt.title('Code Size by Round and Model')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(output_dir / "code_size_by_round.png")
    
    # Create a plot of execution time by round and model
    if test_metrics:
        plt.figure(figsize=(10, 6))
        
        # Extract data
        data_by_model = {}
        
        for metric in test_metrics:
            model = metric.get("model", "Unknown")
            round_num = int(metric.get("round", 0))
            time = metric.get("execution_time", 0)
            
            if model not in data_by_model:
                data_by_model[model] = []
            
            # Ensure the list is long enough
            while len(data_by_model[model]) <= round_num:
                data_by_model[model].append(None)
            
            data_by_model[model][round_num] = time
        
        # Plot execution time by round for each model
        for model, times in data_by_model.items():
            rounds = list(range(len(times)))
            plt.plot(rounds, times, marker='o', label=model)
        
        plt.xlabel('Round')
        plt.ylabel('Execution Time (s)')
        plt.title('Execution Time by Round and Model')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(output_dir / "execution_time_by_round.png")
    
    print(f"Visualizations saved to: {output_dir}")

def main():
    \"\"\"Main function\"\"\"
    parser = argparse.ArgumentParser(description="Analyze LLM tournament results")
    parser.add_argument("--metrics-dir", type=str, default="metrics",
                     help="Directory containing metrics files")
    parser.add_argument("--output-dir", type=str, default="analysis",
                     help="Directory for output files")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load metrics
    metrics = load_metrics(args.metrics_dir)
    
    # Generate markdown report
    report_path = output_dir / "tournament_results_report.md"
    generate_markdown_report(metrics, report_path)
    
    # Create visualizations
    create_visualizations(metrics, output_dir)

if __name__ == "__main__":
    main()
"""
        
        # Save the analyzer script
        analyzer_path = self.output_dir / "analyze_results.py"
        with open(analyzer_path, "w", encoding="utf-8") as f:
            f.write(analyzer_script)
        
        # Make the script executable
        os.chmod(analyzer_path, 0o755)
        
        logger.info(f"Created results analyzer: {analyzer_path}")
        return analyzer_path

    def run_tournament(self) -> Dict[str, List[ModelResponse]]:
        """
        Run the complete tournament through all rounds.
        
        Returns:
            Dictionary mapping model names to lists of their responses for each round
        """
        # Initialize metrics
        self.metrics["start_time"] = datetime.now().isoformat()
        self.metrics["models"] = list(self.models.keys())
        self.metrics["rounds"] = {}
        
        # Run round 0 (initial solutions)
        logger.info("Starting tournament with round 0 (initial solutions)")
        self.run_round(0)
        
        # Run subsequent rounds
        for round_num in range(1, self.rounds + 1):
            logger.info(f"Starting round {round_num}")
            self.run_round(round_num)
            
            # Give a short break between rounds to avoid rate limiting
            if round_num < self.rounds:
                logger.info(f"Completed round {round_num}. Waiting before starting next round...")
                time.sleep(10)
        
        # Record end time
        self.metrics["end_time"] = datetime.now().isoformat()
        
        # Create final artifacts
        logger.info("Tournament completed. Generating final artifacts...")
        
        # Save final metrics
        self.save_metrics()
        
        # Generate metrics report
        self.generate_metrics_report()
        
        # Create test suite
        self.create_test_suite()
        
        # Create results analyzer
        self.create_results_analyzer()
        
        # Organize responses by model
        results = {model_name: [] for model_name in self.models.keys()}
        
        for round_num in range(self.rounds + 1):
            for model_name, response in self.responses.get(round_num, {}).items():
                results[model_name].append(response)
        
        logger.info(f"Tournament completed successfully with {self.rounds + 1} rounds")
        
        return results

    def run_tests(self, test_file: Optional[str] = None) -> None:
        """
        Run tests on all generated solutions.
        
        Args:
            test_file: Path to the file to test on (defaults to self.test_file)
        """
        if not test_file and not self.test_file:
            logger.warning("No test file provided. Skipping tests.")
            return
        
        test_file = test_file or self.test_file
        
        # Ensure the test file exists
        if not os.path.exists(test_file):
            logger.error(f"Test file not found: {test_file}")
            return
        
        # Get the test runner script
        test_runner = self.output_dir / "run_tests.py"
        
        if not test_runner.exists():
            logger.warning("Test runner not found. Creating test suite...")
            self.create_test_suite()
        
        # Run the tests
        logger.info(f"Running tests on {test_file}...")
        
        try:
            cmd = ["python", str(test_runner), "--test-file", test_file]
            subprocess.run(cmd, check=True)
            logger.info("Tests completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running tests: {str(e)}")
    
    def analyze_results(self) -> None:
        """Run the results analyzer to generate reports and visualizations"""
        # Get the analyzer script
        analyzer = self.output_dir / "analyze_results.py"
        
        if not analyzer.exists():
            logger.warning("Results analyzer not found. Creating analyzer...")
            self.create_results_analyzer()
        
        # Run the analyzer
        logger.info("Analyzing tournament results...")
        
        try:
            cmd = [
                "python", 
                str(analyzer), 
                "--metrics-dir", str(self.output_dir / "metrics"),
                "--output-dir", str(self.output_dir / "analysis")
            ]
            subprocess.run(cmd, check=True)
            logger.info("Analysis completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error analyzing results: {str(e)}")


def main():
    """
    Main function to run the LLM tournament from the command line.
    
    This function parses command-line arguments, initializes the tournament,
    and runs it with the specified options.
    """
    parser = argparse.ArgumentParser(
        description="Run a multi-round LLM coding tournament",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="File containing the coding challenge prompt"
    )
    
    # Optional arguments
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=5,
        help="Number of tournament rounds (not including round 0)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="tournament_results",
        help="Directory for tournament results"
    )
    parser.add_argument(
        "--test-file", 
        type=str,
        help="File to use for testing solutions"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for LLM generation (0.0-1.0)"
    )
    parser.add_argument(
        "--concurrent-requests", 
        type=int, 
        default=2,
        help="Maximum number of concurrent API requests"
    )
    parser.add_argument(
        "--skip-tests", 
        action="store_true",
        help="Skip running tests on the solutions"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.prompt):
        parser.error(f"Prompt file not found: {args.prompt}")
    
    if args.test_file and not os.path.exists(args.test_file):
        parser.error(f"Test file not found: {args.test_file}")
    
    if args.rounds < 1:
        parser.error("Number of rounds must be at least 1")
    
    if args.temperature < 0.0 or args.temperature > 1.0:
        parser.error("Temperature must be between 0.0 and 1.0")
    
    # Load the prompt from file
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    
    # Display startup banner
    print(r"""
 _     _     __  __   _____                                                  _   
| |   | |   |  \/  | |_   _|                                                | |  
| |   | |   | \  / |   | |     ___    _   _   _ __   _ __     __ _   _ __   | |_ 
| |   | |   | |\/| |   | |    / _ \  | | | | | '__| | '_ \   / _` | | '_ \  | __|
| |___| |___| |  | |   | |   | (_) | | |_| | | |    | | | | | (_| | | | | | | |_ 
|_____|_____|_|  |_|   |_|    \___/   \__,_| |_|    |_| |_|  \__,_| |_| |_|  \__|
                                                                                 
""")
    print(f"Starting LLM Tournament with {len(MODELS)} models for {args.rounds} rounds")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {', '.join(MODELS.keys())}")
    print()
    
    # Initialize the tournament
    tournament = LLMTournament(
        prompt=prompt_text,
        rounds=args.rounds,
        output_dir=args.output_dir,
        temperature=args.temperature,
        concurrent_requests=args.concurrent_requests,
        test_file=args.test_file,
        verbose=args.verbose
    )
    
    # Run the tournament
    try:
        tournament.run_tournament()
        
        # Run tests if a test file is provided and tests are not skipped
        if args.test_file and not args.skip_tests:
            tournament.run_tests(args.test_file)
        
        # Analyze results
        tournament.analyze_results()
        
        # Print final message
        print("\nTournament completed successfully!")
        print(f"Results are available in: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nTournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running tournament: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()