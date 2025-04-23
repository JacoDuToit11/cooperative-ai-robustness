# Cooperative AI Resource Harvesting Simulation

This project simulates a multi-agent common-pool resource harvesting problem, inspired by scenarios like fishing or forestry management. AI agents, powered by the OpenAI API, decide how much resource to harvest each round from a shared environment.

The current default scenario models a resource that doubles each round after harvesting, up to a fixed limit.

## Features

*   **Configuration via JSON file.**
*   Configurable number of agents, initial resource, resource limit, and simulation rounds.
*   AI agents using specified OpenAI models (e.g., `gpt-4o-mini`) for decision-making.
*   Scenario-specific prompts loaded from `prompts.py`.
*   Resource regeneration model (defined in `environment.py`).
*   Optional "cooperative seeding" rounds.
*   Ability to run multiple simulations with the same parameters and aggregate results.
*   Optional disabling of agent conversation log saving.
*   Calculation and reporting of performance metrics (Sustainability, Equality, Efficiency) for individual runs and averaged across multiple runs.
*   Generation of an aggregated plot showing average resource level and total harvest over time across multiple runs.
*   Generation of individual plots for each run.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Set up OpenAI API Key:**
    Set the `OPENAI_API_KEY` environment variable:
    ```bash
    export OPENAI_API_KEY='your-api-key'
    ```
    Alternatively, modify `agent.py` to load the key from a different source.

3.  **(Optional) Modify Configuration:**
    Edit the `config.json` file in the project root directory to set your desired simulation parameters.

## Running the Simulation

Execute the main script from your terminal, specifying the configuration file:

```bash
python main.py --config config.json
```

Or, if you create a different configuration file (e.g., `configs/scenario_a.json`):

```bash
python main.py --config configs/scenario_a.json
```

### Configuration File (`config.json`)

The simulation parameters are controlled by a JSON file (default: `config.json`). Here is an example structure with explanations:

```json
{
  "run_name": "default_config_run",  // Optional name prefix for the results directory
  "results_base_dir": "results",    // Base directory where run results will be saved
  "num_runs": 1,                   // Number of times to run the simulation with these parameters
  "no_save_conversations": false,  // If true, agent_*.jsonl files will NOT be saved (default: false -> conversations ARE saved)
  "scenario": "fishing",           // Scenario name (determines prompts)
  "num_agents": 4,                 // Number of agents
  "initial_resource": 100.0,       // Starting resource amount
  "resource_limit": 100.0,         // Maximum resource capacity
  "critical_threshold": 5.0,       // Resource level below which simulation ends
  "total_rounds": 50,              // Maximum simulation rounds per run
  "model": "gpt-4o-mini",          // OpenAI model for agent decisions
  "seed_rounds": 0,                // Number of initial cooperative rounds
  "shock_probability": 0.0,        // Probability (0.0-1.0) of a resource shock per round
  "shock_magnitude": 0.2,          // Fraction (0.0-1.0) resource reduction during a shock
  "change_regen_round": 0,         // Round AFTER which regen factor changes (0=no change)
  "new_regen_factor": 1.5,         // New regen factor if change_regen_round > 0
  "random_seed": 42                // (Optional) Integer seed for RNG (shocks). Set to null or omit for non-deterministic runs.
}
```

## Output

The simulation will:
*   Create a main timestamped directory for the set of runs inside the configured `results_base_dir`.
*   Inside the main directory, save:
    *   `master_parameters.json`: The configuration loaded from the JSON file, plus the path to the config file used.
    *   `aggregated_summary.json`: Contains the average metrics and standard deviations calculated across all successful runs (if `num_runs` > 1).
    *   `aggregated_plot.png`: A plot showing the average resource level and average total harvest per round across all successful runs (if `num_runs` > 1).
*   Inside the main directory, create subdirectories for each individual run: `run_1/`, `run_2/`, etc.
*   Inside each `run_X/` directory, save:
    *   `game_log.txt`: Detailed log of that specific simulation run.
    *   `parameters.json`: The configuration parameters used for that specific run (mostly mirrors master, includes run index).
    *   `summary.json`: Final results and metrics for that specific run.
    *   `history.json`: Detailed state changes and actions for every round in that specific run.
    *   `simulation_plot.png`: A plot showing resource level and total harvest over time for this specific run.
    *   `agent_{id}_conversation.jsonl`: (Optional) If `no_save_conversations` in the config is `false`, JSON Lines files containing the prompts and responses for each agent in that run.
*   Print progress and final aggregated metrics (if applicable) to the console. 