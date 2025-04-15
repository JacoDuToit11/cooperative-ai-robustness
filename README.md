# Cooperative AI Resource Harvesting Simulation

This project simulates a multi-agent common-pool resource harvesting problem, inspired by scenarios like fishing or forestry management. AI agents, powered by the OpenAI API, decide how much resource to harvest each round from a shared environment.

The current default scenario models a resource that doubles each round after harvesting, up to a fixed limit.

## Features

*   Configurable number of agents, initial resource, resource limit, and simulation rounds.
*   AI agents using specified OpenAI models (e.g., `gpt-4o-mini`) for decision-making.
*   Scenario-specific prompts loaded from `prompts.py` (currently implements a "fishing" scenario).
*   Resource regeneration model: Doubling with a cap (defined in `environment.py`).
*   Optional "cooperative seeding" rounds to establish a baseline history before AI takes over.
*   Live plotting of resource level and total harvest per round using `matplotlib`.
*   Calculation and reporting of performance metrics:
    *   **Sustainability:** Final resource level relative to the limit.
    *   **Equality:** How evenly the harvest is distributed (1 - Gini Coefficient).
    *   **Efficiency:** Average harvest per agent.

## Project Structure

```
├── main.py             # Main simulation script, argument parsing
├── agent.py            # Defines the generic Agent class (using OpenAI API)
├── environment.py      # Defines the ResourceEnvironment class (manages resource dynamics)
├── prompts.py          # Defines prompt generation logic for different scenarios
├── requirements.txt    # Project dependencies
├── README.md           # This file
└── ... (other potential files like .gitignore)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Requires `openai`, `matplotlib`, `numpy`.* 

4.  **Set up OpenAI API Key:**
    Set the `OPENAI_API_KEY` environment variable:
    ```bash
    export OPENAI_API_KEY='your-api-key'
    ```
    Alternatively, modify `agent.py` to load the key from a different source.

## Running the Simulation

Execute the main script from your terminal:

```bash
python main.py [OPTIONS]
```

### Example Usage:

*   Run with default settings (4 agents, 100 initial/limit, 50 rounds, fishing scenario):
    ```bash
    python main.py
    ```
*   Run with 3 agents for 30 rounds, starting with 50 resource units:
    ```bash
    python main.py --num_agents 3 --rounds 30 --initial_resource 50
    ```
*   Run with 5 cooperative seed rounds before the 50 main rounds:
    ```bash
    python main.py --rounds 50 --seed_rounds 5
    ```
*   Use a different OpenAI model:
    ```bash
    python main.py --model gpt-4o
    ```

### Configuration Options:

*   `--scenario`: Name of the scenario to run (default: `fishing`). Determines prompts.
*   `--num_agents`: Number of AI agents participating (default: 4).
*   `--initial_resource`: Starting amount of the resource (default: 100.0).
*   `--resource_limit`: Maximum resource limit (default: 100.0).
*   `--critical_threshold`: Resource level below which the simulation ends (default: 5.0).
*   `--rounds`: Maximum number of simulation rounds (default: 50).
*   `--model`: OpenAI model for agent decisions (default: `gpt-4o-mini`).
*   `--seed_rounds`: Number of initial rounds to simulate with an optimal sustainable strategy (default: 0).

## Output

The simulation will:
*   Print round-by-round details to the console (resource levels, agent actions, harvests).
*   Display a live plot window showing the resource level and total harvest over time.
*   Print final agent totals and the calculated Sustainability, Equality, and Efficiency metrics at the end. 