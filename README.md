# Cooperative AI Fishing Simulation

This project simulates a common resource sharing problem using multiple AI agents powered by the OpenAI API. The scenario involves agents deciding how many fish to catch from a shared resource that regenerates at a fixed rate.

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

4.  **Set up OpenAI API Key:**
    Set the `OPENAI_API_KEY` environment variable:
    ```bash
    export OPENAI_API_KEY='your-api-key'
    ```
    Alternatively, you can modify the code to load the key from a `.env` file or other configuration method.

## Running the Simulation

Execute the main script:
```bash
python main.py --num_agents 3 --initial_fish 100 --regeneration_rate 1.1 --rounds 10
```

### Configuration Options:

*   `--num_agents`: Number of AI agents participating (default: 2).
*   `--initial_fish`: Starting number of fish in the environment (default: 100).
*   `--regeneration_rate`: Factor by which the fish population multiplies each round (default: 1.1, i.e., 10% growth).
*   `--rounds`: Number of rounds the simulation will run (default: 10).
*   `--max_catch_per_agent`: Maximum number of fish an agent can attempt to catch in a single round (default: 20). 