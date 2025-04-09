import os
import openai
import random
from typing import Optional, List, Dict

class FishingAgent:
    """
    Represents an AI agent that decides how many fish to catch using the OpenAI API.
    """
    def __init__(self, agent_id: int, max_catch_per_round: int, openai_api_key: Optional[str] = None):
        """
        Initializes the fishing agent.

        Args:
            agent_id: A unique identifier for the agent.
            max_catch_per_round: The maximum number of fish this agent can attempt to catch in a single round.
            openai_api_key: The OpenAI API key. If None, attempts to read from OPENAI_API_KEY environment variable.
        """
        self.agent_id = agent_id
        self.max_catch_per_round = max_catch_per_round
        self.total_caught = 0

        # Initialize OpenAI client
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set.")
        self.client = openai.OpenAI(api_key=api_key)

    def _create_prompt(self, state: Dict, history: List[Dict], num_agents: int) -> str:
        """Creates the prompt for the OpenAI API call."""
        prompt = f"You are Agent {self.agent_id} in a fishing simulation with {num_agents} agents total.\n"
        prompt += "The goal is to catch fish from a shared pool. The fish population regenerates each round.\n"
        prompt += "Rules:\n"
        prompt += f"- Current fish available: {state['current_fish']}\n"
        prompt += f"- Fish regenerate at a rate of {state['regeneration_rate']:.2f}x the remaining population each round.\n"
        prompt += f"- You can attempt to catch a maximum of {self.max_catch_per_round} fish this round.\n"
        prompt += f"- If the total attempted catch by all agents exceeds the available fish, fish are distributed proportionally, or potentially zero if the pool is empty.\n"
        prompt += f"- You have caught {self.total_caught} fish in total so far.\n\n"

        if history:
            prompt += "Recent Round History (Your Catch / Total Catch / Fish Start of Round):\n"
            # Show last few rounds, e.g., last 5
            for i, record in enumerate(reversed(history[-5:])):
                # Use 'actual_catches' key as stored in main.py history
                your_catch = record['actual_catches'].get(self.agent_id, 0)
                total_catch = sum(record['actual_catches'].values())
                # Use .get for fish_at_start as well for robustness
                fish_at_start = record.get('fish_at_start', 'N/A')
                prompt += f"- Round {record.get('round', 'N/A')}: You caught {your_catch}, Total caught: {total_catch}, Fish at start: {fish_at_start}\n"
            prompt += "\n"
        else:
            prompt += "This is the first round.\n\n"

        prompt += f"Considering the current fish count ({state['current_fish']}), the regeneration rate ({state['regeneration_rate']:.2f}), the maximum you can catch ({self.max_catch_per_round}), and the history, how many fish do you choose to attempt to catch this round?\n"
        prompt += f"Provide only a single integer number between 0 and {self.max_catch_per_round}."
        return prompt

    def choose_action(self, state: Dict, history: List[Dict], num_agents: int) -> int:
        """
        Uses the OpenAI API to decide how many fish to catch.

        Args:
            state: The current state of the fishing environment.
            history: A list of dictionaries, each representing a past round's results.
            num_agents: The total number of agents in the simulation.

        Returns:
            The number of fish the agent decides to attempt to catch.
        """
        prompt = self._create_prompt(state, history, num_agents)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", # Or a different model like gpt-4
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, # Add some variability but keep it somewhat predictable
                max_tokens=10
            )
            content = response.choices[0].message.content
            # Try to extract the integer
            # More robust parsing: find the first integer in the response
            import re
            match = re.search(r'\d+', content)
            if match:
                decision = int(match.group(0))
            else:
                 print(f"Agent {self.agent_id} (AI) response did not contain a number: '{content}'. Falling back.")
                 raise ValueError("No integer found in response")

            # Clamp the decision to the valid range
            decision = max(0, min(decision, self.max_catch_per_round))
            print(f"Agent {self.agent_id} (AI) decided to catch: {decision}")
            return decision
        except (openai.APIError, ValueError, IndexError) as e:
            print(f"Error getting decision for Agent {self.agent_id}: {e}. Falling back to random choice.")
            # Fallback strategy: catch a random amount within available fish/max catch constraints
            fallback_decision = random.randint(0, min(self.max_catch_per_round, state.get('current_fish', self.max_catch_per_round)))
            print(f"Agent {self.agent_id} (Fallback) decided to catch: {fallback_decision}")
            return fallback_decision
        except Exception as e:
             print(f"An unexpected error occurred for Agent {self.agent_id}: {e}. Falling back to random choice.")
             fallback_decision = random.randint(0, min(self.max_catch_per_round, state.get('current_fish', self.max_catch_per_round)))
             print(f"Agent {self.agent_id} (Fallback) decided to catch: {fallback_decision}")
             return fallback_decision

    def update_catch(self, amount_caught: int):
        """Updates the agent's total fish caught."""
        self.total_caught += amount_caught 