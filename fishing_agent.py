import os
import openai
import random
import re
import textwrap
from typing import Optional, List, Dict

class FishingAgent:
    """
    Represents an AI agent deciding how many fish to catch via OpenAI API.
    """
    def __init__(self, agent_id: int, max_catch_per_round: int, openai_api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initializes the agent.

        Args:
            agent_id: Unique identifier for the agent.
            max_catch_per_round: Max fish this agent can attempt to catch per round.
            openai_api_key: OpenAI API key. Reads from env var `OPENAI_API_KEY` if None.
            model: The OpenAI model to use for decision making.
        """
        self.agent_id = agent_id
        self.max_catch_per_round = max_catch_per_round
        self.total_caught = 0
        self.model = model

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set.")
        self.client = openai.OpenAI(api_key=api_key)

    def _create_prompt(self, state: Dict, history: List[Dict], num_agents: int) -> str:
        """Creates the prompt for the OpenAI API call based on game state and history."""
        prompt_base = textwrap.dedent(f"""
            You are Agent {self.agent_id} in a fishing simulation with {num_agents} agents total.
            The goal is to maximize your fish caught sustainably. The fish population regenerates each round, but will not regenerate if the population is 0.

            Rules:
            - Current fish available: {state['current_fish']}
            - Fish regenerate at {state['regeneration_rate']:.2f}x the remaining population each round.
            - You can attempt to catch between 0 and {self.max_catch_per_round} fish this round.
            - If total attempted catch > available fish, catches are distributed proportionally.
            - Your total catch so far: {self.total_caught}

        """)

        history_str = "\nThis is the first round.\n"
        if history:
            history_str = "\nRecent Round History (Your Catch / Total Catch / Fish Start of Round):\n"
            for record in reversed(history[-5:]):
                your_catch = record['actual_catches'].get(self.agent_id, 0)
                total_catch = sum(record['actual_catches'].values())
                fish_at_start = record.get('fish_at_start', 'N/A')
                history_str += f"- Round {record.get('round', 'N/A')}: You={your_catch}, Total={total_catch}, StartFish={fish_at_start}\n"
        history_str += "\n"

        instruction = f"Considering the rules, the current state ({state['current_fish']} fish, {state['regeneration_rate']:.2f} rate), and history, how many fish do you attempt to catch this round (0-{self.max_catch_per_round})?\n"
        instruction += "Respond with only a single integer."

        return prompt_base + history_str + instruction

    def choose_action(self, state: Dict, history: List[Dict], num_agents: int) -> int:
        """
        Decides how many fish to catch using the OpenAI API.

        Args:
            state: Current environment state.
            history: List of past round results.
            num_agents: Total number of agents.

        Returns:
            Agent's decision (number of fish to attempt catching).
        """
        prompt = self._create_prompt(state, history, num_agents)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=10
            )
            content = response.choices[0].message.content

            match = re.search(r'\d+', content)
            if match:
                decision = int(match.group(0))
            else:
                 print(f"Agent {self.agent_id} (AI) response '{content}' invalid. Falling back.")
                 raise ValueError("No integer found in response")

            # Clamp decision to valid range [0, max_catch_per_round]
            decision = max(0, min(decision, self.max_catch_per_round))
            print(f"Agent {self.agent_id} (AI) decided: {decision}")
            return decision

        except (openai.APIError, ValueError, IndexError) as e:
            print(f"Agent {self.agent_id} decision error: {e}. Falling back to random.")
            # Fallback: random choice within constraints
            max_possible_catch = state.get('current_fish', self.max_catch_per_round)
            fallback_decision = random.randint(0, min(self.max_catch_per_round, max_possible_catch))
            print(f"Agent {self.agent_id} (Fallback) decided: {fallback_decision}")
            return fallback_decision
        except Exception as e:
             # Catch unexpected errors
             print(f"Agent {self.agent_id} unexpected error: {e}. Falling back to random.")
             max_possible_catch = state.get('current_fish', self.max_catch_per_round)
             fallback_decision = random.randint(0, min(self.max_catch_per_round, max_possible_catch))
             print(f"Agent {self.agent_id} (Fallback) decided: {fallback_decision}")
             return fallback_decision

    def update_catch(self, amount_caught: int):
        """Updates the agent's total fish caught."""
        if amount_caught < 0:
             print(f"Warning: Agent {self.agent_id} received negative catch amount: {amount_caught}")
             return # Avoid reducing total for negative values
        self.total_caught += amount_caught 