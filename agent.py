import os
import openai
import random
import re
from typing import Optional, List, Dict

# Import the prompt generation function
from prompts import get_prompt_text

class Agent:
    """
    Represents a generic AI agent using OpenAI API for decision making in simulations.
    """
    def __init__(self, agent_id: int, scenario_name: str, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initializes the agent.

        Args:
            agent_id: Unique identifier for the agent.
            scenario_name: Identifier for the game scenario (used for prompt selection).
            openai_api_key: OpenAI API key. Reads from env var `OPENAI_API_KEY` if None.
            model: The OpenAI model to use for decision making.
        """
        self.agent_id = agent_id
        self.scenario_name = scenario_name
        self.total_payoff = 0.0
        self.model = model

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set.")
        self.client = openai.OpenAI(api_key=api_key)

    def choose_action(self, state: Dict, history: List[Dict], num_agents: int) -> Dict:
        """
        Decides the action (e.g., amount to harvest) using the OpenAI API based on the scenario prompt.

        Args:
            state: Current environment state.
            history: List of past round results.
            num_agents: Total number of agents.

        Returns:
            Dict containing:
                'decision': Agent's decision (float, e.g., amount to harvest, >= 0).
                'prompt': The full prompt text sent to the API (str).
                'response_content': The raw response content from the API (str).
        """
        prompt = get_prompt_text(
            scenario_name=self.scenario_name,
            agent_id=self.agent_id,
            state=state,
            history=history,
            num_agents=num_agents,
            total_payoff=self.total_payoff
        )

        response_content = "<API_CALL_FAILED>"
        decision = 0.0 # Default decision on error

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=15
            )
            response_content = response.choices[0].message.content

            match = re.search(r'[-+]?([0-9]*[.])?[0-9]+', response_content)
            if match:
                decision = float(match.group(0))
            else:
                 print(f"Agent {self.agent_id} (AI) response '{response_content}' invalid (no float). Falling back.")
                 # Keep default decision 0.0 but flag content as invalid
                 response_content += " <INVALID_FLOAT_PARSE>"
                 # Fallback logic moved outside the try block to ensure it uses response_content

            decision = max(0.0, decision)
            print(f"Agent {self.agent_id} (AI) proposes action: {decision:.2f}")
            # Return dict including prompt and response
            return {"decision": decision, "prompt": prompt, "response_content": response_content}

        except (openai.APIError, ValueError, IndexError) as e:
            print(f"Agent {self.agent_id} decision error: {e}. Falling back to random action.")
        except Exception as e:
             print(f"Agent {self.agent_id} unexpected error: {e}. Falling back to random action.")

        # Fallback logic if try block failed or parsing failed previously
        fallback_decision = 0.0
        if self.scenario_name == "fishing":
            max_possible_action = state.get('current_resource', 0.0)
            fallback_decision = random.uniform(0.0, max_possible_action)
        print(f"Agent {self.agent_id} (Fallback) proposes action: {fallback_decision:.2f}")
        # Return dict even on fallback, indicating failure in content
        return {"decision": fallback_decision, "prompt": prompt, "response_content": response_content}

    def update_payoff(self, payoff_increment: float):
        """Updates the agent's total accumulated payoff."""
        if payoff_increment < 0:
             print(f"Warning: Agent {self.agent_id} received negative payoff increment: {payoff_increment:.2f}")
        self.total_payoff += payoff_increment