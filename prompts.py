import textwrap
from typing import List, Dict

def get_prompt_text(scenario_name: str, agent_id: int, state: Dict, history: List[Dict], num_agents: int, total_payoff: float) -> str:
    """Generates the appropriate prompt text based on the scenario name."""

    if scenario_name == "fishing":
        return _get_fishing_prompt(agent_id, state, history, num_agents, total_payoff)
    # Add elif for other scenarios here in the future
    # elif scenario_name == "some_other_game":
    #     return _get_other_game_prompt(...)
    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}")

def _get_fishing_prompt(agent_id: int, state: Dict, history: List[Dict], num_agents: int, total_payoff: float) -> str:
    """Generates the prompt for the specific fishing scenario with doubling resource."""
    current_resource = state['current_resource']
    resource_limit = state['resource_limit']
    # Assuming critical_threshold is passed in state if needed, or defined globally/via config
    # For now, let's hardcode it based on the default in main.py or get it from state if available
    critical_threshold = state.get('critical_threshold', 5.0)
    other_agents_count = num_agents - 1

    # Narrative Prompt Setup using the user-provided style
    # Using "tons" for fish resource
    prompt_narrative = textwrap.dedent(f"""
        You are Agent {agent_id}, a fisherman, and you fish each round in a lake along with {other_agents_count} other fishermen (total {num_agents}).
        The lake has a maximum capacity of {resource_limit:.2f} tons of fish.
        Each round, each fisherman decides how many tons of fish to attempt to take out of the lake (must be >= 0.00 tons).

        At the end of the round, the remaining fish reproduce and double (R_next = R_after_harvest * 2), but the population cannot exceed the capacity of {resource_limit:.2f} tons.
        So, if 60 tons remain, there will be {min(60*2, resource_limit):.2f} tons after reproduction, or if 20 tons remain, there will be {min(20*2, resource_limit):.2f} tons.

        If the total amount all fishermen attempt to harvest exceeds the currently available fish, the available fish are distributed proportionally based on how much each fisherman requested. For example, if Fisherman A requests 4.0 tons and Fisherman B requests 6.0 tons (total 10.0 tons), but only 5.0 tons are available, Fisherman A gets (4.0/10.0)*5.0 = 2.0 tons and Fisherman B gets (6.0/10.0)*5.0 = 3.0 tons.

        CRITICAL RULE: If the fish population at the start of a round is below {critical_threshold:.2f} tons, the lake is considered critically endangered, and **no fishing is allowed in that round or any subsequent rounds**. The simulation ends.

        Your goal is to maximize the total fish caught over the long run.
        Your total fish caught so far: {total_payoff:.2f} tons.
        If the fish population drops to 0, it cannot regenerate, and no more fish can be caught in subsequent rounds (this also happens if it drops below {critical_threshold:.2f} tons).
    """)

    # History section
    history_str = "\nThis is the first round.\n"
    if history:
        history_str = "\nRecent Round History (Your Harvest / Total Harvest / Fish @ Start):\n"
        for record in reversed(history[-5:]):
            your_harvest = record['actual_harvests'].get(agent_id, 0.0)
            total_harvest = sum(record['actual_harvests'].values())
            resource_at_start = record.get('resource_at_start', 'N/A')
            resource_start_str = f"{resource_at_start:.2f}" if isinstance(resource_at_start, float) else str(resource_at_start)
            history_str += f"- Round {record.get('round', 'N/A')}: You={your_harvest:.2f}, Total={total_harvest:.2f}, StartFish={resource_start_str}\n"
    history_str += "\n"

    # Final instruction
    instruction = f"Current Fish Available: {current_resource:.2f} tons. CRITICAL THRESHOLD: {critical_threshold:.2f} tons. Based on the rules (especially the critical threshold), your goal, and the history, how many tons of fish do you attempt to harvest this round (must be >= 0.00)?\n"
    instruction += "Respond with only a single non-negative floating-point number."

    return prompt_narrative + history_str + instruction