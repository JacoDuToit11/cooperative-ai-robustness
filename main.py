import argparse
import os
import traceback
import math
import time
import matplotlib.pyplot as plt
import numpy as np # Import numpy for Gini
from typing import List, Dict
from fishing_environment import ResourceEnvironment # Renamed import
from fishing_agent import ResourceAgent # Renamed import

def calculate_gini(payoffs: List[float]) -> float:
    """Calculates the Gini coefficient for a list of payoffs."""
    if not payoffs or sum(payoffs) == 0:
        return 0.0 # Gini is 0 if no payoffs or all are zero
    payoffs = np.sort(np.array(payoffs))
    n = len(payoffs)
    index = np.arange(1, n + 1)
    # Gini formula: (sum( (2*i - n - 1) * x_i )) / (n * sum(x_i))
    gini = (np.sum((2 * index - n - 1) * payoffs)) / (n * np.sum(payoffs))
    return gini

def generate_cooperative_history(
    seed_rounds: int,
    env: ResourceEnvironment,
    agents: List[ResourceAgent],
    num_agents: int,
    cooperation_harvest_fraction: float = 0.1 # Fraction of current resource to target harvesting total
) -> List[Dict]:
    """Generates a history of rounds with a predefined cooperative strategy."""
    print(f"--- Generating {seed_rounds} Seeded Cooperative Rounds ---")
    history = []
    for seeded_round in range(1, seed_rounds + 1):
        state = env.get_state()
        resource_at_start = state['current_resource']
        print(f" Seed Round {seeded_round}: Start Resource={resource_at_start:.2f}")

        if resource_at_start <= 1e-6: # Use tolerance for float comparison
            print(" Seed Round: Resource depleted during seeding.")
            break

        # Cooperative strategy: Target a small fraction of the current resource, divided equally
        total_cooperative_harvest = resource_at_start * cooperation_harvest_fraction
        target_harvest_per_agent = (total_cooperative_harvest / num_agents) if num_agents > 0 else 0

        # Ensure harvest is within limits
        target_harvest_per_agent = max(0.0, min(target_harvest_per_agent, resource_at_start / num_agents if num_agents > 0 else 0))

        actual_harvests: Dict[int, float] = {}
        total_actually_harvested = 0.0
        for agent in agents:
            harvest_amount = target_harvest_per_agent
            taken = env.take_resource(harvest_amount)
            if not math.isclose(taken, harvest_amount, rel_tol=1e-6):
                 print(f" Seed Round Warning: Agent {agent.agent_id} target {harvest_amount:.2f} != taken {taken:.2f}. Using {taken:.2f}.")
                 harvest_amount = taken
            agent.update_harvest(harvest_amount) # Use updated agent method
            actual_harvests[agent.agent_id] = harvest_amount
            total_actually_harvested += harvest_amount
            print(f" Seed Round Agent {agent.agent_id}: Cooperatively Harvested={harvest_amount:.2f}")

        resource_before_regen = env.get_resource_level()
        print(f" Seed Round: Total Harvested={total_actually_harvested:.2f}, Res Before Regen={resource_before_regen:.2f}")

        # Record history
        history.append({
            'round': seeded_round,
            'resource_at_start': resource_at_start,
            'attempted_harvests': actual_harvests.copy(), # Rename key
            'actual_harvests': actual_harvests.copy(),    # Rename key
            'total_harvested_this_round': total_actually_harvested,
            'resource_before_regen': resource_before_regen
        })

        env.regenerate()
        print(f" Seed Round: Resource After Regen={env.get_resource_level():.2f}")

    print(f"--- End Seeded Rounds ---")
    return history

def run_simulation(
    num_agents: int,
    initial_resource: float,
    resource_limit: float,
    total_rounds: int,
    model: str,
    seed_rounds: int
):
    """Runs the resource harvesting simulation with live plotting and optional seeded history."""

    print("--- Starting Resource Harvesting Simulation ---")
    print(f" Parameters: Agents={num_agents}, Initial Resource={initial_resource:.2f}, Limit={resource_limit:.2f}, Total Rounds={total_rounds}, Seed Rounds={seed_rounds}, Model={model}")

    try:
        env = ResourceEnvironment(initial_resource, resource_limit)
    except ValueError as e:
        print(f"Error initializing environment: {e}")
        return

    agents: List[ResourceAgent] = []
    try:
        for i in range(num_agents):
            agents.append(ResourceAgent(
                agent_id=i + 1,
                model=model
            ))
    except ValueError as e:
        print(f"Error initializing agents: {e}")
        print(" Ensure OPENAI_API_KEY environment variable is set.")
        return
    except Exception as e:
        print(f"Unexpected error initializing agents:")
        traceback.print_exc()
        return

    # Generate Seeded History
    history: List[Dict] = []
    start_round = 1
    if seed_rounds > 0:
        if seed_rounds >= total_rounds:
            print("Warning: Seed rounds >= total rounds. Only seeded rounds will run.")
            seed_rounds = total_rounds
        history = generate_cooperative_history(
            seed_rounds=seed_rounds, env=env, agents=agents,
            num_agents=num_agents
        )
        start_round = seed_rounds + 1

    # Plotting Setup
    plt.ion()
    fig, ax = plt.subplots()
    if history:
        last_seed_round_num = history[-1]['round']
        res_after_last_seed = env.get_resource_level()
        harvest_in_last_seed = history[-1]['total_harvested_this_round']
        round_numbers = [last_seed_round_num]
        resource_levels = [res_after_last_seed]
        total_harvests_this_round = [harvest_in_last_seed]
    else:
        round_numbers = [0]
        resource_levels = [env.get_resource_level()]
        total_harvests_this_round = [0.0]

    line_resource, = ax.plot(round_numbers, resource_levels, marker='o', linestyle='-', color='g', label='Resource Level')
    line_harvest, = ax.plot(round_numbers, total_harvests_this_round, marker='x', linestyle='--', color='r', label='Total Harvest This Round')
    # Add line for resource limit
    ax.axhline(y=env.resource_limit, color='grey', linestyle=':', label=f'Limit ({env.resource_limit:.0f})')
    ax.set_xlabel("Round")
    ax.set_ylabel("Amount")
    ax.set_title("Resource Harvesting Simulation")
    ax.legend()
    ax.grid(True)
    plt.show()

    try:
        # Simulation Loop
        for current_round in range(start_round, total_rounds + 1):
            print(f"\n--- Round {current_round} / {total_rounds} --- ")
            state = env.get_state()
            resource_at_start_of_round = state['current_resource']
            print(f"Start: {resource_at_start_of_round:.2f} resource")

            # Check for termination condition (resource depleted)
            if resource_at_start_of_round <= 1e-6:
                print("Resource depleted. Ending simulation.")
                break

            # 1. Agents choose actions
            attempted_harvests: Dict[int, float] = {}
            total_attempted_harvest = 0.0
            for agent in agents:
                decision = agent.choose_action(state, history, num_agents)
                attempted_harvests[agent.agent_id] = decision
                total_attempted_harvest += decision
            print(f"Attempted total harvest: {total_attempted_harvest:.2f}")

            # 2. Determine actual harvests
            actual_harvests: Dict[int, float] = {}
            available_resource = env.get_resource_level()
            total_actually_harvested = 0.0

            if total_attempted_harvest <= 1e-6:
                pass
            elif available_resource <= 1e-6:
                 print(" No resource available to harvest.")
            elif total_attempted_harvest <= available_resource:
                actual_harvests = attempted_harvests
            else:
                print(f" Attempted harvest ({total_attempted_harvest:.2f}) > available ({available_resource:.2f}). Distributing proportionally.")
                for agent_id, attempted in attempted_harvests.items():
                    proportion = (attempted / total_attempted_harvest) if total_attempted_harvest > 1e-6 else 0
                    actual_harvests[agent_id] = proportion * available_resource # No floor for float

            for agent in agents:
                if agent.agent_id not in actual_harvests:
                    actual_harvests[agent.agent_id] = 0.0
            total_actually_harvested = sum(actual_harvests.values())

            # 3. Update environment and agents
            for agent in agents:
                agent_id = agent.agent_id
                harvest_amount = actual_harvests.get(agent_id, 0.0)
                taken = env.take_resource(harvest_amount)
                # Use tolerance for float comparison
                if not math.isclose(taken, harvest_amount, rel_tol=1e-6):
                    print(f" Warning: Agent {agent_id} allocation {harvest_amount:.2f} != taken {taken:.2f}. Using {taken:.2f}.")
                    harvest_amount = taken
                agent.update_harvest(harvest_amount)
                print(f" Agent {agent_id}: Attempt={attempted_harvests.get(agent_id, 0.0):.2f}, Harvested={harvest_amount:.2f}")

            print(f"Total harvested this round: {total_actually_harvested:.2f}")
            resource_before_regen = env.get_resource_level()
            print(f"Resource before regeneration: {resource_before_regen:.2f}")

            # 4. Record history
            history.append({
                'round': current_round,
                'resource_at_start': resource_at_start_of_round,
                'attempted_harvests': attempted_harvests.copy(),
                'actual_harvests': actual_harvests.copy(),
                'total_harvested_this_round': total_actually_harvested,
                'resource_before_regen': resource_before_regen
            })

            # 5. Regenerate resource
            env.regenerate()
            current_resource_after_regen = env.get_resource_level()
            print(f"Resource after regeneration: {current_resource_after_regen:.2f}")

            # Update Plot
            round_numbers.append(current_round)
            resource_levels.append(current_resource_after_regen)
            total_harvests_this_round.append(total_actually_harvested)

            line_resource.set_data(round_numbers, resource_levels)
            line_harvest.set_data(round_numbers, total_harvests_this_round)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)

    except Exception:
        print(f"\nAn unexpected error occurred during simulation:")
        traceback.print_exc()
    finally:
        # Cleanup and Final Stats
        plt.ioff()
        print("\n--- Simulation Ended --- ")
        final_resource_level = env.get_resource_level()
        print(f"Final Resource Level: {final_resource_level:.2f} / {env.resource_limit:.2f}")

        # Calculate Metrics
        agent_payoffs = [agent.total_harvested for agent in agents]
        total_harvest_overall = sum(agent_payoffs)

        # Sustainability: Final resource relative to limit
        sustainability = (final_resource_level / env.resource_limit) if env.resource_limit > 0 else 0.0
        # Equality: Gini coefficient of payoffs
        equality_gini = calculate_gini(agent_payoffs)
        # Efficiency: Average payoff per agent
        efficiency = (total_harvest_overall / num_agents) if num_agents > 0 else 0.0

        print("\n--- Performance Metrics ---")
        print(f"Sustainability (Final Resource / Limit): {sustainability:.3f}")
        print(f"Equality (1 - Gini Coefficient): {1.0 - equality_gini:.3f} (Gini={equality_gini:.3f}) ")
        print(f"Efficiency (Avg. Payoff per Agent): {efficiency:.3f}")

        print("\nAgent Totals:")
        for agent in agents:
            print(f"- Agent {agent.agent_id}: Harvested {agent.total_harvested:.2f}")
        print(f"Total resource harvested overall: {total_harvest_overall:.2f}")

        print("\nClose the plot window to exit.")
        plt.show()

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a multi-agent resource harvesting simulation (doubling growth model).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num_agents", type=int, default=4, help="Number of agents.")
    parser.add_argument("--initial_resource", type=float, default=100.0, help="Initial amount of resource.")
    parser.add_argument("--resource_limit", type=float, default=100.0, help="Maximum resource limit.")
    parser.add_argument("--rounds", dest='total_rounds', type=int, default=50, help="Maximum number of simulation rounds.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model for agent decisions.")
    parser.add_argument("--seed_rounds", type=int, default=0, help="Number of initial rounds to simulate with a cooperative strategy.")

    args = parser.parse_args()

    if args.num_agents <= 0:
        parser.error("Number of agents must be positive.")
    if args.initial_resource < 0:
        parser.error("Initial resource cannot be negative.")
    if args.resource_limit <= 0:
        parser.error("Resource limit must be positive.")
    if args.total_rounds <= 0:
        parser.error("Total number of rounds must be positive.")
    if args.seed_rounds < 0:
        parser.error("Seed rounds cannot be negative.")

    return args

if __name__ == "__main__":
    import os # Ensure os is imported if needed later, though not strictly necessary now
    import math
    import numpy as np

    args = parse_args()
    run_simulation(
        num_agents=args.num_agents,
        initial_resource=args.initial_resource,
        resource_limit=args.resource_limit,
        total_rounds=args.total_rounds,
        model=args.model,
        seed_rounds=args.seed_rounds
    )
