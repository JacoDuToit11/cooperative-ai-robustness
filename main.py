import argparse
import traceback
import math
import time
import matplotlib.pyplot as plt
from typing import List, Dict
from fishing_environment import FishingEnvironment
from fishing_agent import FishingAgent

def run_simulation(
    num_agents: int,
    initial_fish: int,
    regeneration_rate: float,
    rounds: int,
    max_catch_per_agent: int,
    model: str
):
    """Runs the fishing simulation with live plotting."""

    print("--- Starting Fishing Simulation ---")
    print(f" Parameters: Agents={num_agents}, Initial Fish={initial_fish}, Regen Rate={regeneration_rate}, Rounds={rounds}, Max Catch/Agent={max_catch_per_agent}, Model={model}")

    try:
        env = FishingEnvironment(initial_fish, regeneration_rate)
    except ValueError as e:
        print(f"Error initializing environment: {e}")
        return

    agents: List[FishingAgent] = []
    try:
        for i in range(num_agents):
            agents.append(FishingAgent(
                agent_id=i + 1,
                max_catch_per_round=max_catch_per_agent,
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

    history: List[Dict] = []

    # --- Plotting Setup ---
    plt.ion() # Interactive mode ON
    fig, ax = plt.subplots()
    round_numbers = [0]
    fish_counts = [env.get_fish_count()]
    total_catches_this_round = [0]

    line_fish, = ax.plot(round_numbers, fish_counts, marker='o', linestyle='-', color='b', label='Fish Population')
    line_catch, = ax.plot(round_numbers, total_catches_this_round, marker='x', linestyle='--', color='r', label='Total Catch This Round')
    ax.set_xlabel("Round")
    ax.set_ylabel("Count")
    ax.set_title("Fishing Simulation Live View")
    ax.legend()
    ax.grid(True)
    plt.show()
    # --- End Plotting Setup ---

    try:
        # --- Simulation Loop ---
        for current_round in range(1, rounds + 1):
            print(f"\n--- Round {current_round} / {rounds} --- ")
            state = env.get_state()
            fish_at_start_of_round = state['current_fish']
            print(f"Start: {fish_at_start_of_round} fish")

            if fish_at_start_of_round <= 0:
                print("Resource depleted. Ending simulation.")
                break

            # 1. Agents choose actions
            attempted_catches: Dict[int, int] = {}
            total_attempted_catch = 0
            for agent in agents:
                decision = agent.choose_action(state, history, num_agents)
                attempted_catches[agent.agent_id] = decision
                total_attempted_catch += decision
            print(f"Attempted total catch: {total_attempted_catch}")

            # 2. Determine actual catches based on availability
            actual_catches: Dict[int, int] = {}
            available_fish = env.get_fish_count()
            total_actually_caught = 0

            if total_attempted_catch <= 0:
                pass # No catches attempted
            elif available_fish <= 0:
                 print(" No fish available to catch.")
            elif total_attempted_catch <= available_fish:
                # Sufficient fish
                actual_catches = attempted_catches
            else:
                # Insufficient fish: distribute proportionally
                print(f" Attempted catch ({total_attempted_catch}) > available ({available_fish}). Distributing proportionally.")
                for agent_id, attempted in attempted_catches.items():
                    # Ensure division by zero is avoided if total_attempted_catch is unexpectedly 0 here
                    proportion = (attempted / total_attempted_catch) if total_attempted_catch > 0 else 0
                    actual_catches[agent_id] = math.floor(proportion * available_fish)

            # Ensure actual_catches dict is fully populated for agents who attempted 0
            for agent in agents:
                if agent.agent_id not in actual_catches:
                    actual_catches[agent.agent_id] = 0

            # Calculate total actually caught for this round
            total_actually_caught = sum(actual_catches.values())

            # 3. Update environment (take fish) and agents (update totals)
            for agent in agents:
                agent_id = agent.agent_id
                caught_amount = actual_catches.get(agent_id, 0)
                taken = env.take_fish(caught_amount)
                if taken != caught_amount:
                    # This warning indicates a potential logic error in allocation vs taking
                    print(f" Warning: Agent {agent_id} allocation {caught_amount} != taken {taken}. Using {taken}.")
                    caught_amount = taken

                agent.update_catch(caught_amount)
                print(f" Agent {agent_id}: Attempt={attempted_catches.get(agent_id, 0)}, Caught={caught_amount}")

            print(f"Total caught this round: {total_actually_caught}")
            fish_before_regen = env.get_fish_count()
            print(f"Fish before regeneration: {fish_before_regen}")

            # 4. Record history
            history.append({
                'round': current_round,
                'fish_at_start': fish_at_start_of_round,
                'attempted_catches': attempted_catches.copy(),
                'actual_catches': actual_catches.copy(),
                'total_caught_this_round': total_actually_caught,
                'fish_before_regen': fish_before_regen
            })

            # 5. Regenerate resource
            env.regenerate()
            current_fish_after_regen = env.get_fish_count()
            print(f"Fish after regeneration: {current_fish_after_regen}")

            # --- Update Plot ---
            round_numbers.append(current_round)
            fish_counts.append(current_fish_after_regen)
            total_catches_this_round.append(total_actually_caught)

            line_fish.set_data(round_numbers, fish_counts)
            line_catch.set_data(round_numbers, total_catches_this_round)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1) # Pause for plot update visibility
            # --- End Update Plot ---
        # --- End Simulation Loop ---

    except Exception:
        print(f"\nAn unexpected error occurred during simulation:")
        traceback.print_exc()
    finally:
        # --- Cleanup and Final Stats ---
        plt.ioff() # Interactive mode OFF
        print("\n--- Simulation Ended --- ")
        final_fish_count = env.get_fish_count()
        print(f"Final Fish Count: {final_fish_count}")
        print("Agent Totals:")
        agent_totals = []
        for agent in agents:
            agent_totals.append(agent.total_caught)
            print(f"- Agent {agent.agent_id}: Caught {agent.total_caught} fish")

        total_fish_caught_overall = sum(agent_totals)
        print(f"Total fish caught overall: {total_fish_caught_overall}")

        # Optional: Save plot?
        # fig.savefig("simulation_plot.png")

        print("\nClose the plot window to exit.")
        plt.show() # Keep plot window open
        # --- End Cleanup ---

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a multi-agent fishing simulation with live plotting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents.")
    parser.add_argument("--initial_fish", type=int, default=100, help="Initial number of fish.")
    parser.add_argument("--regeneration_rate", type=float, default=1.1, help="Fish regeneration rate.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of simulation rounds.")
    parser.add_argument("--max_catch_per_agent", type=int, default=20, help="Max fish an agent can attempt per round.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model for agent decisions (e.g., gpt-4, gpt-4o).")

    args = parser.parse_args()

    # Validate arguments
    if args.num_agents <= 0:
        parser.error("Number of agents must be positive.")
    if args.initial_fish < 0:
        parser.error("Initial fish count cannot be negative.")
    if args.regeneration_rate <= 0:
        parser.error("Regeneration rate must be positive.")
    if args.rounds <= 0:
        parser.error("Number of rounds must be positive.")
    if args.max_catch_per_agent < 0:
        parser.error("Maximum catch per agent cannot be negative.")

    return args

if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        num_agents=args.num_agents,
        initial_fish=args.initial_fish,
        regeneration_rate=args.regeneration_rate,
        rounds=args.rounds,
        max_catch_per_agent=args.max_catch_per_agent,
        model=args.model
    )
