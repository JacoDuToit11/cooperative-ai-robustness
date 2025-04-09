import argparse
import os
from fishing_environment import FishingEnvironment
from fishing_agent import FishingAgent
import matplotlib.pyplot as plt
import time # To pause slightly for plot updates
import traceback # Import traceback module

def run_simulation(num_agents: int, initial_fish: int, regeneration_rate: float, rounds: int, max_catch_per_agent: int):
    """Runs the fishing simulation with live plotting."""

    print("--- Starting Fishing Simulation ---")
    print(f"Parameters: Agents={num_agents}, Initial Fish={initial_fish}, Regen Rate={regeneration_rate}, Rounds={rounds}, Max Catch/Agent={max_catch_per_agent}")

    # Initialize environment
    try:
        env = FishingEnvironment(initial_fish, regeneration_rate)
    except ValueError as e:
        print(f"Error initializing environment: {e}")
        return

    # Initialize agents
    agents: list[FishingAgent] = []
    try:
        for i in range(num_agents):
            # Agent IDs are 1-based for prompts, index is 0-based
            agents.append(FishingAgent(agent_id=i + 1, max_catch_per_round=max_catch_per_agent))
    except ValueError as e:
        print(f"Error initializing agents: {e}")
        print("Ensure OPENAI_API_KEY environment variable is set.")
        return
    except Exception as e:
        print(f"Unexpected error initializing agents: {e}")
        return

    history = []

    # --- Plotting Setup ---
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots()
    round_numbers = []
    fish_counts = []
    total_catches_per_round = []
    # Initialize plot with starting state
    round_numbers.append(0)
    fish_counts.append(env.get_fish_count())
    total_catches_per_round.append(0) # No catch before round 1
    line_fish, = ax.plot(round_numbers, fish_counts, marker='o', linestyle='-', color='b', label='Fish Population')
    line_catch, = ax.plot(round_numbers, total_catches_per_round, marker='x', linestyle='--', color='r', label='Total Catch This Round')
    ax.set_xlabel("Round")
    ax.set_ylabel("Count")
    ax.set_title("Fishing Simulation Live View")
    ax.legend()
    ax.grid(True)
    plt.show()
    # --- End Plotting Setup ---

    try:
        # Simulation loop
        for current_round in range(1, rounds + 1):
            print(f"\n--- Round {current_round} --- ")
            state = env.get_state()
            fish_at_start_of_round = state['current_fish']
            print(f"Start of Round: {fish_at_start_of_round} fish available.")

            if fish_at_start_of_round <= 0:
                print("No fish left. Ending simulation early.")
                break

            # 1. Agents decide how much to fish
            attempted_catches = {}
            total_attempted_catch = 0
            for agent in agents:
                # Pass current state, history, and total agent count to the agent
                decision = agent.choose_action(state, history, num_agents)
                attempted_catches[agent.agent_id] = decision
                total_attempted_catch += decision

            print(f"Total attempted catch: {total_attempted_catch}")

            # 2. Determine actual catches based on available fish
            actual_catches = {}
            available_fish = env.get_fish_count()

            if total_attempted_catch <= 0:
                # No one tried to fish, or negative values were clamped to 0
                for agent in agents:
                    actual_catches[agent.agent_id] = 0
            elif available_fish <= 0:
                 # No fish left to take
                 print("No fish available to catch despite attempts.")
                 for agent in agents:
                    actual_catches[agent.agent_id] = 0
            elif total_attempted_catch <= available_fish:
                # Enough fish for everyone's request
                for agent_id, attempted in attempted_catches.items():
                    actual_catches[agent_id] = attempted
            else:
                # Not enough fish - distribute proportionally
                print(f"Attempted catch ({total_attempted_catch}) exceeds available fish ({available_fish}). Distributing proportionally.")
                for agent_id, attempted in attempted_catches.items():
                    proportion = attempted / total_attempted_catch
                    actual_catch = int(proportion * available_fish) # Take the floor
                    actual_catches[agent_id] = actual_catch

            # 3. Update environment (take fish) and agents (update totals)
            total_actually_caught = 0
            for agent in agents:
                agent_id = agent.agent_id
                caught_amount = actual_catches[agent_id]
                # The environment handles taking the fish
                taken = env.take_fish(caught_amount) # Should match caught_amount if logic is correct
                if taken != caught_amount:
                     print(f"Warning: Agent {agent_id} was allocated {caught_amount} but environment only gave {taken}.") # Should not happen with current logic
                     caught_amount = taken # Adjust agent's record to what was actually taken

                agent.update_catch(caught_amount)
                total_actually_caught += caught_amount
                print(f"Agent {agent_id}: Attempted={attempted_catches[agent_id]}, Caught={caught_amount}")

            print(f"Total fish actually caught this round: {total_actually_caught}")
            print(f"Fish remaining before regeneration: {env.get_fish_count()}")

            # 4. Record history for this round *before* regeneration
            history.append({
                'round': current_round,
                'fish_at_start': fish_at_start_of_round,
                'attempted_catches': attempted_catches.copy(),
                'actual_catches': actual_catches.copy(),
                'fish_before_regen': env.get_fish_count()
            })

            # 5. Environment regenerates fish
            env.regenerate()
            current_fish_after_regen = env.get_fish_count()
            print(f"Fish after regeneration: {current_fish_after_regen}")

            # --- Update Plot ---
            round_numbers.append(current_round)
            fish_counts.append(current_fish_after_regen)
            total_catches_per_round.append(total_actually_caught)

            # Update plot data
            line_fish.set_data(round_numbers, fish_counts)
            line_catch.set_data(round_numbers, total_catches_per_round)

            # Adjust plot limits
            ax.relim()
            ax.autoscale_view()

            # Redraw plot and pause
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1) # Small pause to allow plot to update
            # --- End Update Plot ---

    except Exception as e:
        print(f"\nAn error occurred during simulation:")
        traceback.print_exc() # Print the full traceback
    finally:
        # --- Keep Plot Open ---
        plt.ioff() # Turn off interactive mode
        print("\n--- Simulation Ended (or Interrupted) ---")
        print(f"Final fish count: {env.get_fish_count()}")
        print("Agent Totals:")
        for agent in agents:
            print(f"- Agent {agent.agent_id}: Caught {agent.total_caught} fish")

        total_fish_caught_overall = sum(agent.total_caught for agent in agents)
        print(f"Total fish caught by all agents across all rounds: {total_fish_caught_overall}")

        print("\nClose the plot window to exit.")
        plt.show() # Display the final plot and block until closed
        # --- End Keep Plot Open ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-agent fishing simulation with live plotting.")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents.")
    parser.add_argument("--initial_fish", type=int, default=100, help="Initial number of fish.")
    parser.add_argument("--regeneration_rate", type=float, default=1.1, help="Fish regeneration rate (e.g., 1.1 for 10%% growth).")
    parser.add_argument("--rounds", type=int, default=10, help="Number of simulation rounds.")
    parser.add_argument("--max_catch_per_agent", type=int, default=20, help="Maximum fish an agent can attempt to catch per round.")

    args = parser.parse_args()

    # Basic validation for args
    if args.num_agents <= 0:
        print("Error: Number of agents must be positive.")
    elif args.initial_fish < 0:
        print("Error: Initial fish count cannot be negative.")
    elif args.regeneration_rate <= 0:
        print("Error: Regeneration rate must be positive.")
    elif args.rounds <= 0:
        print("Error: Number of rounds must be positive.")
    elif args.max_catch_per_agent < 0:
        print("Error: Maximum catch per agent cannot be negative.")
    else:
        run_simulation(
            num_agents=args.num_agents,
            initial_fish=args.initial_fish,
            regeneration_rate=args.regeneration_rate,
            rounds=args.rounds,
            max_catch_per_agent=args.max_catch_per_agent
        )
