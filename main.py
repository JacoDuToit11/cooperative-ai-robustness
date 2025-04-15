import argparse
import os
import traceback
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import datetime
import json
import sys

from environment import ResourceEnvironment
from agent import Agent

class Logger:
    def __init__(self, log_file_path: str, print_to_console: bool = True):
        self.log_file = open(log_file_path, 'w')
        self.print_to_console = print_to_console

    def log(self, message: str):
        if self.print_to_console:
            print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def calculate_gini(payoffs: List[float]) -> float:
    """Calculates the Gini coefficient for a list of payoffs."""
    if not payoffs or sum(payoffs) == 0:
        return 0.0
    payoffs = np.sort(np.array(payoffs, dtype=np.float64))
    n = len(payoffs)
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * payoffs)) / (n * np.sum(payoffs))
    return gini

def generate_cooperative_history(
    seed_rounds: int,
    env: ResourceEnvironment,
    agents: List[Agent],
    num_agents: int,
    logger: Logger
) -> List[Dict]:
    """Generates a history of rounds using an optimal sustainable harvesting strategy
       for the doubling-with-limit environment.
    """
    logger.log(f"--- Generating {seed_rounds} Seeded Optimal Sustainable Rounds ---")
    history = []
    for seeded_round in range(1, seed_rounds + 1):
        state = env.get_state()
        resource_at_start = state['current_resource']
        resource_limit = state['resource_limit']
        logger.log(f" Seed Round {seeded_round}: Start Resource={resource_at_start:.2f}")

        if resource_at_start <= 1e-6:
            logger.log(" Seed Round: Resource depleted during seeding.")
            break

        # Optimal sustainable strategy for doubling w/ limit:
        target_remaining = resource_limit / 2.0
        if resource_at_start > target_remaining:
            total_optimal_harvest = resource_at_start - target_remaining
        else:
            total_optimal_harvest = 0.0

        target_harvest_per_agent = (total_optimal_harvest / num_agents) if num_agents > 0 else 0
        target_harvest_per_agent = max(0.0, target_harvest_per_agent)
        target_harvest_per_agent = min(target_harvest_per_agent, resource_at_start / num_agents if num_agents > 0 else 0)

        actual_harvests: Dict[int, float] = {}
        total_actually_harvested = 0.0
        for agent in agents:
            harvest_amount = target_harvest_per_agent
            taken = env.take_resource(harvest_amount)
            if not math.isclose(taken, harvest_amount, rel_tol=1e-6):
                 logger.log(f" Seed Round Warning: Agent {agent.agent_id} target {harvest_amount:.2f} != taken {taken:.2f}. Using {taken:.2f}.")
                 harvest_amount = taken
            agent.update_payoff(harvest_amount)
            actual_harvests[agent.agent_id] = harvest_amount
            total_actually_harvested += harvest_amount

        resource_before_regen = env.get_resource_level()
        logger.log(f" Seed Round: Total Harvested={total_actually_harvested:.2f}, Res Before Regen={resource_before_regen:.2f}")

        history.append({
            'round': seeded_round,
            'resource_at_start': resource_at_start,
            'attempted_harvests': actual_harvests.copy(),
            'actual_harvests': actual_harvests.copy(),
            'total_harvested_this_round': total_actually_harvested,
            'resource_before_regen': resource_before_regen
        })

        env.regenerate()
        logger.log(f" Seed Round: Resource After Regen={env.get_resource_level():.2f}")

    logger.log(f"--- End Seeded Rounds ---")
    return history

def run_simulation(
    num_agents: int,
    initial_resource: float,
    resource_limit: float,
    total_rounds: int,
    model: str,
    seed_rounds: int,
    scenario_name: str,
    critical_threshold: float,
    shock_probability: float,
    shock_magnitude: float,
    change_regen_round: int,
    new_regen_factor: float,
    results_dir: str
):
    """Runs the resource harvesting simulation, logging results to files."""

    log_file_path = os.path.join(results_dir, "game_log.txt")
    logger = Logger(log_file_path)
    logger.log("--- Starting Resource Harvesting Simulation ---")
    param_string1 = f" Parameters: Agents={num_agents}, Initial Resource={initial_resource:.2f}, Limit={resource_limit:.2f}, Threshold={critical_threshold:.2f}"
    param_string2 = f"             Rounds={total_rounds}, Seed Rounds={seed_rounds}, Shock Prob={shock_probability:.2f}, Shock Mag={shock_magnitude:.2f}"
    regen_change_info = f", Regen Change @ Rnd {change_regen_round} to {new_regen_factor:.2f}" if change_regen_round > 0 else ""
    param_string3 = f"             Model={model}, Scenario={scenario_name}{regen_change_info}"
    logger.log(param_string1)
    logger.log(param_string2)
    logger.log(param_string3)

    params = {
        "num_agents": num_agents, "initial_resource": initial_resource, "resource_limit": resource_limit,
        "critical_threshold": critical_threshold, "total_rounds": total_rounds, "model": model,
        "seed_rounds": seed_rounds, "scenario_name": scenario_name, "shock_probability": shock_probability,
        "shock_magnitude": shock_magnitude, "change_regen_round": change_regen_round,
        "new_regen_factor": new_regen_factor, "results_dir": results_dir
    }
    with open(os.path.join(results_dir, "parameters.json"), 'w') as f:
        json.dump(params, f, indent=4)


    agent_log_files = {}
    try:
        for i in range(num_agents):
            agent_id = i + 1
            fname = os.path.join(results_dir, f"agent_{agent_id}_conversation.jsonl")
            agent_log_files[agent_id] = open(fname, 'a', buffering=1)
    except IOError as e:
        logger.log(f"Error opening agent log files: {e}")
        return


    try:
        env = ResourceEnvironment(initial_resource, resource_limit)
    except ValueError as e:
        logger.log(f"Error initializing environment: {e}")
        for f in agent_log_files.values(): f.close()
        logger.close()
        return

    agents: List[Agent] = []
    try:
        for i in range(num_agents):
            agents.append(Agent(
                agent_id=i + 1,
                scenario_name=scenario_name,
                model=model
            ))
    except ValueError as e:
        logger.log(f"Error initializing agents: {e}")
        logger.log(" Ensure OPENAI_API_KEY environment variable is set.")
        for f in agent_log_files.values(): f.close()
        logger.close()
        return
    except Exception as e:
        logger.log(f"Unexpected error initializing agents:")
        logger.log(traceback.format_exc())
        for f in agent_log_files.values(): f.close()
        logger.close()
        return

    history: List[Dict] = []
    start_round = 1
    if seed_rounds > 0:
        if seed_rounds >= total_rounds:
            logger.log("Warning: Seed rounds >= total rounds. Only seeded rounds will run.")
            seed_rounds = total_rounds
        history = generate_cooperative_history(
            seed_rounds=seed_rounds, env=env, agents=agents,
            num_agents=num_agents, logger=logger
        )
        start_round = seed_rounds + 1

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
    ax.axhline(y=env.resource_limit, color='grey', linestyle=':', label=f'Limit ({env.resource_limit:.0f})')
    ax.axhline(y=critical_threshold, color='orange', linestyle=':', label=f'Threshold ({critical_threshold:.1f})')
    ax.set_xlabel("Round")
    ax.set_ylabel("Amount")
    ax.set_title("Resource Harvesting Simulation")
    ax.legend()
    ax.grid(True)

    try:
        for current_round in range(start_round, total_rounds + 1):
            logger.log(f"\n--- Round {current_round} / {total_rounds} --- ")

            regen_changed_this_round = False
            old_regen_factor = None
            if change_regen_round > 0 and current_round == change_regen_round + 1:
                try:
                    old_regen_factor = env.regeneration_factor
                    logger.log(f"  --- Regeneration factor changing from {old_regen_factor:.2f} to {new_regen_factor:.2f} --- ")
                    env.set_regeneration_factor(new_regen_factor)
                    regen_changed_this_round = True
                except ValueError as e:
                    logger.log(f" Error setting new regeneration factor: {e}")
                    break

            resource_at_start_of_round = env.get_resource_level()
            logger.log(f"Resource at round start (before potential shock): {resource_at_start_of_round:.2f}")

            # Apply natural shock
            shock_reduction = env.apply_shock(shock_probability, shock_magnitude)
            resource_after_shock = env.get_resource_level()

            # Update state
            state = env.get_state()
            state['shock_occurred_this_round'] = (shock_reduction > 0)
            state['resource_before_shock'] = resource_at_start_of_round
            if regen_changed_this_round:
                state['regen_changed_this_round'] = True
                state['old_regen_factor'] = old_regen_factor

            # Check termination conditions
            if resource_after_shock <= 1e-6:
                logger.log("Resource depleted (post-shock). Ending simulation.")
                break
            if resource_after_shock < critical_threshold:
                logger.log(f"Resource level ({resource_after_shock:.2f}) below critical threshold ({critical_threshold:.2f}) (post-shock). Ending simulation.")
                break

            logger.log(f"Start Resource (available for harvest): {resource_after_shock:.2f}")
            logger.log(f"Current Regeneration Factor: {env.regeneration_factor:.2f}")

            # 1. Agents choose actions & Log conversations
            attempted_harvests: Dict[int, float] = {}
            total_attempted_harvest = 0.0
            agent_action_details = []
            for agent in agents:
                action_result = agent.choose_action(state, history, num_agents)
                decision = action_result['decision']
                prompt_text = action_result['prompt']
                response_text = action_result['response_content']

                attempted_harvests[agent.agent_id] = decision
                total_attempted_harvest += decision
                agent_action_details.append({'id': agent.agent_id, 'attempt': decision})

                log_entry = {
                    "round": current_round,
                    "prompt": prompt_text,
                    "response": response_text,
                    "decision": decision
                }
                try:
                    agent_log_files[agent.agent_id].write(json.dumps(log_entry) + '\n')
                except IOError as e:
                     logger.log(f"Error writing to agent {agent.agent_id} log file: {e}")

            logger.log(f"Attempted total harvest: {total_attempted_harvest:.2f}")

            # 2. Determine actual harvests
            actual_harvests: Dict[int, float] = {}
            available_resource = env.get_resource_level()
            total_actually_harvested = 0.0

            if total_attempted_harvest <= 1e-6:
                for agent in agents:
                    actual_harvests[agent.agent_id] = 0.0
            elif available_resource <= 1e-6:
                 logger.log(" No resource available to harvest.")
                 for agent in agents:
                    actual_harvests[agent.agent_id] = 0.0
            elif total_attempted_harvest <= available_resource:
                actual_harvests = attempted_harvests
            else:
                logger.log(f" Attempted harvest ({total_attempted_harvest:.2f}) > available ({available_resource:.2f}). Distributing proportionally.")
                for agent_id, attempted in attempted_harvests.items():
                    proportion = (attempted / total_attempted_harvest) if total_attempted_harvest > 1e-6 else 0
                    actual_harvests[agent_id] = proportion * available_resource

            for agent in agents:
                if agent.agent_id not in actual_harvests:
                    actual_harvests[agent.agent_id] = 0.0
            total_actually_harvested = sum(actual_harvests.values())

            # 3. Update environment and agents
            for agent in agents:
                agent_id = agent.agent_id
                harvest_amount = actual_harvests.get(agent_id, 0.0)
                taken = env.take_resource(harvest_amount)
                if not math.isclose(taken, harvest_amount, rel_tol=1e-6):
                    logger.log(f" Warning: Agent {agent_id} allocation {harvest_amount:.2f} != taken {taken:.2f}. Using {taken:.2f}.")
                    harvest_amount = taken
                agent.update_payoff(harvest_amount)
                original_attempt = attempted_harvests.get(agent_id, 0.0)
                logger.log(f" Agent {agent_id}: Attempt={original_attempt:.2f}, Harvested={harvest_amount:.2f}")

            logger.log(f"Total harvested this round: {total_actually_harvested:.2f}")
            resource_before_regen = env.get_resource_level()
            logger.log(f"Resource before regeneration: {resource_before_regen:.2f}")

            # 4. Record history
            history_entry = {
                'round': current_round,
                'resource_at_start': resource_at_start_of_round,
                'shock_reduction': shock_reduction,
                'resource_after_shock': resource_after_shock,
                'regeneration_factor': env.regeneration_factor,
                'attempted_harvests': attempted_harvests.copy(),
                'actual_harvests': actual_harvests.copy(),
                'total_harvested_this_round': total_actually_harvested,
                'resource_before_regen': resource_before_regen
            }
            history.append(history_entry)

            # 5. Regenerate resource
            env.regenerate()
            current_resource_after_regen = env.get_resource_level()
            logger.log(f"Resource after regeneration: {current_resource_after_regen:.2f}")

            round_numbers.append(current_round)
            resource_levels.append(current_resource_after_regen)
            total_harvests_this_round.append(total_actually_harvested)

            line_resource.set_data(round_numbers, resource_levels)
            line_harvest.set_data(round_numbers, total_harvests_this_round)
            ax.relim()
            ax.autoscale_view()

    except Exception as e:
        logger.log(f"\nAn unexpected error occurred during simulation:")
        logger.log(traceback.format_exc())
    finally:
        logger.log("\n--- Simulation Ended --- ")
        final_resource_level = env.get_resource_level()
        logger.log(f"Final Resource Level: {final_resource_level:.2f} / {env.resource_limit:.2f}")

        agent_payoffs = [agent.total_payoff for agent in agents]
        total_harvest_overall = sum(agent_payoffs)

        sustainability = (final_resource_level / env.resource_limit) if env.resource_limit > 0 else 0.0
        equality_gini = calculate_gini(agent_payoffs)
        efficiency = (total_harvest_overall / num_agents) if num_agents > 0 else 0.0

        summary_data = {
            "parameters": params,
            "final_resource_level": final_resource_level,
            "agent_final_payoffs": {agent.agent_id: agent.total_payoff for agent in agents},
            "total_harvest_overall": total_harvest_overall,
            "metrics": {
                "sustainability": sustainability,
                "equality_gini": equality_gini,
                "equality_1_minus_gini": 1.0 - equality_gini,
                "efficiency_avg_payoff": efficiency
            }
        }

        logger.log("\n--- Performance Metrics ---")
        logger.log(f"Sustainability (Final Resource / Limit): {sustainability:.3f}")
        logger.log(f"Equality (1 - Gini Coefficient): {1.0 - equality_gini:.3f} (Gini={equality_gini:.3f}) ")
        logger.log(f"Efficiency (Avg. Payoff per Agent): {efficiency:.3f}")

        logger.log("\nAgent Totals:")
        for agent in agents:
            logger.log(f"- Agent {agent.agent_id}: Harvested {agent.total_payoff:.2f}")
        logger.log(f"Total resource harvested overall: {total_harvest_overall:.2f}")

        # Save summary
        try:
            summary_file_path = os.path.join(results_dir, "summary.json")
            with open(summary_file_path, 'w') as f:
                json.dump(summary_data, f, indent=4)
            logger.log(f"Summary saved to {summary_file_path}")
        except IOError as e:
             logger.log(f"Error saving summary file: {e}")

        # Save history
        try:
            history_file_path = os.path.join(results_dir, "history.json")
            with open(history_file_path, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            logger.log(f"Full history saved to {history_file_path}")
        except IOError as e:
             logger.log(f"Error saving history file: {e}")
        except TypeError as e:
             logger.log(f"Error serializing history data: {e}")


        try:
            plot_file_path = os.path.join(results_dir, "simulation_plot.png")
            fig.savefig(plot_file_path)
            logger.log(f"Plot saved to {plot_file_path}")
        except Exception as e:
            logger.log(f"Error saving plot: {e}")
        finally:
            plt.close(fig)

        for f in agent_log_files.values():
            f.close()
        logger.close()

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a multi-agent resource harvesting simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--scenario", type=str, default="fishing", help="Name of the scenario to run (determines prompts and potentially environment behavior).")
    parser.add_argument("--num_agents", type=int, default=4, help="Number of agents.")
    parser.add_argument("--initial_resource", type=float, default=100.0, help="Initial amount of resource.")
    parser.add_argument("--resource_limit", type=float, default=100.0, help="Maximum resource limit.")
    parser.add_argument("--critical_threshold", type=float, default=5.0, help="Resource level below which the simulation ends.")
    parser.add_argument("--rounds", dest='total_rounds', type=int, default=50, help="Maximum number of simulation rounds.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model for agent decisions.")
    parser.add_argument("--seed_rounds", type=int, default=0, help="Number of initial rounds to simulate with a cooperative strategy.")
    parser.add_argument("--shock_probability", type=float, default=0.0, help="Probability (0.0 to 1.0) of a natural shock reducing resource each round.")
    parser.add_argument("--shock_magnitude", type=float, default=0.2, help="Fraction (0.0 to 1.0) by which resource is reduced during a shock.")
    parser.add_argument("--change_regen_round", type=int, default=0, help="Round AFTER which the regeneration factor changes (0 for no change).")
    parser.add_argument("--new_regen_factor", type=float, default=1.5, help="The new regeneration factor to apply after change_regen_round.")
    parser.add_argument("--results_base_dir", type=str, default="results", help="Base directory for saving simulation results.")
    parser.add_argument("--run_name", type=str, default="", help="Optional specific name for the run directory (appended after timestamp).")

    args = parser.parse_args()

    if args.change_regen_round < 0:
        parser.error("change_regen_round cannot be negative.")
    if args.change_regen_round > 0 and args.change_regen_round >= args.total_rounds:
         parser.error("change_regen_round must be less than total_rounds.")
    if args.new_regen_factor < 0:
         parser.error("new_regen_factor cannot be negative.")
    if args.shock_probability < 0.0 or args.shock_probability > 1.0:
        parser.error("Shock probability must be between 0.0 and 1.0.")
    if args.shock_magnitude < 0.0 or args.shock_magnitude > 1.0:
        parser.error("Shock magnitude must be between 0.0 and 1.0.")
    if args.critical_threshold < 0:
        parser.error("Critical threshold cannot be negative.")
    if args.critical_threshold >= args.resource_limit:
        parser.error("Critical threshold must be less than the resource limit.")
    if args.initial_resource < args.critical_threshold:
         print(f"Warning: Initial resource ({args.initial_resource}) is already below critical threshold ({args.critical_threshold}). Simulation may end immediately.")
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
    import os
    import math
    import numpy as np
    import datetime
    import json
    import sys

    args = parse_args()

    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name_suffix = f"_{args.run_name}" if args.run_name else ""
    results_dir_name = f"{timestamp}{run_name_suffix}"
    results_dir_path = os.path.join(args.results_base_dir, results_dir_name)
    try:
        os.makedirs(results_dir_path, exist_ok=True)
        print(f"Results will be saved to: {results_dir_path}")
    except OSError as e:
        print(f"Error creating results directory {results_dir_path}: {e}")
        sys.exit(1)

    run_simulation(
        num_agents=args.num_agents,
        initial_resource=args.initial_resource,
        resource_limit=args.resource_limit,
        total_rounds=args.total_rounds,
        model=args.model,
        seed_rounds=args.seed_rounds,
        scenario_name=args.scenario,
        critical_threshold=args.critical_threshold,
        shock_probability=args.shock_probability,
        shock_magnitude=args.shock_magnitude,
        change_regen_round=args.change_regen_round,
        new_regen_factor=args.new_regen_factor,
        results_dir=results_dir_path
    )
