import argparse
import os
import traceback
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import datetime
import json
import sys
import time
import statistics
import random

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
) -> Tuple[List[Dict], List[Tuple[int, float, float]]]:
    """Generates a history of rounds using an optimal sustainable harvesting strategy
       for the doubling-with-limit environment.
    """
    logger.log(f"--- Generating {seed_rounds} Seeded Optimal Sustainable Rounds ---")
    history = []
    round_data_for_plot = []
    for seeded_round in range(1, seed_rounds + 1):
        state = env.get_state()
        resource_at_start = state['current_resource']
        resource_limit = state['resource_limit']
        logger.log(f" Seed Round {seeded_round}: Start Resource={resource_at_start:.2f}")

        if resource_at_start <= 1e-6:
            logger.log(" Seed Round: Resource depleted during seeding.")
            break

        target_remaining = resource_limit / 2.0
        total_optimal_harvest = max(0.0, resource_at_start - target_remaining)
        target_harvest_per_agent = (total_optimal_harvest / num_agents) if num_agents > 0 else 0
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
        resource_after_regen = env.get_resource_level()
        logger.log(f" Seed Round: Resource After Regen={resource_after_regen:.2f}")
        # round_data_for_plot.append((seeded_round, resource_after_regen, total_actually_harvested))

    logger.log(f"--- End Seeded Rounds ---")
    # Return history and placeholder (empty list) for plot data as it's recalculated later
    return history, [] 


def run_simulation(
    run_index: int,
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
    run_results_dir: str,
    save_conversations: bool
) -> Tuple[Dict, List[int], List[float], List[float], List[Tuple[int, float, float]]]:
    """Runs a single resource harvesting simulation instance."""

    log_file_path = os.path.join(run_results_dir, "game_log.txt")
    logger = Logger(log_file_path, print_to_console=(run_index == 1))
    logger.log(f"--- Starting Simulation Run {run_index} --- ")
    param_string1 = f" Parameters: Agents={num_agents}, Initial Resource={initial_resource:.2f}, Limit={resource_limit:.2f}, Threshold={critical_threshold:.2f}"
    param_string2 = f"             Rounds={total_rounds}, Seed Rounds={seed_rounds}, Shock Prob={shock_probability:.2f}, Shock Mag={shock_magnitude:.2f}"
    regen_change_info = f", Regen Change @ Rnd {change_regen_round} to {new_regen_factor:.2f}" if change_regen_round > 0 else ""
    param_string3 = f"             Model={model}, Scenario={scenario_name}{regen_change_info}"
    logger.log(param_string1); logger.log(param_string2); logger.log(param_string3)
    logger.log(f"             Save Conversations: {save_conversations}")

    params = {
        "run_index": run_index,
        "num_agents": num_agents, "initial_resource": initial_resource, "resource_limit": resource_limit,
        "critical_threshold": critical_threshold, "total_rounds": total_rounds, "model": model,
        "seed_rounds": seed_rounds, "scenario_name": scenario_name, "shock_probability": shock_probability,
        "shock_magnitude": shock_magnitude, "change_regen_round": change_regen_round,
        "new_regen_factor": new_regen_factor, "run_results_dir": run_results_dir,
        "save_conversations": save_conversations
    }
    with open(os.path.join(run_results_dir, "parameters.json"), 'w') as f:
        json.dump(params, f, indent=4)

    agent_log_files = {}
    if save_conversations:
        try:
            for i in range(num_agents):
                agent_id = i + 1
                fname = os.path.join(run_results_dir, f"agent_{agent_id}_conversation.jsonl")
                agent_log_files[agent_id] = open(fname, 'a', buffering=1)
        except IOError as e:
            logger.log(f"Error opening agent log files: {e}")
            return {}, [], [], [], []

    try:
        env = ResourceEnvironment(initial_resource, resource_limit)
    except ValueError as e:
        logger.log(f"Error initializing environment: {e}"); logger.close()
        for f in agent_log_files.values(): f.close()
        return {}, [], [], [], []

    agents: List[Agent] = []
    try:
        for i in range(num_agents):
            agents.append(Agent(agent_id=i + 1, scenario_name=scenario_name, model=model))
    except ValueError as e:
        logger.log(f"Error initializing agents: {e}"); logger.log(" Ensure OPENAI_API_KEY is set."); logger.close()
        for f in agent_log_files.values(): f.close()
        return {}, [], [], [], []
    except Exception as e:
        logger.log(f"Unexpected error initializing agents:"); logger.log(traceback.format_exc()); logger.close()
        for f in agent_log_files.values(): f.close()
        return {}, [], [], [], []

    history: List[Dict] = []
    start_round = 1
    all_rounds = []
    all_resource_levels = []
    all_total_harvests = []
    shock_details = []

    if seed_rounds > 0:
        if seed_rounds >= total_rounds:
            logger.log("Warning: Seed rounds >= total rounds. Only seeded rounds will run.")
            seed_rounds = total_rounds
        history, _ = generate_cooperative_history(
            seed_rounds=seed_rounds, env=env, agents=agents,
            num_agents=num_agents, logger=logger
        )
        start_round = seed_rounds + 1
        for item in history:
            if item['round'] <= seed_rounds:
                round_num = item['round']
                res_start = item['resource_at_start']
                harvest = item['total_harvested_this_round']
                all_rounds.append(round_num)
                all_resource_levels.append(res_start)
                all_total_harvests.append(harvest)

    fig_run = None
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
            shock_reduction = env.apply_shock(shock_probability, shock_magnitude)
            resource_after_shock = env.get_resource_level()
            
            shock_occurred = shock_reduction > 0
            if shock_occurred:
                shock_details.append((current_round, resource_at_start_of_round, resource_after_shock))
                logger.log(f"  *** Resource Shock Occurred! Reduction: {shock_reduction:.2f} ***")

            state = env.get_state()
            state['shock_occurred_this_round'] = shock_occurred
            state['resource_before_shock'] = resource_at_start_of_round
            if regen_changed_this_round:
                state['regen_changed_this_round'] = True
                state['old_regen_factor'] = old_regen_factor
            if resource_after_shock <= 1e-6:
                logger.log("Resource depleted (post-shock). Ending simulation.")
                all_rounds.append(current_round)
                all_resource_levels.append(resource_after_shock)
                all_total_harvests.append(0.0)
                break
            if resource_after_shock < critical_threshold:
                logger.log(f"Resource level ({resource_after_shock:.2f}) below critical threshold ({critical_threshold:.2f}) (post-shock). Ending simulation.")
                all_rounds.append(current_round)
                all_resource_levels.append(resource_after_shock)
                all_total_harvests.append(0.0)
                break
            logger.log(f"Start Resource (available for harvest): {resource_after_shock:.2f}")
            logger.log(f"Current Regeneration Factor: {env.regeneration_factor:.2f}")

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
                if save_conversations and agent.agent_id in agent_log_files:
                    log_entry = {"round": current_round, "prompt": prompt_text, "response": response_text, "decision": decision}
                    try:
                        agent_log_files[agent.agent_id].write(json.dumps(log_entry) + '\n')
                    except IOError as e:
                        logger.log(f"Error writing to agent {agent.agent_id} log file: {e}")
            logger.log(f"Attempted total harvest: {total_attempted_harvest:.2f}")
            actual_harvests: Dict[int, float] = {}
            available_resource = resource_after_shock
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

            history_entry = { 'round': current_round, 'resource_at_start': resource_at_start_of_round, 'shock_reduction': shock_reduction, 'resource_after_shock': resource_after_shock, 'regeneration_factor': env.regeneration_factor, 'attempted_harvests': attempted_harvests.copy(), 'actual_harvests': actual_harvests.copy(), 'total_harvested_this_round': total_actually_harvested, 'resource_before_regen': resource_before_regen }
            history.append(history_entry)

            env.regenerate()
            current_resource_after_regen = env.get_resource_level()
            logger.log(f"Resource after regeneration: {current_resource_after_regen:.2f}")

            all_rounds.append(current_round)
            all_resource_levels.append(resource_after_shock)
            all_total_harvests.append(total_actually_harvested)

    except Exception as e:
        logger.log(f"\nAn unexpected error occurred during simulation run {run_index}:")
        logger.log(traceback.format_exc())
    finally:
        logger.log(f"\n--- Simulation Run {run_index} Ended --- ")
        final_resource_level = env.get_resource_level()
        logger.log(f"Final Resource Level: {final_resource_level:.2f} / {env.resource_limit:.2f}")
        agent_payoffs = [agent.total_payoff for agent in agents]
        total_harvest_overall = sum(agent_payoffs)
        sustainability = (final_resource_level / env.resource_limit) if env.resource_limit > 0 else 0.0
        equality_gini = calculate_gini(agent_payoffs)
        efficiency = (total_harvest_overall / num_agents) if num_agents > 0 else 0.0
        summary_data = { "run_index": run_index, "parameters": params, "final_resource_level": final_resource_level, "agent_final_payoffs": {agent.agent_id: agent.total_payoff for agent in agents}, "total_harvest_overall": total_harvest_overall, "metrics": { "sustainability": sustainability, "equality_gini": equality_gini, "equality_1_minus_gini": 1.0 - equality_gini, "efficiency_avg_payoff": efficiency } }

        logger.log("\n--- Performance Metrics --- ")
        logger.log(f"Sustainability (Final Resource / Limit): {sustainability:.3f}")
        logger.log(f"Equality (1 - Gini Coefficient): {1.0 - equality_gini:.3f} (Gini={equality_gini:.3f}) ")
        logger.log(f"Efficiency (Avg. Payoff per Agent): {efficiency:.3f}")
        logger.log("\nAgent Totals:")
        for agent in agents:
            logger.log(f"- Agent {agent.agent_id}: Harvested {agent.total_payoff:.2f}")
        logger.log(f"Total resource harvested overall: {total_harvest_overall:.2f}")

        try:
            summary_file_path = os.path.join(run_results_dir, "summary.json")
            with open(summary_file_path, 'w') as f:
                json.dump(summary_data, f, indent=4)
            logger.log(f"Summary saved to {summary_file_path}")
        except IOError as e:
             logger.log(f"Error saving summary file: {e}")

        try:
            history_file_path = os.path.join(run_results_dir, "history.json")
            with open(history_file_path, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            logger.log(f"Full history saved to {history_file_path}")
        except IOError as e:
             logger.log(f"Error saving history file: {e}")
        except TypeError as e:
             logger.log(f"Error serializing history data: {e}")

        # Generate and save individual plot for this run
        try:
            if all_rounds:
                fig_run, ax_run = plt.subplots(figsize=(10, 6))

                line_resource, = ax_run.plot(all_rounds, all_resource_levels, marker='o', linestyle='-', color='g', label='Resource at Round Start', zorder=5)
                line_harvest, = ax_run.plot(all_rounds, all_total_harvests, marker='x', linestyle='--', color='r', label='Total Harvest During Round', zorder=5)
                line_limit = ax_run.axhline(y=resource_limit, color='grey', linestyle=':', label=f'Limit ({resource_limit:.0f})', zorder=1)
                line_threshold = ax_run.axhline(y=critical_threshold, color='orange', linestyle=':', label=f'Threshold ({critical_threshold:.1f})', zorder=1)

                shock_line_handle = None
                for i, (r, pre_shock, post_shock) in enumerate(shock_details):
                    line = ax_run.plot([r, r], [pre_shock, post_shock], 
                                       color='purple', linestyle='dashed', linewidth=1.5, 
                                       label='Shock Drop' if i == 0 else "", zorder=4, alpha=0.9)
                    if i == 0:
                        shock_line_handle = line[0]

                handles = [line_resource, line_harvest, line_limit, line_threshold]
                labels = [h.get_label() for h in handles]
                if shock_line_handle:
                    handles.append(shock_line_handle)
                    labels.append(shock_line_handle.get_label())

                ax_run.set_xlabel("Round")
                ax_run.set_ylabel("Amount")
                ax_run.set_title(f"Simulation Run {run_index}")
                ax_run.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                ax_run.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                ax_run.minorticks_on()
                ax_run.set_ylim(bottom=-max(resource_limit, initial_resource)*0.02)
                ax_run.set_xlim(left=0.5, right=max(all_rounds) + 0.5 if all_rounds else 1.5)

                fig_run.legend(handles=handles, labels=labels, loc='lower center', 
                               bbox_to_anchor=(0.5, -0.18),
                               ncol=3,
                               frameon=False)

                fig_run.subplots_adjust(bottom=0.25)

                plot_file_path = os.path.join(run_results_dir, "simulation_plot.png")
                fig_run.savefig(plot_file_path, bbox_inches='tight')
                logger.log(f"Individual run plot saved to {plot_file_path}")
            else:
                logger.log("No data to generate individual plot for this run.")
        except Exception as e:
            logger.log(f"Error generating or saving individual run plot: {e}")
            logger.log(traceback.format_exc())
        finally:
            if fig_run is not None:
                 plt.close(fig_run)

        for f in agent_log_files.values():
            f.close()
        logger.close()

        metrics_to_return = summary_data.get('metrics', {})
        return metrics_to_return, all_rounds, all_resource_levels, all_total_harvests, shock_details

def parse_args():
    """Parses command-line arguments, expecting only the config file path."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent resource harvesting simulations based on a JSON config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON configuration file for the simulation.")

    args = parser.parse_args()
    return args

def validate_config(config: Dict):
    """Validates the loaded configuration dictionary."""
    required_keys = [
        "scenario", "num_agents", "initial_resource", "resource_limit",
        "critical_threshold", "total_rounds", "model", "seed_rounds",
        "shock_probability", "shock_magnitude", "change_regen_round",
        "new_regen_factor", "results_base_dir", "run_name", "num_runs",
        "save_conversations", "random_seed"
    ]
    config_keys = config.keys()
    missing_keys = [key for key in required_keys if key not in config_keys and key != "random_seed"]
    if "random_seed" not in config_keys:
        config["random_seed"] = None
        print("Note: 'random_seed' not found in config, using default None (non-deterministic)." )

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    if not isinstance(config["num_runs"], int): raise TypeError("num_runs must be an integer")
    if not isinstance(config["save_conversations"], bool): raise TypeError("save_conversations must be a boolean")
    if config["random_seed"] is not None and not isinstance(config["random_seed"], int):
         raise TypeError("random_seed must be an integer or null/None")
    if not isinstance(config["num_agents"], int): raise TypeError("num_agents must be an integer")
    if not isinstance(config["total_rounds"], int): raise TypeError("total_rounds must be an integer")

    if config["num_runs"] <= 0:
        raise ValueError("Number of runs (num_runs) must be positive.")
    if config["change_regen_round"] < 0:
        raise ValueError("change_regen_round cannot be negative.")
    if config["change_regen_round"] > 0 and config["change_regen_round"] >= config["total_rounds"]:
        raise ValueError("change_regen_round must be less than total_rounds.")
    if config["new_regen_factor"] < 0:
        raise ValueError("new_regen_factor cannot be negative.")
    if not (0.0 <= config["shock_probability"] <= 1.0):
        raise ValueError("Shock probability must be between 0.0 and 1.0.")
    if not (0.0 <= config["shock_magnitude"] <= 1.0):
        raise ValueError("Shock magnitude must be between 0.0 and 1.0.")
    if config["critical_threshold"] < 0:
        raise ValueError("Critical threshold cannot be negative.")
    if config["critical_threshold"] >= config["resource_limit"]:
        raise ValueError("Critical threshold must be less than the resource limit.")
    if config["initial_resource"] < config["critical_threshold"]:
        print(f"Warning: Initial resource ({config['initial_resource']}) is below critical threshold ({config['critical_threshold']}). Simulation may end immediately.")
    if config["num_agents"] <= 0:
        raise ValueError("Number of agents must be positive.")
    if config["initial_resource"] < 0:
        raise ValueError("Initial resource cannot be negative.")
    if config["resource_limit"] <= 0:
        raise ValueError("Resource limit must be positive.")
    if config["total_rounds"] <= 0:
        raise ValueError("Total number of rounds must be positive.")
    if config["seed_rounds"] < 0:
        raise ValueError("Seed rounds cannot be negative.")

    print("Configuration validation successful.")

def aggregate_plot_data(all_run_data: List[Tuple[List[int], List[float], List[float]]], max_rounds: int) -> Tuple[List[int], List[float], List[float]]:
    """Aggregates resource and harvest data across runs for plotting averages."""
    agg_rounds = list(range(1, max_rounds + 1))
    agg_resource = {r: [] for r in agg_rounds}
    agg_harvest = {r: [] for r in agg_rounds}

    for rounds, resources, harvests in all_run_data:
        min_len = min(len(rounds), len(resources), len(harvests))
        for i in range(min_len):
            r = rounds[i]
            if r in agg_rounds:
                agg_resource[r].append(resources[i])
                agg_harvest[r].append(harvests[i])

    avg_resource = [statistics.mean(agg_resource[r]) if agg_resource[r] else 0 for r in agg_rounds]
    avg_harvest = [statistics.mean(agg_harvest[r]) if agg_harvest[r] else 0 for r in agg_rounds]

    return agg_rounds, avg_resource, avg_harvest

if __name__ == "__main__":
    args = parse_args()

    try:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON configuration file {args.config}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        sys.exit(1)

    try:
        validate_config(config_data)
    except (ValueError, TypeError) as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    num_runs = config_data["num_runs"]
    random_seed = config_data["random_seed"]
    save_conversations = config_data["save_conversations"]
    run_name = config_data.get("run_name", "")
    results_base_dir = config_data["results_base_dir"]
    total_rounds_config = config_data["total_rounds"]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name_suffix = f"_{run_name}" if run_name else ""
    main_results_dir_name = f"{timestamp}{run_name_suffix}"
    main_results_dir_path = os.path.join(results_base_dir, main_results_dir_name)
    try:
        os.makedirs(main_results_dir_path, exist_ok=True)
        print(f"Main results directory: {main_results_dir_path}")
    except OSError as e:
        print(f"Error creating main results directory {main_results_dir_path}: {e}")
        sys.exit(1)

    master_params = config_data.copy()
    master_params["config_file_path"] = args.config
    master_params["main_results_dir"] = main_results_dir_path
    try:
        master_params_path = os.path.join(main_results_dir_path, "master_parameters.json")
        with open(master_params_path, 'w') as f:
            json.dump(master_params, f, indent=4)
        print(f"Master parameters saved to {master_params_path}")
    except IOError as e:
        print(f"Error saving master parameters: {e}")

    all_run_metrics = []
    all_run_plot_data_no_shocks = []

    print(f"\nStarting {num_runs} simulation runs based on {args.config}...")
    for i in range(1, num_runs + 1):
        if isinstance(random_seed, int):
            print(f" Seeding run {i} with seed: {random_seed}")
            random.seed(random_seed)

        run_dir_name = f"run_{i}"
        run_dir_path = os.path.join(main_results_dir_path, run_dir_name)
        try:
            os.makedirs(run_dir_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory for run {i}: {run_dir_path}, Error: {e}")
            continue

        print(f" Running simulation {i}/{num_runs} -> {run_dir_path}")
        try:
            metrics, rounds, resources, harvests, run_shock_details = run_simulation(
                run_index=i,
                num_agents=config_data["num_agents"],
                initial_resource=config_data["initial_resource"],
                resource_limit=config_data["resource_limit"],
                total_rounds=config_data["total_rounds"],
                model=config_data["model"],
                seed_rounds=config_data["seed_rounds"],
                scenario_name=config_data["scenario"],
                critical_threshold=config_data["critical_threshold"],
                shock_probability=config_data["shock_probability"],
                shock_magnitude=config_data["shock_magnitude"],
                change_regen_round=config_data["change_regen_round"],
                new_regen_factor=config_data["new_regen_factor"],
                run_results_dir=run_dir_path,
                save_conversations=save_conversations
            )
            if metrics:
                all_run_metrics.append(metrics)
                all_run_plot_data_no_shocks.append((rounds, resources, harvests))
            else:
                print(f" Run {i} did not return valid results.")
        except Exception as e:
            print(f"Error during simulation run {i}:")
            print(traceback.format_exc())

    print(f"\n--- Aggregating Results from {len(all_run_metrics)} successful runs --- ")
    if not all_run_metrics:
        print("No successful simulation runs to aggregate.")
        sys.exit(0)

    avg_metrics = {}
    metric_keys = all_run_metrics[0].keys()
    for key in metric_keys:
        try:
            values = [m[key] for m in all_run_metrics if key in m]
            if values:
                avg_metrics[key] = statistics.mean(values)
                avg_metrics[key + "_stddev"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                 avg_metrics[key] = None
                 avg_metrics[key + "_stddev"] = None
        except KeyError:
             print(f" Warning: Metric key '{key}' not found in all runs.")
             avg_metrics[key] = None
             avg_metrics[key + "_stddev"] = None
        except statistics.StatisticsError as e:
             print(f" Warning: Could not calculate statistics for metric '{key}': {e}")
             avg_metrics[key] = None
             avg_metrics[key + "_stddev"] = None

    aggregated_summary = {
        "num_successful_runs": len(all_run_metrics),
        "total_runs_requested": num_runs,
        "average_metrics": avg_metrics,
        "config_file_used": args.config
    }
    try:
        agg_summary_path = os.path.join(main_results_dir_path, "aggregated_summary.json")
        with open(agg_summary_path, 'w') as f:
            json.dump(aggregated_summary, f, indent=4)
        print(f"Aggregated summary saved to {agg_summary_path}")
        print("\n--- Aggregated Metrics (Average over successful runs) ---")
        for key, value in avg_metrics.items():
            if value is not None and not key.endswith("_stddev"):
                stddev_key = key + "_stddev"
                stddev = avg_metrics.get(stddev_key, 0.0)
                print(f" {key.replace('_', ' ').title()}: {value:.4f} (StdDev: {stddev:.4f})")
    except IOError as e:
        print(f"Error saving aggregated summary: {e}")

    try:
        if all_run_plot_data_no_shocks:
            agg_rounds, avg_resource, avg_harvest = aggregate_plot_data(all_run_plot_data_no_shocks, total_rounds_config)

            fig_agg, ax_agg = plt.subplots(figsize=(10, 6))
            if agg_rounds:
                line1, = ax_agg.plot(agg_rounds, avg_resource, marker='o', linestyle='-', color='g', label='Avg. Resource at Round Start')
                line2, = ax_agg.plot(agg_rounds, avg_harvest, marker='x', linestyle='--', color='r', label='Avg. Total Harvest During Round')
                line_limit_agg = ax_agg.axhline(y=config_data["resource_limit"], color='grey', linestyle=':', label=f'Limit ({config_data["resource_limit"]:.0f})')
                line_thresh_agg = ax_agg.axhline(y=config_data["critical_threshold"], color='orange', linestyle=':', label=f'Threshold ({config_data["critical_threshold"]:.1f})')
                
                handles = [line1, line2, line_limit_agg, line_thresh_agg]
                labels = [h.get_label() for h in handles]

                ax_agg.set_xlabel("Round")
                ax_agg.set_ylabel("Average Amount")
                ax_agg.set_title("Aggregated Simulation Results") 
                ax_agg.grid(True)
                ax_agg.set_ylim(bottom=0)
                ax_agg.set_xlim(left=0.5, right=max(agg_rounds) + 0.5)

                fig_agg.legend(handles=handles, labels=labels, loc='lower center', 
                               bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
                
                fig_agg.subplots_adjust(bottom=0.2)

                agg_plot_path = os.path.join(main_results_dir_path, "aggregated_plot.png")
                fig_agg.savefig(agg_plot_path, bbox_inches='tight')
                print(f"Aggregated plot saved to {agg_plot_path}")
            else:
                 print("Aggregated rounds list is empty, skipping aggregated plot generation.")
            plt.close(fig_agg)
        else:
            print("No plot data from successful runs to aggregate or plot.")

    except Exception as e:
        print(f"Error generating or saving aggregated plot: {e}")
        print(traceback.format_exc())

    print("\nMulti-run simulation and aggregation complete.")
