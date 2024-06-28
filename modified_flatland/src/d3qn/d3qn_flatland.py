import random
from enum import Enum

import numpy as np
import torch
from flatland.envs.rail_env import RailEnvActions
from flatland.utils.rendertools import RenderTool,AgentRenderVariant
# try:
#     import wandb

#     use_wandb = True
# except ImportError as e:
#     print("wandb is not installed, TensorBoard on specified directory will be used!")
use_wandb = False

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.common.action_skipping_masking import get_action_masking
from src.common.flatland_railenv import FlatlandRailEnv
from src.common.utils import Timer, TensorBoardLogger
from src.d3qn.policy import D3QNPolicy
import matplotlib.pyplot as plt 

class FingerprintType(Enum):
    """
    How the fingerprint is made
    """

    EPSILON = 1
    STEP = 2
    EPISODE = 3
    EPSILON_STEP = 4
    EPSILON_EPISODE = 5
    STEP_EPISODE = 6
    EPSILON_STEP_EPISODE = 7


def add_fingerprints(obs, num_agents, fingerprint_type, eps, step, episode):
    """

    :param obs: The observation to modify
    :param num_agents: Total number of agents
    :param fingerprint_type: instructs the function on how to build the fingerprint
    :param eps: the current exploration rate
    :param step: the current time step
    :param episode: the current episode
    :return: the observations with fingerprints
    """

    for a in range(num_agents):
        if obs[a] is not None:
            fingerprint = []

            if fingerprint_type in [FingerprintType.EPSILON, FingerprintType.EPSILON_STEP,
                                    FingerprintType.EPSILON_EPISODE, FingerprintType.EPSILON_STEP_EPISODE]:
                fingerprint.append(eps)

            if fingerprint_type in [FingerprintType.STEP, FingerprintType.EPSILON_STEP, FingerprintType.STEP_EPISODE,
                                    FingerprintType.EPSILON_STEP_EPISODE]:
                fingerprint.append(step)

            if fingerprint_type in [FingerprintType.EPISODE, FingerprintType.EPSILON_EPISODE,
                                    FingerprintType.STEP_EPISODE, FingerprintType.EPSILON_STEP_EPISODE]:
                fingerprint.append(episode)

            assert not any(map(lambda x: x is None, fingerprint)), "Fingerprint cannot be made, some arguments are " \
                                                                   "None."

            obs[a] = np.append(obs[a], fingerprint)

    return obs
import PIL 
def render_env(env_renderer,frames_list, show=False, frames=False, show_observations=True,):
    """
    Renders the current state of the environment
    """
    env_renderer.render_env(show=show, frames=frames, show_observations=show_observations)
    image = env_renderer.gl.get_image()
    frames_list.append(PIL.Image.fromarray(env_renderer.gl.get_image()))

def train_multiple_agents(env_params, train_params):
    # Initialize wandb
    if use_wandb:
        wandb.init(project=train_params.wandb_project,
                   entity=train_params.wandb_entity,
                   tags=train_params.wandb_tag,
                   config={**vars(train_params), **vars(env_params)},
                   sync_tensorboard=True)

    # Environment parameters
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth

    # Training parameters
    eps_start = train_params.eps_start
    eps_end = train_params.eps_end
    eps_decay = train_params.eps_decay

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    env = FlatlandRailEnv(train_params,
                          env_params,
                          tree_observation,
                          env_params.custom_observations,
                          env_params.reward_shaping,
                          train_params.print_stats)
    
    env.reset()

    # The action space of flatland is 5 discrete actions
    action_size = env.get_rail_env().action_space[0]

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))

    # To conform with previous version where this option was not enabled
    if "fingerprint_type" not in train_params:
        train_params.fingerprint_type = FingerprintType.EPSILON_STEP

    # Double Dueling DQN policy
    if train_params.fingerprints and train_params.fingerprint_type in [FingerprintType.EPISODE,
                                                                       FingerprintType.EPSILON,
                                                                       FingerprintType.STEP]:
        # With single fingerprints
        policy = D3QNPolicy(env.state_size + 1, action_size, train_params)
    elif train_params.fingerprints and train_params.fingerprint_type in [FingerprintType.EPSILON_STEP,
                                                                         FingerprintType.EPSILON_EPISODE,
                                                                         FingerprintType.STEP_EPISODE]:
        # With double fingerprints
        policy = D3QNPolicy(env.state_size + 2, action_size, train_params)
    elif train_params.fingerprints and train_params.fingerprint_type is FingerprintType.EPSILON_STEP_EPISODE:
        # With triple fingerprints
        policy = D3QNPolicy(env.state_size + 3, action_size, train_params)
    else:
        # Without fingerprints
        policy = D3QNPolicy(env.state_size, action_size, train_params)

    if use_wandb:
        wandb.watch(policy.qnetwork_local)

    # Timers
    training_timer = Timer()
    step_timer = Timer()
    reset_timer = Timer()
    learn_timer = Timer()

    # TensorBoard writer
    tensorboard_logger = TensorBoardLogger(wandb.run.dir if use_wandb else train_params.tensorboard_path)

    ####################################################################################################################
    # Training starts
    training_timer.start()

    print("\nTraining {} trains on {}x{} grid for {} episodes.\n"
          .format(env_params.n_agents, env_params.x_dim, env_params.y_dim, train_params.n_episodes))

    agent_prev_obs = [None] * env_params.n_agents
    agent_prev_action = [2] * env_params.n_agents

    timestep = 0

    for episode in range(train_params.n_episodes + 1):
        # Reset timers
        step_timer.reset()
        reset_timer.reset()
        learn_timer.reset()
        frames_list=[]
        # Reset environment
        reset_timer.start()
        obs, info = env.reset()
       
        if train_params.fingerprints:
            obs = add_fingerprints(obs, env_params.n_agents, train_params.fingerprint_type, eps_start, timestep,
                                   episode)

        reset_timer.end()

        # Build agent specific observations
        for agent in range(env_params.n_agents):
            if obs[agent] is not None:
                agent_prev_obs[agent] = obs[agent].copy()

        # Run episode
        for step in range(max_steps):
            # Action dictionary to feed to step
            action_dict = dict()

            # Set used to track agents that didn't skipped the action
            agents_in_action = set()

            for agent in range(env_params.n_agents):
                # Create action mask
                action_mask = get_action_masking(env, agent, action_size, train_params)

                # Fill action dict
                # If agent is not arrived, moving between two cells or trapped in a deadlock (the latter is caught only
                # when the agent is moving in the deadlock triggering the second case)
                if info["action_required"][agent]:
                    # If an action is required, the actor predicts an action
                    agents_in_action.add(agent)
                    action_dict[agent] = policy.act(obs[agent], action_mask=action_mask, eps=eps_start)
                """
                Here it is not necessary an else branch to update the dict.
                By default when no action is given to RailEnv.step() for a specific agent, DO_NOTHING is assigned by the
                Flatland engine.
                """

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = env.step(action_dict)
            if train_params.fingerprints:
                next_obs = add_fingerprints(next_obs, env_params.n_agents, train_params.fingerprint_type, eps_start,
                                            timestep, episode)
            step_timer.end()

            for agent in range(env_params.n_agents):
                """
                Update memory and try to perform a learning step only when the agent has finished or when an action was 
                taken and thus relevant information is present, otherwise, for example when an agent is moving from a
                cell to another, the agent is ignored.
                """
                if agent in agents_in_action or (done[agent] and train_params.type == 1):
                    learn_timer.start()
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], obs[agent],
                                done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = obs[agent].copy()

                    # Agent shouldn't be in action_dict in order to print correctly the action's stats
                    if agent not in action_dict:
                        agent_prev_action[agent] = int(RailEnvActions.DO_NOTHING)
                    else:
                        agent_prev_action[agent] = action_dict[agent]

                if next_obs[agent] is not None:
                    obs[agent] = next_obs[agent]

            if train_params.render:
                env.env.show_render()

            timestep += 1
            # render_env(env.env.env_renderer,frames_list,show=True, show_observations=True,frames=True)
            if done["__all__"]:
                break

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Save checkpoints
        if "checkpoint_interval" in train_params and episode % train_params.checkpoint_interval == 0:
            if "save_model_path" in train_params:
                policy.save(train_params.save_model_path + "_ep_{}.pt".format(episode)
                            if "automatic_name_saving" in train_params and train_params.automatic_name_saving else
                            train_params.save_model_path)
        # Rendering
        if train_params.render:
            env.env.close()

        # Update total time
        training_timer.end()

        if train_params.print_stats:
            tensorboard_logger.update_tensorboard(env.env,
                                                  {"loss": policy.get_stat("loss"),
                                                   "q_expected": policy.get_stat("q_expected"),
                                                   "q_targets": policy.get_stat("q_targets"),
                                                   "eps": eps_start,
                                                   "memory_size": len(policy.memory)}
                                                  if policy.are_stats_ready() else {},
                                                  {"step": step_timer,
                                                   "reset": reset_timer,
                                                   "learn": learn_timer,
                                                   "train": training_timer})
        policy.reset_stats()

        # frames_list[0].save(f"flatland_dqn_agent_{episode}.gif", save_all=True, append_images=frames_list[1:], duration=7, loop=0)
    # plt.plot([i+1 for i in range(len(env.env.accumulated_normalized_score))],env.env.accumulated_normalized_score) 
    # plt.show()
    return env.env.accumulated_normalized_score, \
           env.env.accumulated_completion, \
           env.env.accumulated_deadlocks
           
