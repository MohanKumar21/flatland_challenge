import random

import numpy as np
import torch

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.common.action_skipping_masking import get_action_masking
from src.common.flatland_railenv import FlatlandRailEnv
from src.psppo.policy import PsPPOPolicy
from src.psppo.ps_ppo_flatland import get_agent_ids
import matplotlib.pyplot as plt 
import PIL
def render_env(env_renderer,frames_list, show=False, frames=False, show_observations=True,):
    """
    Renders the current state of the environment
    """
    env_renderer.render_env(show=show, frames=frames, show_observations=show_observations)
    image = env_renderer.gl.get_image()
    frames_list.append(PIL.Image.fromarray(env_renderer.gl.get_image()))
def eval_policy(env_params, train_params):

    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth

    n_eval_episodes = train_params.eval_episodes

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

    # + 1 because the agent id is used as input in the neural network
    ppo = PsPPOPolicy(env.state_size + 1,
                      action_size,
                      train_params,
                      env_params.n_agents)

    print("\nEvaluating {} trains on {}x{} grid for {} episodes.\n"
          .format(env_params.n_agents, env_params.x_dim, env_params.y_dim, n_eval_episodes))
    
    for episode_idx in range(n_eval_episodes):
        frames_list=[]
        prev_obs, info = env.reset()

        done = {a: False for a in range(env_params.n_agents)}
        done["__all__"] = all(done.values())

        agent_ids = get_agent_ids(env.get_rail_env().agents, env_params.malfunction_parameters.malfunction_rate)

        for step in range(max_steps):

            # Action dictionary to feed to step
            action_dict = dict()

            # Flag to control the last step of an episode
            is_last_step = step == (max_steps - 1)

            for agent in prev_obs:

                # Create action mask
                action_mask = get_action_masking(env, agent, action_size, train_params)

                if info["action_required"][agent]:
                    # If an action is required, the actor predicts an action and the obs, actions, masks are stored
                    action_dict[agent] = ppo.act(np.append(prev_obs[agent], [agent_ids[agent]]),
                                                 action_mask, agent_id=agent)

            next_obs, rewards, done, info = env.step(action_dict)
          
            # print(env.flatland_env.agents)
            for a in range(env_params.n_agents):
                
                if not done[a]:
                    prev_obs[a] = next_obs[a].copy()
            render_env(env.env.env_renderer,frames_list,show=True, show_observations=False,frames=True)
            # if train_params.render:
            #     env.env.show_render()

            if done['__all__']:
                break
    
        # if train_params.render:
        #     env.env.close()

    frames_list[0].save(f"flatland_ppo_agent_{episode_idx}.gif", save_all=True, append_images=frames_list[1:], duration=7, loop=0)
