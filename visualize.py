import importlib
import flatland
importlib.reload(flatland)
import numpy as np
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent, load_env_agent
#from environments.custom_rail_generator import simple_rail_generator
#from environments.custom_schedule_generator import sparse_schedule_generator
#from environments.observations import TreeObsForRailEnv
#from environments.visualization_utils import animate_env, get_patch, render_env
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.core.grid.grid4_utils import get_new_position
#from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.step_utils.states import TrainState


from flatland.utils.rendertools import RenderTool, AgentRenderVariant


import numpy as np
import time
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv


n_agents = 3
x_dim = 100
y_dim = 100
n_cities =3
max_rails_between_cities = 30
max_rails_in_city = 30
seed = 0
malfunction_rate = 1 / 200
n_episodes = 100

malfunction_parameters = MalfunctionParameters(
    malfunction_rate=malfunction_rate,
    min_duration=20,
    max_duration=50
)

# Different agent types (trains) with different speeds.
# speed_ration_map = {
#     1.: 1.0,  # 100% of fast passenger train
#     1. / 2.: 0.0,  # 0% of fast freight train
#     1. / 3.: 0.0,  # 0% of slow commuter train
#     1. / 4.: 0.0  # 0% of slow freight train
# }

observation_tree_depth = 2
observation_tree_depth = 2

observation_builder = TreeObsForRailEnv(max_depth=observation_tree_depth)

env1 = RailEnv(
    width=x_dim,
    height=y_dim,
    rail_generator=sparse_rail_generator(
        max_num_cities=n_cities,
        seed=seed,

        grid_mode=False,
        max_rails_between_cities=max_rails_between_cities,
        # max_rail_pairs_in_city=max_rails_in_city,
        city_positions=[(40,6),(15,16),(25,45)],
        platforms={0:5,1:5,2:7}
    ),
    line_generator=sparse_line_generator(),
    number_of_agents=n_agents,
#     malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
    obs_builder_object=observation_builder
)
env1.reset()
import PIL
from flatland.utils.rendertools import RenderTool
from IPython.display import clear_output


# Render the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
env_renderer = RenderTool(env1, gl="PILSVG",
                        #   agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
  
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          screen_height=726,
                          screen_width=1240)
env1.reset()
env_renderer.reset()
import matplotlib.pyplot as plt
def render_env(env_renderer, show=False, frames=False, show_observations=True):
    """
    Renders the current state of the environment
    """
    env_renderer.render_env(show=show,show_agents=True, frames=frames, show_observations=show_observations)
    image = env_renderer.gl.get_image()
    plt.figure(figsize=(image.shape[1] / 72.0, image.shape[0] / 72.0), dpi = 72)
    plt.axis("off")
    plt.imshow(image)
    plt.show()
# print(env1.rail.grid)
render_env(env_renderer)
from flatland.graphs.graph_utils import * 
graph=RailEnvGraph(env1)
print(graph.G)
