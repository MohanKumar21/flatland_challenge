from flatland.envs.rail_trainrun_data_structures import Waypoint
import numpy as np
import warnings

from typing import Tuple, Optional, NamedTuple, List

from attr import attr,attrs, attrib, Factory

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.timetable_utils import Line

from flatland.envs.step_utils.action_saver import ActionSaver
from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState
from flatland.envs.step_utils.malfunction_handler import MalfunctionHandler

Agent = NamedTuple('Agent', [
                            ('train_name',str),
                            ('train_id',int),
                            ('train_type',str),
                            ('train_max_speed',int),
                            ('train_length',int),
                            ('train_schedule',List[Tuple[int,int]]),
                            ('initial_position', Tuple[int, int]),
                             ('initial_direction', Grid4TransitionsEnum),
                             ('direction', Grid4TransitionsEnum),
                             ('target', Tuple[int, int]),
                             ('moving', bool),
                             ('earliest_departure', int),
                             ('latest_arrival', int),
                             ('handle', int),
                             ('position', Tuple[int, int]),
                             ('arrival_time', int),
                             ('old_direction', Grid4TransitionsEnum),
                             ('old_position', Tuple[int, int]),
                             ('speed_counter', SpeedCounter),
                             ('action_saver', ActionSaver),
                             ('state_machine', TrainStateMachine),
                             ('malfunction_handler', MalfunctionHandler),
                             ])


def load_env_agent(agent_tuple: Agent):
     return EnvAgent(
                        train_name= agent_tuple.train_name,
                        train_id= agent_tuple.train_id,
                        train_type=agent_tuple.train_type,
                        train_max_speed=agent_tuple.train_max_speed,
                        train_length=agent_tuple.train_length,
                        train_schedule=agent_tuple.train_schedule,
                        
                        initial_position = agent_tuple.initial_position,
                        initial_direction = agent_tuple.initial_direction,
                        direction = agent_tuple.direction,
                        target = agent_tuple.target,
                        moving = agent_tuple.moving,
                        earliest_departure = agent_tuple.earliest_departure,
                        latest_arrival = agent_tuple.latest_arrival,
                        handle = agent_tuple.handle,
                        position = agent_tuple.position,
                        arrival_time = agent_tuple.arrival_time,
                        old_direction = agent_tuple.old_direction,
                        old_position = agent_tuple.old_position,
                        speed_counter = agent_tuple.speed_counter,
                        action_saver = agent_tuple.action_saver,
                        state_machine = agent_tuple.state_machine,
                        malfunction_handler = agent_tuple.malfunction_handler,
                    )

@attrs
class EnvAgent:
    # INIT FROM HERE IN _from_line()
    initial_position = attrib(type=Tuple[int, int])
    initial_direction = attrib(type=Grid4TransitionsEnum)
    direction = attrib(type=Grid4TransitionsEnum)
    target = attrib(type=Tuple[int, int])
    moving = attrib(default=False, type=bool)

    # NEW : EnvAgent - Schedule properties
    earliest_departure = attrib(default=None, type=int)  # default None during _from_line()
    latest_arrival = attrib(default=None, type=int)  # default None during _from_line()

    handle = attrib(default=None)
    # INIT TILL HERE IN _from_line()

    # Env step facelift
    speed_counter = attrib(default = Factory(lambda: SpeedCounter(1.0)), type=SpeedCounter)
    action_saver = attrib(default = Factory(lambda: ActionSaver()), type=ActionSaver)
    state_machine = attrib(default= Factory(lambda: TrainStateMachine(initial_state=TrainState.WAITING)) ,
                           type=TrainStateMachine)
    malfunction_handler = attrib(default = Factory(lambda: MalfunctionHandler()), type=MalfunctionHandler)

    position = attrib(default=None, type=Optional[Tuple[int, int]])

    # NEW : EnvAgent Reward Handling
    arrival_time = attrib(default=None, type=int)

    # used in rendering
    old_direction = attrib(default=None)
    old_position = attrib(default=None)
# Rostoki 
# TrainName -Train Name
# TrainNumber - Integer (Train Number given by Indian Railways)
# TrainId - Integer (Train Index, uniquely identifying each train)
# TrainType - 0, 1, 2 (Train Type: 0 for super-fast, 1 for express, 2 for passenger)
# TrainMaxSpeed - (0, 150) km/hr (Maximum Speed of the train)
# TrainLength - (0, 2000) m (Train Length)
# TrainSchedule - vector (A sequence of tuples representing station, arrival time, and departure time)
    

    train_name= attrib(default=None,type=str)
    train_id=attrib(default=None,type=int)
    train_type=attrib(default=None,type=int)
    train_max_speed=attrib(default=None,type=int)
    train_length=attrib(default=None,type=int)
    train_schedule=attrib(default=None,type=List[Tuple[int,int,int]])


    def reset(self):
        """
        Resets the agents to their initial values of the episode. Called after ScheduleTime generation.
        """
        self.position = None
        # TODO: set direction to None
        self.direction = self.initial_direction
        self.old_position = None
        self.old_direction = None
        self.moving = False
        self.arrival_time = None

        self.malfunction_handler.reset()

        self.action_saver.clear_saved_action()
        self.speed_counter.reset_counter()
        self.state_machine.reset()


# Rostoki 
       

    def to_agent(self) -> Agent:
        return Agent(
                    train_name=self.train_name,
                    train_id=self.train_id,
                    train_type=self.train_type,
                    train_max_speed=self.train_max_speed,
                    train_length=self.train_length,
                    train_schedule=self.train_schedule,
                    
                    initial_position=self.initial_position,
                     initial_direction=self.initial_direction,
                     direction=self.direction,
                     target=self.target,
                     moving=self.moving,
                     earliest_departure=self.earliest_departure,
                     latest_arrival=self.latest_arrival,
                     handle=self.handle,
                     position=self.position,
                     old_direction=self.old_direction,
                     old_position=self.old_position,
                     speed_counter=self.speed_counter,
                     action_saver=self.action_saver,
                     arrival_time=self.arrival_time,
                     state_machine=self.state_machine,
                     malfunction_handler=self.malfunction_handler)

    def get_shortest_path(self, distance_map) -> List[Waypoint]:
        from flatland.envs.rail_env_shortest_paths import get_shortest_paths # Circular dep fix
        return get_shortest_paths(distance_map=distance_map, agent_handle=self.handle)[self.handle]

    def get_travel_time_on_shortest_path(self, distance_map) -> int:
        shortest_path = self.get_shortest_path(distance_map)
        if shortest_path is not None:
            distance = len(shortest_path)
        else:
            distance = 0
        speed = self.speed_counter.speed
        return int(np.ceil(distance / speed))

    def get_time_remaining_until_latest_arrival(self, elapsed_steps: int) -> int:
        return self.latest_arrival - elapsed_steps

    def get_current_delay(self, elapsed_steps: int, distance_map) -> int:
        '''
        +ve if arrival time is projected before latest arrival
        -ve if arrival time is projected after latest arrival
        '''
        return self.get_time_remaining_until_latest_arrival(elapsed_steps) - \
               self.get_travel_time_on_shortest_path(distance_map)


    @classmethod
    def from_line(cls, line: Line):
        """ Create a list of EnvAgent from lists of positions, directions and targets
        """
        num_agents = len(line.agent_positions)

        agent_list = []
        for i_agent in range(num_agents):
            speed = line.agent_speeds[i_agent] if line.agent_speeds is not None else 1.0

            agent = EnvAgent(initial_position = line.agent_positions[i_agent],
                            initial_direction = line.agent_directions[i_agent],
                            direction = line.agent_directions[i_agent],
                            target = line.agent_targets[i_agent],
                            moving = False,
                            earliest_departure = None,
                            latest_arrival = None,
                            handle = i_agent,
                            speed_counter = SpeedCounter(speed=speed))
            agent_list.append(agent)

        return agent_list

    @classmethod
    def load_legacy_static_agent(cls, static_agents_data: Tuple):
        agents = []
        for i, static_agent in enumerate(static_agents_data):
            if len(static_agent) >= 6:
                agent = EnvAgent(initial_position=static_agent[0], initial_direction=static_agent[1],
                                direction=static_agent[1], target=static_agent[2], moving=static_agent[3],
                                speed_counter=SpeedCounter(static_agent[4]['speed']), handle=i)
            else:
                agent = EnvAgent(initial_position=static_agent[0], initial_direction=static_agent[1],
                                direction=static_agent[1], target=static_agent[2],
                                moving=False,
                                speed_counter=SpeedCounter(1.0),
                                handle=i)
            agents.append(agent)
        return agents

    def __str__(self):
        return f"\n \
                 handle(agent index): {self.handle} \n \
                 initial_position: {self.initial_position}  \n \
                 initial_direction: {self.initial_direction} \n \
                 position: {self.position}  \n \
                 direction: {self.direction}  \n \
                 target: {self.target} \n \
                 old_position: {self.old_position} \n \
                 old_direction {self.old_direction} \n \
                 earliest_departure: {self.earliest_departure}  \n \
                 latest_arrival: {self.latest_arrival} \n \
                 state: {str(self.state)} \n \
                 malfunction_handler: {self.malfunction_handler} \n \
                 action_saver: {self.action_saver} \n \
                 speed_counter: {self.speed_counter}"

    @property
    def state(self):
        return self.state_machine.state

    @state.setter
    def state(self, state):
        self._set_state(state)

    def _set_state(self, state):
        warnings.warn("Not recommended to set the state with this function unless completely required")
        self.state_machine.set_state(state)

    @property
    def malfunction_data(self):
        raise ValueError("agent.malunction_data is deprecated, please use agent.malfunction_hander instead")

    @property
    def speed_data(self):
        raise ValueError("agent.speed_data is deprecated, please use agent.speed_counter instead")




