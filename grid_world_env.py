import collections
import cv2
import numpy as np
import copy
import itertools
from cached_property import cached_property
import curses
from gym import spaces


class Terrain(object):
  def __init__(
      self,
      name,
      symbol,
      passable,
      reward,
      is_terminal,
      color=(255,255,255),
      **kwargs
  ):
    self.name = name
    self.symbol = symbol
    self.passable = passable
    self.reward = reward
    self.is_terminal = is_terminal
    self.color = color
    self.other_properties = kwargs

  def modify_env(self, env):
    pass


class Food(Terrain):
  def __init__(
      self,
      **kwargs
  ):
    init_kwargs = dict(
      name="food",
      symbol='.',
      passable=True,
      reward=1,
      is_terminal=False,
      color=(255, 255, 255),
    )
    init_kwargs.update(kwargs)
    super().__init__(**init_kwargs)

  def modify_env(self, env):
    """
    The current pixel becomes a ground
    """
    assert isinstance(env, GridWorldEnv)
    for index, terrain in env.grid_spec.items():
      if terrain.name == "ground":
        env.grid[env.agent_pos] = index
        break


class Agent(Terrain):
  def __init__(
      self,
      **kwargs
  ):
    init_kwargs = dict(
      name="agent",
      symbol='A',
      passable=False,
      reward=0,
      is_terminal=False,
      color=(0,255,255),
    )
    init_kwargs.update(kwargs)
    super().__init__(**init_kwargs)


class Ground(Terrain):
  def __init__(
      self,
      **kwargs
  ):
    init_kwargs = dict(
      name="ground",
      symbol=' ',
      passable=True,
      reward=0,
      is_terminal=False,
    )
    init_kwargs.update(kwargs)
    super().__init__(**init_kwargs)


class SpawnLocation(Terrain):
  def __init__(self, **kwargs):
    init_kwargs = dict(
      name="spawn",
      symbol='S',
      passable=True,
      reward=0,
      is_terminal=False,
    )
    init_kwargs.update(kwargs)
    super().__init__(**init_kwargs)


class Goal(Terrain):
  def __init__(
      self,
      reward,
      **kwargs
  ):
    init_kwargs = dict(
      name="goal",
      symbol='G',
      passable=True,
      reward=reward,
      is_terminal=True,
    )
    init_kwargs.update(kwargs)
    super().__init__(**init_kwargs)


class Wall(Terrain):
  def __init__(
      self,
      **kwargs
  ):
    init_kwargs = dict(
      name="wall",
      symbol='#',
      passable=False,
      reward=0,
      is_terminal=False,
    )
    init_kwargs.update(kwargs)
    super().__init__(**init_kwargs)


class GridWorldEnv:
  """
  State: the current grid, represented as (h, w, c), where c corresponds to
      the terrain index / agent.
  Action: noop, up, down, left, right
  Reward, terminal: depends on grid spec

  grid:
      a 2D int array showing terrain locations
      (i, j): i-th row, j-th column; indexed from (0,0) from top-left
      height = n_row, width = n_col
  the agent overwrites the existing terrain
  e.g.
  'A' = agent
  'G' = goal (terminal and gets reward)
  '#' = wall
  ' ' = free terrain
  '.' = collectable reward (food)
  #####
  #  G#
  #  .#
  #A  #
  #####

  grid_spec: a dict() mapping terrain index to a terrain object
      The indices must be 0, 1, ..., N-1 (consecutive).
      No new terrains should appear during rollouts, or otherwise the state
      representation can change.

  terrain:
  In general, a terrain determines how the grid changes if the agent steps
  onto it. We do not consider global changes yet.
  """
  def __init__(
      self,
      init_grid,
      grid_spec,
      init_agent_pos,
      name="grid_world",
      obs_format="NCHW",
      rendered_block_size=80,
      render_type="opencv",
      find_subgraph=False,
  ):
    """

    :param init_grid:
    :param grid_spec:
    :param init_agent_pos:
    :param name:
    :param obs_format:
    :param rendered_block_size: size (in pixels) of a rendered block when calling self.render()
    """
    self.init_params = locals()
    self.init_params.pop("self")
    self.name = name
    assert all([
      isinstance(init_grid, np.ndarray),
      init_grid.dtype in [np.uint8, int],
      len(init_grid.shape) == 2,
      ])
    self._init_grid = init_grid
    self.grid = np.copy(init_grid)  # must not use shallow copy!

    assert all([
      isinstance(init_agent_pos, tuple),
      len(init_agent_pos) == 2,
      init_agent_pos[0] in range(self.grid_height),
      init_agent_pos[1] in range(self.grid_width),
      ])
    self._init_agent_pos = init_agent_pos
    self.agent_pos = init_agent_pos

    self.grid_spec = grid_spec
    indices = self.all_terrain_indices
    assert all([
      isinstance(i, int) or isinstance(i, np.uint8)
      for i in indices
    ])
    assert len(indices) == ((indices[-1] - indices[0]) + 1)

    assert obs_format in ["NHWC", "NCHW"]
    self.obs_format = obs_format

    self.rendered_block_size = rendered_block_size
    self.render_type = render_type
    self._stdsrc = None
    self.find_subgraph = find_subgraph

    self.action_list = [
      "noop", "up", "down", "left", "right"
    ]
    self._reward = 0
    self.unpicklable_list = []

  @property
  def grid_height(self):
    return self.grid.shape[0]

  @property
  def grid_width(self):
    return self.grid.shape[1]

  @property
  def all_terrain_indices(self):
    indices = sorted(self.grid_spec.keys())
    return indices

  def reset(self):
    self.grid = np.copy(self._init_grid)
    self.agent_pos = self._init_agent_pos
    return self.obs

  @property
  def obs(self):
    """
    Shape: (height, width, channel)
    Each channel corresponding to a terrain
    Try print(np.transpose(cur_state, (2,0,1))) to visualize the state
    :return:
    """
    M = int(np.prod(self.grid.shape))  # number of square tiles
    N = len(self.grid_spec.keys()) + 1  # number of terrain types + agent itself

    s_1d = np.zeros((M, N), dtype=np.uint8)  # flattened one-hot representation
    grid_copy = self.grid.copy()
    grid_copy[self.agent_pos[0], self.agent_pos[1]] = N - 1  # agent location
    grid_1d = grid_copy.ravel()
    s_1d[np.arange(M), grid_1d] = 1
    obs = s_1d.reshape(self.grid.shape + (N,)).astype(np.float32)  # 2D one-hot representation

    if self.obs_format == "NHWC":
      obs_permuted = obs
    elif self.obs_format == "NCHW":
      obs_permuted = np.transpose(obs, (2, 0, 1))
    else:
      raise NotImplementedError
    return obs_permuted

  @cached_property
  def observation_space(self):
    return spaces.Box(low=0, high=1, dtype=np.float32, shape=self.obs.shape)

  @property
  def action_space(self):
    return spaces.Discrete(len(self.action_list))

  @property
  def state(self):
    return self.grid, self.agent_pos

  @property
  def init_state(self):
    return self._init_grid, self._init_agent_pos

  def set_state(self, state):
    grid, agent_pos = state
    self.grid = grid
    self.agent_pos = agent_pos

  @property
  def reward(self):
    return self._reward

  @property
  def is_terminal(self):
    i, j = self.agent_pos
    cur_terrain = self.grid_spec[self.grid[i, j]]
    return cur_terrain.is_terminal

  @property
  def n_action(self):
    return len(self.action_list)

  def step(self, action):
    assert not self.is_terminal
    s_action = self.action_list[action] # semantic action
    i0, j0 = self.agent_pos

    if s_action == "noop":
      i1, j1 = i0, j0
    elif s_action == "up":
      i1, j1 = i0 - 1, j0
    elif s_action == "down":
      i1, j1 = i0 + 1, j0
    elif s_action == "left":
      i1, j1 = i0, j0 - 1
    elif s_action == "right":
      i1, j1 = i0, j0 + 1
    else:
      raise NotImplementedError

    new_terrain = self.grid_spec[self.grid[i1, j1]]
    if new_terrain.passable:
      self.agent_pos = (i1, j1)
      new_terrain.modify_env(self)

    self._reward = new_terrain.reward
    return self.obs, self._reward, self.is_terminal, {}

  @staticmethod
  def render_grid(grid, grid_spec, rendered_block_size):
    # show in an opencv window for interactive control
    def gen_one_block(
        text,
        size=rendered_block_size,
        fontScale=0.5,
        background_color=(0, 0, 0),
        border_color=(128, 128, 128),
        border_size=1,
        text_color=(255, 255, 255),
    ):
      block = np.tile(border_color, (size, size, 1)).astype(np.uint8)
      bs = border_size
      nbs = size - 2 * border_size
      block[bs: -bs, bs: -bs] = np.tile(background_color, (nbs, nbs, 1))
      cv2.putText(
        block,text,
        org=(int(size * 0.5), int(size*0.5)),fontFace=0,fontScale=fontScale,
        color=text_color
      )
      return block

    n_row, n_col = grid.shape
    index_color = (0, 255, 255)
    background_color = (0, 0, 0)

    empty = gen_one_block(text="",text_color=(0, 0, 0), border_color=(0, 0, 0))
    img_grid = []
    img_row_0 = [empty]
    for col in range(n_col):
      img_row_0.append(gen_one_block(text="%d" % col,text_color=index_color, border_color=(0, 0, 0)))
    img_grid.append(img_row_0)

    for row in range(n_row):
      img_row = []
      img_row.append(gen_one_block(text="%d" % row,text_color=index_color, border_color=(0, 0, 0)))
      for col in range(n_col):
        terrain = grid_spec[grid[row, col]]
        img_row.append(gen_one_block(
          text=terrain.symbol,
          text_color=terrain.color,
          background_color=background_color,
        ))
      img_grid.append(img_row)
    I = organize_imgs(
      img_grid=img_grid,
      border_size=0,
      border_color=(0,0,0),
      scale=1.,
    )
    return I

  @staticmethod
  def get_grid_spec_with_agent(grid_spec):
    grid_spec_with_agent = copy.deepcopy(grid_spec)
    M = len(grid_spec.keys())
    agent_index = M + 1
    grid_spec_with_agent[agent_index] = Agent()
    return grid_spec_with_agent

  @staticmethod
  def get_grid_with_agent(grid, grid_spec, agent_pos):
    grid_with_agent = np.copy(grid)
    agent_index = len(grid_spec.keys()) + 1
    grid_with_agent[agent_pos] = agent_index
    return grid_with_agent

  def render(self, window_name=None, close=False):
    grid = GridWorldEnv.get_grid_with_agent(self.grid, self.grid_spec, self.agent_pos)
    grid_spec=GridWorldEnv.get_grid_spec_with_agent(self.grid_spec)

    if self.render_type == "opencv":
      if window_name is None:
        window_name = self.name
      else:
        assert isinstance(window_name, str)
      I = GridWorldEnv.render_grid(
        grid=grid, grid_spec=grid_spec,
        rendered_block_size=self.rendered_block_size,
      )
      cv2.namedWindow(window_name)
      cv2.imshow(window_name, I)
      cv2.waitKey(1)
      if close:
        cv2.destroyWindow(window_name)

    elif self.render_type == "console":
      map_string = GridWorldEnv.grid_to_symbolic(grid, grid_spec)
      title_string = self.name
      render_string = "\n".join([title_string, map_string])
      if self._stdsrc is None:
        self._stdsrc = curses.initscr()
        curses.noecho()
        curses.cbreak()
      lines = render_string.split('\n')
      for i, line in enumerate(lines):
        self._stdsrc.addstr(i, 0, line)
      self._stdsrc.refresh()

      if close:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
        self._stdsrc = None
    elif self.render_type == "print":
      map_string = GridWorldEnv.grid_to_symbolic(grid, grid_spec)
      title_string = self.name
      render_string = "\n".join([title_string, map_string])
      print(render_string)
    else:
      raise NotImplementedError

  def simulate(self):
    is_terminal = False
    total_reward = 0.
    while True:
      if is_terminal:
        print("Finished with total reward %g" % total_reward)
        self.reset()
        is_terminal = False
        total_reward = 0.
      self.render()
      key = cv2.waitKey(-1)
      s_action = GridWorldEnv.key_to_semantic_action(key)
      if s_action == "quit":
        print("Quit")
        break
      if not is_terminal:
        action = self.action_list.index(s_action)
        next_state, reward, is_terminal, env_info = self.step(action)
        if reward != 0:
          print("Reward: %g" % reward)
          total_reward += reward
    self.render(close=True)

  @staticmethod
  def key_to_semantic_action(key):
    """
    Maps one key to the semantic action
    """
    D = {
      ord('w'): "up",
      ord('s'): "down",
      ord('a'): "left",
      ord('d'): "right",
      ord('q'): "quit",
    }
    if key in D:
      s_action = D[key]
    else:
      s_action = "noop"
    return s_action

  @staticmethod
  def parse_symbolic_grid(symbolic_grid, grid_spec):
    """
    A convenient way to parse symbolic grid to numeric grid

    symbolic_grid: a string consisting of symbols and delimeters ('\n')
    Do not call this method repeatedly. It is inefficient.
    """
    symbol_to_index_dict = {}
    for index, terrain in grid_spec.items():
      symbol_to_index_dict[terrain.symbol] = index

    symbolic_rows = symbolic_grid.split('\n')
    # remove the first and last empty lines
    symbolic_rows = symbolic_rows[1:-1]
    # remove spaces padded to the left and right
    min_n_left_space = min([len(row) - len(row.lstrip(' ')) for row in symbolic_rows])
    min_n_right_space = min([len(row) - len(row.rstrip(' ')) for row in symbolic_rows])
    row_end = - min_n_right_space if min_n_right_space > 0 else None
    symbolic_rows = [row[min_n_left_space: row_end] for row in symbolic_rows]

    height = len(symbolic_rows)
    width = len(symbolic_rows[0])

    grid = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
      for j in range(width):
        try:
          symbol = symbolic_rows[i][j]
          grid[i,j] = symbol_to_index_dict[symbol]
        except:
          import pdb; pdb.set_trace()
    return grid

  @staticmethod
  def grid_to_symbolic(grid, grid_spec):
    symbolic_grid = ""
    height, width = grid.shape

    for i in range(height):
      row = ""
      for j in range(width):
        terrain_index = grid[i, j]
        terrain = grid_spec[terrain_index]
        symbol = terrain.symbol
        row += symbol
      row += "\n"
      symbolic_grid += row
    return symbolic_grid

  @property
  def all_terrain_classes(self):
    return set(terrain.__class__ for terrain in self.grid_spec.values())

  @staticmethod
  def get_terrain_index(grid_spec, terrain_class):
    return [k for k, v in grid_spec.items() if isinstance(v, terrain_class)][0]

  def update_params(self, global_vars, training_args):
    pass

  def set_seed(self, seed):
    pass

  def prepare_sharing(self):
    self.share_params = dict()


class FoodPickupEnv(GridWorldEnv):
  def __init__(
      self,
      symbolic_grid,
      init_grid=None,  # if provided, overrides symbolic_grid
      name="food_pickup_world",
      obs_format="NCHW",
      rendered_block_size=80,
      goal_reward=10,
      food_reward=1,
  ):
    self.goal_reward = goal_reward
    self.food_reward = food_reward

    terrain_list = [Ground(), Wall(), Goal(reward=goal_reward), Food(reward=food_reward), Agent()]
    ground_index = 0
    agent_index = len(terrain_list) - 1
    grid_spec = {i: terrain for i, terrain in enumerate(terrain_list)}
    if init_grid is None:
      init_grid = GridWorldEnv.parse_symbolic_grid(
        symbolic_grid=symbolic_grid,
        grid_spec=grid_spec,
      )
    xx, yy = np.where(init_grid == agent_index)
    assert len(xx) == len(yy) == 1
    init_agent_pos = (xx[0], yy[0])
    init_grid[init_agent_pos[0], init_agent_pos[1]] = ground_index
    grid_spec.pop(agent_index)
    super().__init__(init_grid=init_grid, grid_spec=grid_spec, init_agent_pos=init_agent_pos,
                     name=name, obs_format=obs_format, rendered_block_size=rendered_block_size)

  @cached_property
  def all_possible_agent_pos(self):
    passable_terrain_classes = [terrain.__class__ for terrain in self.grid_spec.values() if terrain.passable]
    passable_terrain_indices = [GridWorldEnv.get_terrain_index(self.grid_spec, cls) for cls in passable_terrain_classes]
    possible_agent_pos_list = [
      (x, y) for x, y in itertools.product(range(self.grid.shape[0]), range(self.grid.shape[1]))
      if self._init_grid[x, y] in passable_terrain_indices
    ]
    return possible_agent_pos_list

  @cached_property
  def food_index(self):
    return GridWorldEnv.get_terrain_index(self.grid_spec, Food)

  @cached_property
  def ground_index(self):
    return GridWorldEnv.get_terrain_index(self.grid_spec, Ground)

  @cached_property
  def init_food_pos_list(self):
    return sorted(list(zip(*np.where(self._init_grid == self.food_index))))

  def all_food_status_to_grid(self, all_food_status):
    grid = np.copy(self._init_grid)
    for food_pos, food_status in zip(self.init_food_pos_list, all_food_status):
      x, y = food_pos
      if food_status == 0:
        grid[x, y] = self.ground_index
      else:
        grid[x, y] = self.food_index
    return grid

  @cached_property
  def all_possible_states(self):
    """
    Enumerate over all agent positions at passable terrains
    Enumerate over mutable terrains (Food --> Ground)
    This overcounts all possible states attainable from the init_agent_pos, because the agent may not get to
      a position without getting food in the way
    :return:
    """
    states = []
    for agent_pos in self.all_possible_agent_pos:
      all_food_status_lists = [
        [0] if food_pos == agent_pos else [0, 1]
        for food_pos in self.init_food_pos_list
      ]
      for all_food_status in itertools.product(*all_food_status_lists):
        states.append((self.all_food_status_to_grid(all_food_status), agent_pos))
    self.reset()
    return states

  def state_to_vertex(self, state):
    """
    Based on the init grid, obtain a tuple (hashable) representation of the state
    """
    grid, agent_pos = state
    assert self.all_terrain_classes <= {Ground, Wall, Goal, Food}
    food_index = GridWorldEnv.get_terrain_index(self.grid_spec, Food)
    init_food_pos_list = sorted(list(zip(*np.where(self._init_grid == food_index))))
    all_food_status = tuple(
      int(grid[pos] == food_index)
      for pos in init_food_pos_list
    )
    return agent_pos, all_food_status

  @cached_property
  def all_possible_vertices(self):
    return [self.state_to_vertex(state) for state in self.all_possible_states]

  def vertex_to_state(self, vertex):
    assert vertex in self.all_possible_vertices
    agent_pos, all_food_status = vertex
    grid = self.all_food_status_to_grid(all_food_status)
    state = (grid, agent_pos)
    return state


  def compute_all_vertex_obs_pairs_from_vertices(self, vertices):
    pairs = []
    for vertex in vertices:
      state = self.vertex_to_state(vertex)
      self.set_state(state)
      obs = np.copy(self.obs)
      pairs.append((vertex, obs))
    self.reset()
    return pairs


def organize_imgs(img_grid, interpolation=cv2.INTER_LINEAR,
                  border_size=10, border_color=(0, 0, 0), scale=1.):
    """
    img_grid: a list of lists of images, or a 2D array of images (same shape)
    """
    img1 = img_grid[0][0]
    n_row = len(img_grid)
    n_col = len(img_grid[0])
    assert all([n_col == len(img_grid[i]) for i in range(n_row)]), [len(row) for row in img_grid]

    if len(img1.shape) == 2:
        assert isinstance(border_color, int)
    elif len(img1.shape) == 3:
        assert len(border_color) == 3
    else:
        raise NotImplementedError
    h1, w1 = img1.shape[:2]
    row_length = border_size * (n_col - 1) + w1 * n_col

    if isinstance(border_color, int):
        col_border = (np.ones((h1, border_size)) * border_color).astype(img1.dtype)
        row_border = (np.ones((border_size, row_length)) * border_color).astype(img1.dtype)
    elif isinstance(border_color, tuple) and len(border_color) == 3:
        col_border = np.tile(border_color, (h1, border_size, 1)).astype(img1.dtype)
        row_border = np.tile(border_color, (border_size, row_length, 1)).astype(img1.dtype)
    else:
        raise NotImplementedError

    all_img_list = []
    for img_row in img_grid:
        row_img_list = [img_row[0]]
        for img in img_row[1:]:
            assert img.shape == img1.shape
            row_img_list += [col_border, img]
        I_row = np.concatenate(row_img_list, axis=1)
        all_img_list += [I_row, row_border]
    I = np.concatenate(all_img_list, axis=0)

    w, h = I.shape[:2]
    I = cv2.resize(I, (int(scale * h), int(scale * w)), interpolation=interpolation)
    return I

# env generator -------------------------------------------------------------------------------------------------------
EnvSpec = collections.namedtuple("env_spec", ["symbolic_grid", "food_reward", "goal_reward"])
test_env_specs = {
  '000': EnvSpec(
    symbolic_grid="""
      #####
      #  G#
      #   #
      #A .#
      #####
    """,
    food_reward=1,
    goal_reward=10,
  ),
  '000a': EnvSpec(
    symbolic_grid="""
      #####
      #. G#
      #   #
      #A  #
      #####
    """,
    food_reward=1,
    goal_reward=10,
  ),
  '000b': EnvSpec(
    symbolic_grid="""
      #####
      #. G#
      # . #
      #A .#
      #####
    """,
    food_reward=1,
    goal_reward=10,
  ),
  '000c': EnvSpec(
    symbolic_grid="""
      #####
      #..G#
      #.. #
      #A .#
      #####
    """,
    food_reward=1,
    goal_reward=10,
  ),

  '001': EnvSpec(
    symbolic_grid="""
      ##################
      #. G# .###########
      #   #  ###########
      #   #  ###########
      #              . #
      #A               #
      ##################
    """,
    food_reward=10,
    goal_reward=1,
  ),


  '002': EnvSpec(
    symbolic_grid="""
      ##################
      #. .# .###########
      #   # .###########
      #   # .###########
      #   #     #      #
      #A  #     #      #
      #   #  #  #  #   #
      #   #  #  #  #   #
      #   #  #  #  #   #
      #   #  #  #  #   #
      #      #     #   #
      #      #    .# G #
      ##################
    """,
    food_reward=1,
    goal_reward=100,
  ),

  '003': EnvSpec(
    symbolic_grid="""
    #######
    ###.###
    ### ###
    #. A .#
    ### ###
    ###G###
    #######
    """,
    food_reward=1,
    goal_reward=1,
  ),

  '004': EnvSpec(
    symbolic_grid="""
    ###################
    ##.# #.# #.# #.#.##
    #.A              G#
    ###################
    """,
    food_reward=1,
    goal_reward=1,
  ),
  '004a': EnvSpec(
    symbolic_grid="""
    ###################
    ##.#G#.############
    #.A    .###########
    ###################
    """,
    food_reward=1,
    goal_reward=1,
  ),
  '004b': EnvSpec(
    symbolic_grid="""
    ###################
    ##.#G#.#.##########
    #.A      .#########
    ###################
    """,
    food_reward=1,
    goal_reward=1,
  ),
  '004c': EnvSpec(
    symbolic_grid="""
    ###################
    ##.# # # #G########
    #.A        .#######
    ###################
    """,
    food_reward=1,
    goal_reward=1,
  ),
  '004d': EnvSpec(
    symbolic_grid="""
    ###################
    ##G# #.# #.# ######
    #.A          .#####
    ###################
    """,
    food_reward=1,
    goal_reward=1,
  ),
  '005a': EnvSpec(
    symbolic_grid="""
    ###########
    #A# # #.# #
    #         #
    # # # # # #
    #         #
    # #.# # #G#
    ###########
    """,
    food_reward=1,
    goal_reward=1,
  ),
}


def get_test_env(version):
  name = "grid_world_" + version
  spec = test_env_specs[version]
  kwargs = spec._asdict()
  env = FoodPickupEnv(name=name, **kwargs)
  return env


def test_simulation():
  print('Env simulating in a separate window')
  env = get_test_env('001')
  env.simulate()


if __name__ == "__main__":
  test_simulation()
