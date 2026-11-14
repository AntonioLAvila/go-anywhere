"""MDP terms for go-anywhere.

Re-exports mjlab's generic and velocity-task terms so task configs can use a
single ``mdp`` namespace, then adds the goal-task terms defined here.
"""

from mjlab.envs.mdp import *  # noqa: F401,F403
from mjlab.tasks.velocity.mdp import *  # noqa: F401,F403

from go_anywhere.mdp import contact as contact
from go_anywhere.mdp.cells import CellLabels, get_cell_labels
from go_anywhere.mdp.commands import COMMAND_DIM, GoalCommand, GoalCommandCfg
from go_anywhere.mdp.contact import (
  IMPACT_ONSET_BW,
  IMPACT_TERMINATE_BW,
  feet_edge,
  foot_impact,
  foot_impact_exceeded,
  peak_foot_impact_bw,
  penetration_cost,
  stumble,
  trunk_contact,
  world_contact_cost,
)
from go_anywhere.mdp.curriculums import (
  command_speed_stages,
  context_length_stages,
  event_param_stages,
  terrain_levels_goal,
)
from go_anywhere.mdp.events import (
  reset_airborne,
  reset_fallen,
  reset_from_reference_states,
)
from go_anywhere.mdp.observations import (
  base_height_above_feet,
  foot_contact_forces_b,
  task_id,
)
from go_anywhere.mdp.rewards import (
  cost_of_transport,
  dont_wait,
  goal_arrival,
  goal_progress,
  heading_tracking,
  mechanical_power,
)
from go_anywhere.mdp.terminations import goal_timeout, route_complete

__all__ = [
  "contact",
  "CellLabels",
  "get_cell_labels",
  "COMMAND_DIM",
  "GoalCommand",
  "GoalCommandCfg",
  "IMPACT_ONSET_BW",
  "IMPACT_TERMINATE_BW",
  "feet_edge",
  "foot_impact",
  "foot_impact_exceeded",
  "peak_foot_impact_bw",
  "penetration_cost",
  "stumble",
  "trunk_contact",
  "world_contact_cost",
  "command_speed_stages",
  "context_length_stages",
  "event_param_stages",
  "terrain_levels_goal",
  "reset_airborne",
  "reset_fallen",
  "reset_from_reference_states",
  "base_height_above_feet",
  "foot_contact_forces_b",
  "task_id",
  "cost_of_transport",
  "dont_wait",
  "goal_arrival",
  "goal_progress",
  "heading_tracking",
  "mechanical_power",
  "goal_timeout",
  "route_complete",
]
