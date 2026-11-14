"""Base goal-reaching environment for the Go2.

Adapted from mjlab's ``make_velocity_env_cfg`` + Go1 overrides, with:
- the ``GoalCommand`` interface instead of velocity commands,
- composable observation groups with history (``proprio_hist`` 25 x 50 Hz,
  ``terrain_hist`` 50 x 50 Hz strided to 10 Hz in the model, ``terrain_priv``,
  ``command``, ``privileged``, ``task_id``),
- the unified world-contact scheme (``mdp.contact``),
- 50 Hz PD joint-position control (0.005 s x decimation 4).
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from go_anywhere import mdp
from go_anywhere.assets.go2 import (
  BASE_BODY,
  FOOT_GEOMS,
  FOOT_SITES,
  GO2_ACTION_SCALE,
  NON_FOOT_GEOMS,
  TRUNK_GEOMS,
  get_go2_robot_cfg,
)

PROPRIO_HISTORY = 25
"""0.5 s at 50 Hz."""
TERRAIN_HISTORY = 50
"""1.0 s buffer at 50 Hz; the model takes every 5th frame (10 Hz)."""
TERRAIN_SCAN_SIZE = (1.6, 1.0)
TERRAIN_SCAN_RES = 0.1
TERRAIN_SCAN_MAX = 5.0

CONTEXT_STAGES = [
  {"step": 0, "visible_context": 8},
  {"step": 1500 * 24, "visible_context": 16},
  {"step": 3000 * 24, "visible_context": PROPRIO_HISTORY},
]


def make_go_anywhere_env_cfg(
  terrain_generator: TerrainGeneratorCfg | None,
  play: bool = False,
  *,
  trunk_contact_terminates: bool = True,
  foot_impact_terminates: bool = False,
  goal_timeout_terminates: bool = True,
  world_contact_weight: float = -1.0,
  episode_length_s: float = 60.0,
) -> ManagerBasedRlEnvCfg:
  """Build the base env cfg. ``terrain_generator=None`` gives a flat plane."""

  robot_cfg = get_go2_robot_cfg()

  ##
  # Sensors.
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name=BASE_BODY, entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=TERRAIN_SCAN_SIZE, resolution=TERRAIN_SCAN_RES),
    max_distance=TERRAIN_SCAN_MAX,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  foot_frames = tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES)
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=foot_frames,
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    reduction="min",
  )
  foot_edge_ring = TerrainHeightSensorCfg(
    name="foot_edge_ring",
    frame=foot_frames,
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.06, num_samples=8),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    reduction="none",
  )
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=4,
  )
  world_contact = ContactSensorCfg(
    name="world_contact",
    primary=ContactMatch(mode="geom", pattern=NON_FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  trunk_contact = ContactSensorCfg(
    name="trunk_contact",
    primary=ContactMatch(mode="geom", pattern=TRUNK_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  sensors = (
    terrain_scan,
    foot_height_scan,
    foot_edge_ring,
    feet_ground,
    world_contact,
    trunk_contact,
    self_collision,
  )

  ##
  # Observations.
  ##

  proprio_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/gyro"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  command_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "goal"}
    ),
  }
  terrain_hist_terms = {
    "height_scan": ObservationTermCfg(
      func=mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1.0 / TERRAIN_SCAN_MAX,
      delay_min_lag=2,
      delay_max_lag=5,
    ),
  }
  terrain_priv_terms = {
    "height_scan_clean": ObservationTermCfg(
      func=mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1.0 / TERRAIN_SCAN_MAX,
    ),
  }
  privileged_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "joint_pos_clean": ObservationTermCfg(func=mdp.joint_pos_rel),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height, params={"sensor_name": "foot_height_scan"}
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces_b, params={"sensor_name": "feet_ground_contact"}
    ),
  }
  task_id_terms = {
    "task_id": ObservationTermCfg(
      func=mdp.task_id,
      params={"foot_height_sensor": "foot_height_scan"},
    ),
  }

  observations = {
    "proprio_hist": ObservationGroupCfg(
      terms=proprio_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=PROPRIO_HISTORY,
      flatten_history_dim=False,
    ),
    "command": ObservationGroupCfg(terms=command_terms, concatenate_terms=True),
    "terrain_hist": ObservationGroupCfg(
      terms=terrain_hist_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=TERRAIN_HISTORY,
      flatten_history_dim=False,
    ),
    "terrain_priv": ObservationGroupCfg(
      terms=terrain_priv_terms, concatenate_terms=True
    ),
    "privileged": ObservationGroupCfg(terms=privileged_terms, concatenate_terms=True),
    "task_id": ObservationGroupCfg(terms=task_id_terms, concatenate_terms=True),
  }

  ##
  # Actions (50 Hz PD joint-position targets, offsets from the home pose).
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=GO2_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  ##
  # Commands.
  ##

  commands: dict[str, CommandTermCfg] = {
    "goal": mdp.GoalCommandCfg(
      entity_name="robot",
      goal_source="cell" if terrain_generator is not None else "ring",
      debug_vis=True,
    )
  }

  ##
  # Events.
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_from_flat_patches,
      mode="reset",
      params={
        "patch_name": "spawn",
        "pose_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "z": (0.02, 0.05),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 8.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.3, 0.3),
          "roll": (-0.4, 0.4),
          "pitch": (-0.4, 0.4),
          "yaw": (-0.6, 0.6),
        },
      },
    ),
    "foot_friction_slide": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
        "operation": "abs",
        "axes": [0],
        "ranges": (0.3, 1.5),
        "shared_random": True,
      },
    ),
    "foot_friction_spin": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [1],
        "ranges": (1e-4, 2e-2),
        "shared_random": True,
      },
    ),
    "foot_friction_roll": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [2],
        "ranges": (1e-5, 5e-3),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,)),
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
      },
    ),
  }

  ##
  # Rewards. mjlab scales every term by dt, as legged_gym-style frameworks do,
  # so literature weights carry over: a fall at -200 costs 4, against 10 per goal
  # reached (``goal_arrival`` is per event) and 2 per metre of progress.
  ##

  rewards = {
    "goal_arrival": RewardTermCfg(
      func=mdp.goal_arrival, weight=10.0, params={"command_name": "goal"}
    ),
    "goal_progress": RewardTermCfg(
      func=mdp.goal_progress, weight=2.0, params={"command_name": "goal"}
    ),
    "heading_tracking": RewardTermCfg(
      func=mdp.heading_tracking, weight=0.5, params={"command_name": "goal"}
    ),
    "dont_wait": RewardTermCfg(
      func=mdp.dont_wait, weight=-1.0, params={"command_name": "goal"}
    ),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "world_contact": RewardTermCfg(
      func=mdp.world_contact_cost,
      weight=world_contact_weight,
      params={"sensor_name": "world_contact"},
    ),
    "self_collision": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision"},
    ),
    "stumble": RewardTermCfg(
      func=mdp.stumble, weight=-1.0, params={"sensor_name": "feet_ground_contact"}
    ),
    "feet_edge": RewardTermCfg(
      func=mdp.feet_edge,
      weight=-1.0,
      params={
        "ring_sensor_name": "foot_edge_ring",
        "contact_sensor_name": "feet_ground_contact",
      },
    ),
    "foot_impact": RewardTermCfg(
      func=mdp.foot_impact, weight=-1.0, params={"sensor_name": "feet_ground_contact"}
    ),
    "energy": RewardTermCfg(func=mdp.mechanical_power, weight=-5e-4),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-5),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
  }

  ##
  # Terminations.
  ##

  terminations = {
    # A long route at the slowest required speed takes just under a minute.
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # A diverged world is reset (as a time-out: it says nothing about the policy)
    # instead of feeding NaN observations to the learner.
    "nan": TerminationTermCfg(func=mdp.nan_detection, time_out=True),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds, time_out=True
    ),
  }
  if terrain_generator is not None:
    terminations["route_complete"] = TerminationTermCfg(
      func=mdp.route_complete, params={"command_name": "goal"}, time_out=True
    )
  if trunk_contact_terminates:
    terminations["trunk_contact"] = TerminationTermCfg(
      func=mdp.trunk_contact, params={"sensor_name": "trunk_contact"}
    )
  if foot_impact_terminates:
    terminations["foot_impact"] = TerminationTermCfg(
      func=mdp.foot_impact_exceeded, params={"sensor_name": "feet_ground_contact"}
    )
  if goal_timeout_terminates:
    terminations["goal_timeout"] = TerminationTermCfg(
      func=mdp.goal_timeout, params={"command_name": "goal"}
    )

  ##
  # Curriculum.
  ##

  curriculum = {
    "command_speed": CurriculumTermCfg(
      func=mdp.command_speed_stages,
      params={
        "command_name": "goal",
        "stages": [
          {"step": 0, "required_speed_range": (0.3, 1.0)},
          {"step": 2000 * 24, "required_speed_range": (0.3, 1.5)},
          {"step": 5000 * 24, "required_speed_range": (0.3, 2.5)},
        ],
      },
    ),
    "context_length": CurriculumTermCfg(
      func=mdp.context_length_stages, params={"stages": CONTEXT_STAGES}
    ),
  }
  if terrain_generator is not None:
    curriculum["terrain_levels"] = CurriculumTermCfg(
      func=mdp.terrain_levels_goal, params={"command_name": "goal"}
    )

  ##
  # Scene / sim.
  ##

  if terrain_generator is not None:
    terrain = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=terrain_generator,
      max_init_terrain_level=3,
    )
  else:
    terrain = TerrainEntityCfg(terrain_type="plane")

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=terrain,
      entities={"robot": robot_cfg},
      sensors=sensors,
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=BASE_BODY,
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      # Per-world contact budget. Measured under random actions: mean ~9, peak 27
      # (drops) contacts per world; contacts share one arena across worlds. The CCD
      # scratch memory scales with nconmax * num_envs * ccd_iterations, so keep both
      # modest (200 x 200 cost 10 GB at 4096 envs).
      nconmax=48,
      njmax=2000,
      contact_sensor_maxmatch=500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
        ccd_iterations=100,
      ),
    ),
    decimation=4,
    episode_length_s=episode_length_s,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    for group in ("proprio_hist", "terrain_hist"):
      cfg.observations[group].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations.pop("goal_timeout", None)
    cfg.curriculum = {}
    if terrain_generator is not None:
      cfg.events["randomize_terrain"] = EventTermCfg(
        func=mdp.randomize_terrain, mode="reset", params={}
      )
      terrain_generator.curriculum = False
      terrain_generator.num_rows = 5
      terrain_generator.num_cols = max(5, len(terrain_generator.sub_terrains))
      terrain_generator.border_width = 10.0
  return cfg
