"""Unitree Go2 constants for go-anywhere.

Mirrors ``mjlab/asset_zoo/robots/unitree_go1/go1_constants.py`` but loads the
MJCF from the ``go2_mjcf`` submodule. The submodule XML already carries inlined
``<motor>`` actuators, keyframes and sensors (see its header comment); the
spec function below strips what mjlab replaces and adds what the locomotion
stack expects.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CameraCfg, CollisionCfg

from go_anywhere import GO2_MJCF_DIR

##
# MJCF.
##

GO2_XML: Path = GO2_MJCF_DIR / "go2.xml"
BUILD_POSE_HEIGHT = 3.0


def get_spec() -> mujoco.MjSpec:
  """Load the Go2 spec and adapt it for mjlab.

  - Deletes the XML ``<motor>`` actuators (mjlab adds PD actuators) and the XML
    keyframes (their ``ctrl`` vectors would no longer match ``nu``; mjlab adds
    its own initial-state keyframe).
  - Adds the IMU velocimeter and the subtree angular momentum sensor that the
    mjlab locomotion rewards/observations read by name. Candidates to upstream
    into ``go2_mjcf`` later.
  """
  if not GO2_XML.exists():
    raise FileNotFoundError(
      f"Go2 MJCF not found at {GO2_XML}. Initialize the submodule with "
      "`git submodule update --init`."
    )
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for key in list(spec.keys):
    spec.delete(key)
  # mujoco_warp sizes its contact check from mj_forward on the XML default pose
  # (not mjlab's init keyframe). Park that pose above any terrain so the check
  # sees no contacts; spawn heights come from InitialStateCfg and are unaffected.
  spec.body("base").pos = (0.0, 0.0, BUILD_POSE_HEIGHT)
  existing = {s.name for s in spec.sensors}
  if "imu_lin_vel" not in existing:
    spec.add_sensor(
      name="imu_lin_vel",
      type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname="imu",
    )
  if "root_angmom" not in existing:
    spec.add_sensor(
      name="root_angmom",
      type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      objname="base",
    )
  return spec


##
# Names.
##

BASE_BODY = "base"
IMU_SITE = "imu"
LEGS = ("FL", "FR", "RL", "RR")
FOOT_GEOMS = tuple(LEGS)
FOOT_SITES = tuple(f"{leg}_site" for leg in LEGS)
HIP_GEOMS = tuple(f"{leg}_hip_col" for leg in LEGS)
THIGH_GEOMS = tuple(f"{leg}_thigh_col" for leg in LEGS)
CALF_GEOMS = tuple(f"{leg}_calf_{part}" for leg in LEGS for part in ("upper", "lower"))
TRUNK_GEOMS = ("torso_box", "head_cyl", "head_sphere")
NON_FOOT_GEOMS = TRUNK_GEOMS + HIP_GEOMS + THIGH_GEOMS + CALF_GEOMS
ALL_COLLISION_GEOMS = NON_FOOT_GEOMS + FOOT_GEOMS

# Joint order in the XML (and therefore actuator order after mjlab sorting).
JOINT_NAMES = tuple(
  f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")
)

##
# Mass. Base 6.921 kg + 4 x (0.678 + 1.152 + 0.241) kg from the MJCF inertials.
##

TOTAL_MASS_KG = 6.921 + 4 * (0.678 + 1.152 + 0.241352)
GRAVITY = 9.81
BODY_WEIGHT_N = TOTAL_MASS_KG * GRAVITY

##
# Actuators. Datasheet peaks: 23.7 Nm hip/thigh, 45.43 Nm calf (1.5 gear).
# Velocity limits mirror the Go1 numbers in mjlab (30.1 / 20.06 rad/s); the
# Go2 datasheet lists 30 rad/s at the hip/thigh. TODO(verify) calf no-load speed.
# Reflected inertia is taken from the XML ``armature`` values (0.01 / 0.02).
##

HIP_EFFORT = 23.7
CALF_EFFORT = 45.43
HIP_VELOCITY_LIMIT = 30.1
CALF_VELOCITY_LIMIT = 20.06
HIP_ARMATURE = 0.01
CALF_ARMATURE = 0.02

NATURAL_FREQ = 10.0 * 2.0 * math.pi  # 10 Hz, as in mjlab's Go1.
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_ARMATURE * NATURAL_FREQ**2
DAMPING_HIP = 2.0 * DAMPING_RATIO * HIP_ARMATURE * NATURAL_FREQ
STIFFNESS_CALF = CALF_ARMATURE * NATURAL_FREQ**2
DAMPING_CALF = 2.0 * DAMPING_RATIO * CALF_ARMATURE * NATURAL_FREQ

GO2_HIP_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS_HIP,
  damping=DAMPING_HIP,
  effort_limit=HIP_EFFORT,
  saturation_effort=HIP_EFFORT,
  velocity_limit=HIP_VELOCITY_LIMIT,
  armature=HIP_ARMATURE,
)
GO2_CALF_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS_CALF,
  damping=DAMPING_CALF,
  effort_limit=CALF_EFFORT,
  saturation_effort=CALF_EFFORT,
  velocity_limit=CALF_VELOCITY_LIMIT,
  armature=CALF_ARMATURE,
)

##
# Initial states (from the XML ``home`` / ``tuck`` keyframes).
##

HOME_JOINT_POS: dict[str, float] = {
  ".*_hip_joint": 0.0,
  ".*_thigh_joint": 0.9,
  ".*_calf_joint": -1.8,
}
TUCK_JOINT_POS: dict[str, float] = {
  ".*_hip_joint": 0.0,
  ".*_thigh_joint": 2.2,
  ".*_calf_joint": -2.7,
}

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.27),
  joint_pos=dict(HOME_JOINT_POS),
  joint_vel={".*": 0.0},
)

##
# Collisions. Feet get condim 6 (torsional + rolling friction) and priority 1
# as in the XML; every other collision geom keeps a regular frictional contact
# so thighs/knees can bear load on edges (climbing, recovery).
##

_FOOT_REGEX = r"^(FL|FR|RL|RR)$"
_NON_FOOT_REGEX = r"^(torso_box|head_cyl|head_sphere|.*_hip_col|.*_thigh_col|.*_calf_upper|.*_calf_lower)$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_REGEX, _NON_FOOT_REGEX),
  contype=1,
  conaffinity=1,
  condim={_FOOT_REGEX: 6, _NON_FOOT_REGEX: 3},
  priority={_FOOT_REGEX: 1, _NON_FOOT_REGEX: 0},
  friction={_FOOT_REGEX: (0.8, 0.02, 0.01), _NON_FOOT_REGEX: (0.6,)},
  solref=(0.01, 1.0),
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_REGEX,),
  contype=0,
  conaffinity=1,
  condim=6,
  priority=1,
  friction=(0.8, 0.02, 0.01),
)

##
# Head camera. Reserved for the depth student (Phase 3); harmless otherwise.
##


def _forward_camera_quat(pitch_down_deg: float) -> tuple[float, float, float, float]:
  """Quaternion (w, x, y, z) of a MuJoCo camera looking along body +X.

  MuJoCo cameras look along their local -Z with +Y up. A level forward camera
  therefore has axes x_c = -Y_b, y_c = +Z_b, z_c = -X_b. A positive
  ``pitch_down_deg`` tilts the view toward the ground.
  """
  level = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
  )
  phi = -math.radians(pitch_down_deg)
  rot_x = np.array(
    [
      [1.0, 0.0, 0.0],
      [0.0, math.cos(phi), -math.sin(phi)],
      [0.0, math.sin(phi), math.cos(phi)],
    ]
  )
  mat = level @ rot_x
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, mat.flatten())
  return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


HEAD_CAMERA = CameraCfg(
  name="head_camera",
  body=BASE_BODY,
  pos=(0.32, 0.0, 0.03),
  quat=_forward_camera_quat(15.0),
  fovy=58.0,
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_HIP_ACTUATOR_CFG, GO2_CALF_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg(feet_only_collision: bool = False) -> EntityCfg:
  """Return a fresh Go2 entity config (fresh instance avoids shared mutation)."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FEET_ONLY_COLLISION if feet_only_collision else FULL_COLLISION,),
    cameras=(HEAD_CAMERA,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
    sort_actuators=True,  # ctrl follows joint order (FL, FR, RL, RR x hip, thigh, calf).
  )


GO2_ACTION_SCALE: dict[str, float] = {}
for _a in GO2_ARTICULATION.actuators:
  assert isinstance(_a, DcMotorActuatorCfg)
  for _n in _a.target_names_expr:
    GO2_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())
  viewer.launch(robot.spec.compile())
