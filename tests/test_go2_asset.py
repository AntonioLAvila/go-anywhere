import mujoco
import numpy as np
from mjlab.entity.entity import Entity

from go_anywhere.assets.go2 import (
  ALL_COLLISION_GEOMS,
  FOOT_GEOMS,
  FOOT_SITES,
  JOINT_NAMES,
  NON_FOOT_GEOMS,
  TOTAL_MASS_KG,
  get_go2_robot_cfg,
  get_spec,
)


def _names(model, objtype, n):
  return [mujoco.mj_id2name(model, objtype, i) for i in range(n)]


def test_spec_compiles_without_xml_actuators_or_keys():
  spec = get_spec()
  assert len(spec.actuators) == 0
  assert len(spec.keys) == 0
  model = spec.compile()
  sensors = _names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor)
  assert {"imu_lin_vel", "root_angmom", "gyro"} <= set(sensors)


def test_entity_has_12_actuators_in_joint_order_and_camera():
  entity = Entity(get_go2_robot_cfg())
  model = entity.spec.compile()
  assert model.nu == 12
  joints = [
    n
    for n in _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    if n != "floating_base"
  ]
  assert joints == list(JOINT_NAMES)
  for i in range(model.nu):
    jid = model.actuator_trnid[i, 0]
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) == JOINT_NAMES[i]
  assert "head_camera" in _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
  geoms = set(_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom))
  assert set(ALL_COLLISION_GEOMS) <= geoms
  sites = set(_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite))
  assert set(FOOT_SITES) <= sites
  assert not set(FOOT_GEOMS) & set(NON_FOOT_GEOMS)
  assert abs(float(np.sum(model.body_mass)) - TOTAL_MASS_KG) < 0.5


def test_only_named_collision_geoms_collide():
  entity = Entity(get_go2_robot_cfg())
  model = entity.spec.compile()
  for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    colliding = model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
    assert colliding == (name in ALL_COLLISION_GEOMS), name
