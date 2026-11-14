"""Compile the Go2 asset, print its structure, optionally open the viewer.

Usage: bin/uv-run python scripts/check_go2.py [--viewer]
"""

from __future__ import annotations

import sys

import mujoco
from mjlab.entity.entity import Entity

from go_anywhere.assets.go2 import (
  ALL_COLLISION_GEOMS,
  BODY_WEIGHT_N,
  JOINT_NAMES,
  TOTAL_MASS_KG,
  get_go2_robot_cfg,
)


def main() -> None:
  robot = Entity(get_go2_robot_cfg())
  model = robot.spec.compile()
  print(
    f"bodies={model.nbody} joints={model.njnt} actuators={model.nu} sensors={model.nsensor} cameras={model.ncam}"
  )
  act_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
  ]
  print("actuators:", act_names)
  jnt = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
  ]
  print("joints:", jnt)
  print("expected joint order:", list(JOINT_NAMES))
  print(
    "sensors:",
    [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
      for i in range(model.nsensor)
    ],
  )
  print(
    "cameras:",
    [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)
    ],
  )
  print("collision geoms:", [g for g in ALL_COLLISION_GEOMS])
  print(
    f"mass constant {TOTAL_MASS_KG:.2f} kg vs model {sum(model.body_mass):.2f} kg; BW {BODY_WEIGHT_N:.0f} N"
  )
  if "--viewer" in sys.argv:
    import mujoco.viewer as viewer

    viewer.launch(model)


if __name__ == "__main__":
  main()
