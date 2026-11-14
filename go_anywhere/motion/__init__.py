"""Motion priors (Phase 4 placeholder).

Planned: an AMP-style discriminator reward over dog mocap features (joint
positions/velocities, base height, foot contacts) reusing mjlab's motion npz
format (``joint_pos``, ``joint_vel``, ``body_pos_w``, ...). rsl_rl 5.4.2 has no
AMP, so the discriminator and its replay buffer live here as an algorithm
extension.
"""
