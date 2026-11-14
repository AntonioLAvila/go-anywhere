"""Hurdles expert: two-stage soft-to-hard curriculum (Zhuang et al. 2023).

Stage A (``...-Hurdles-Go2``): bars are non-colliding; a penetration cost over
sampled body points teaches clearance. Stage B (``...-Hurdles-Hard-Go2``):
bars collide; resume from the stage-A checkpoint with ``--agent.resume``/
``--agent.init-from``.
"""

from go_anywhere.tasks.experts.common import register_expert

TASK_ID_SOFT = register_expert("hurdles", hurdles_collidable=False)
TASK_ID_HARD = register_expert("hurdles", suffix="-Hard", hurdles_collidable=True)
