"""go-anywhere: near-arbitrary locomotion for the Unitree Go2 on mjlab."""

from pathlib import Path

GO_ANYWHERE_SRC_PATH: Path = Path(__file__).parent
GO_ANYWHERE_ROOT: Path = GO_ANYWHERE_SRC_PATH.parent
GO2_MJCF_DIR: Path = GO_ANYWHERE_ROOT / "go2_mjcf"
CONFIGS_DIR: Path = GO_ANYWHERE_ROOT / "configs"

# Import mjlab before any submodule: mjlab's import runs task discovery through
# the ``mjlab.tasks`` entry point, which imports ``go_anywhere.tasks``. Doing it
# here (with only the path constants defined) avoids half-initialized modules
# when a submodule such as ``assets.go2`` is the first thing imported.
import mjlab  # noqa: E402,F401
