"""Stand-up-from-squat locomotion task.

The robot starts in the deep squat pose defined by the humanoid-control policy contract and
is rewarded for rising to and holding a standing base height while staying upright. Shares the
observation/action layout (and thus the sim<->real contract) with the velocity/walk task.
"""

from .config import *  # noqa: F401, F403
