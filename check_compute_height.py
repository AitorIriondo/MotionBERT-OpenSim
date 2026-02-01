"""Check compute_height function signature."""
from Pose2Sim.common import compute_height
import inspect

print("compute_height signature:")
print(inspect.signature(compute_height))
print()
print("Docstring:")
print(compute_height.__doc__[:2000] if compute_height.__doc__ else "No docstring")
