"""Check Pose2Sim skeletons and compute_height function."""
try:
    from Pose2Sim import skeletons
    print('Pose2Sim skeletons module loaded')
    print('Available:', [x for x in dir(skeletons) if not x.startswith('_')])
except Exception as e:
    print(f'Error loading skeletons: {e}')

try:
    from Pose2Sim.common import compute_height
    print('compute_height function available')
except Exception as e:
    print(f'Error loading compute_height: {e}')

try:
    from Pose2Sim import Pose2Sim as P2S
    print('Pose2Sim main module loaded')
    print('Functions:', [x for x in dir(P2S) if not x.startswith('_')])
except Exception as e:
    print(f'Error loading Pose2Sim: {e}')
