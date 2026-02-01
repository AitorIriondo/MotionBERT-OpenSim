"""Check HALPE_26 skeleton from Pose2Sim."""
import sys
sys.path.insert(0, 'E:/KineticsToolkit/src')

from trc_pose2sim import HALPE_26_MARKERS
print('HALPE_26 markers from KineticsToolkit:')
for i, (name, mtype, source) in enumerate(HALPE_26_MARKERS):
    print(f'  {i}: {name} ({mtype})')
