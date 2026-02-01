"""Get HALPE_26 skeleton info from Pose2Sim."""
from Pose2Sim.skeletons import HALPE_26
from anytree import RenderTree

print('HALPE_26 skeleton tree:')
for pre, fill, node in RenderTree(HALPE_26):
    print(f'{pre}{node.name} (id={node.id})')

print('\nNode list by ID:')
from anytree import PreOrderIter
nodes = list(PreOrderIter(HALPE_26))
for node in sorted(nodes, key=lambda x: x.id):
    print(f'  {node.id}: {node.name}')
