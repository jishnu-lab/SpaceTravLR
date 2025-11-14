import sys
import scanpy as sc
sys.path.append('../src')

from SpaceTravLR.spaceship import SpaceShip

spacetravlr = SpaceShip(
    name='myTonsil', 
    outdir='/ocean/projects/cis240075p/awang22/output_test/'
)
assert spacetravlr.is_everything_ok()

spacetravlr.fit()
