import sys
import scanpy as sc
sys.path.append('../src')

from SpaceTravLR.spaceship import SpaceShip

spacetravlr = SpaceShip(
    name='myTonsil', 
    outdir='./my_tonsil'
)
assert spacetravlr.is_everything_ok()

spacetravlr.fit()
