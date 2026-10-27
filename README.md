# Hex map tiles (3D-printable)

73 interlocking hex tiles for tabletop maps. Terrain is sculpted (noise-based mountains,
hills, dunes; real tree shapes, houses, castle, temples) rather than stacked blocks.

## Dimensions
- 40 mm flat-to-flat (printed 39.8 mm so tiles fit without sanding), 5 mm base slab, 0.4 mm bottom chamfer
- Tallest tile 21 mm (mountains); most are 5-17 mm
- Pointy-top hex. Edge midpoints are at 0/60/120/180/240/300 degrees
- Every tile has a perfectly flat 2 mm rim, and rivers (6 mm wide, 1 mm deep) and
  roads (4 mm wide, 0.5 mm raised) cross every edge at the midpoint with an identical
  profile, so **any tile joins any tile, in any rotation**

## Magnets
Two 3 x 1 mm disc magnet pockets on each edge (12 per tile), open on the side, 8 mm apart.
Put them in so that, going anticlockwise around each edge, the first pocket faces **N** outward
and the second faces **S** outward. That makes every edge attract every other edge, in any
rotation (a single magnet per edge would repel when a tile is turned by 60 degrees).
Regenerate for other magnets:  `python3 generate.py --magnet-d 4 --magnet-t 2`  (or `--no-magnets`).

## The tiles (stl/ folder)
| Group | Files |
|---|---|
| Terrain | plains, hills, hills_b, mountains, mountains_b, forest, forest_b, desert, swamp, water |
| Roads (on plains, hills, forest, desert) | road_{dead_end, straight, curve_gentle, curve_sharp, fork_y, fork_t, cross}_{terrain} |
| Rivers (on plains, hills, forest) | river_{same 7 patterns}_{terrain} |
| Crossings | bridge_plains, ford_plains |
| Villages | village_0 (no road), village_1 (road ends), village_2 (road passes through) |
| Cities (walled) | city_1, city_2, city_3 (one, two opposite, three gates) |
| Others | fort_1, temple_0/1, ziggurat_0/1, dungeon_1 |

Patterns are listed by their edges (edge 0 is the lower-right edge in the preview images);
print one and rotate it. `fork_t` is a one-handed T: mirror it in your slicer for the other hand.
Lakes: the `water` tile sits 1 mm below the land, which is exactly the river floor, so rivers run
straight into it.

## Printing
- Flat base on the bed, no supports needed (tree skirts and arches are within ~40 degrees; the bridge
  deck, temple roof and dungeon arch are short bridges of a few mm)
- 0.4 mm nozzle, 0.12-0.16 mm layers, 15-20 % infill (the base is thick enough for magnets)
- Mountains and trees look best at 0.12 mm layers; fine tower tops are 0.6-1 mm wide

## Regenerating / editing
```
pip install numpy scipy trimesh manifold3d pillow
python3 generate.py --set all            # everything (about 20 seconds)
python3 generate.py --only city_2 forest --flat 60     # chosen tiles, 60 mm tiles
python3 validate.py stl                  # watertight / single-body / overhang report
python3 preview.py sheet.png stl/*.stl --cols 4        # contact sheet (no GPU needed)
```
`seed` values in `generate.py` change the look of any tile.
