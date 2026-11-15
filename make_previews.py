#!/usr/bin/env python3
"""make_previews.py - rebuild the README images in previews/ from a folder of generated tiles.

  python3 generate.py --set all            # writes stl/*.stl and stl/*.mat.npz
  python3 make_previews.py                 # renders every sheet (a few minutes, runs in parallel)
  python3 make_previews.py overview coasts # only these sheets        [--stl stl] [--out previews]"""
import argparse, os
from multiprocessing import Pool
from PIL import Image, ImageDraw
from preview import Tile, render_scene

# name: (tiles, columns, tile size in px, extra render options)
SHEETS = {
    'overview': (['plains', 'hills', 'mountains', 'forest', 'village_2', 'city_2', 'fort_1', 'temple_1', 'ziggurat_1',
                  'dungeon_1', 'river_curve_sharp_hills', 'water'], 4, 400, {}),
    'roads': (['road_dead_end_plains', 'road_straight_plains', 'road_curve_gentle_forest', 'road_curve_sharp_hills',
               'road_fork_y_plains', 'road_fork_t_forest', 'road_cross_plains', 'road_straight_desert'], 4, 400, {}),
    'rivers': (['river_dead_end_plains', 'river_straight_plains', 'river_curve_gentle_hills', 'river_curve_sharp_plains',
                'river_fork_y_plains', 'river_fork_t_forest', 'river_cross_hills', 'river_straight_forest'], 4, 400, {}),
    'misc': (['swamp', 'desert', 'hills_b', 'forest_b', 'mountains_b', 'village_0', 'city_3', 'temple_0'], 4, 400, {}),
    'bridge_ford': (['bridge_plains', 'ford_plains', 'river_straight_plains'], 3, 560, {}),
    'coasts': (['coast_1_plains', 'coast_2_plains', 'coast_3_plains', 'coast_4_plains', 'coast_5_plains', 'coast_3_hills',
                'coast_2_forest', 'coast_1_river_plains', 'coast_2_river_hills', 'coast_3_river_plains',
                'coast_3_river_forest', 'coast_1_hills'], 4, 400, {}),
    'lakes': (['lake_pond_plains', 'lake_pond_forest', 'lake_river_1_plains', 'lake_river_straight_plains',
               'lake_river_curve_gentle_forest', 'lake_river_curve_sharp_plains', 'lake_river_fork_y_plains',
               'lake_river_fork_y_forest'], 4, 400, {}),
    'features2': (['ruins_plains', 'mine_1', 'stones_0', 'watchtower_0', 'harbour_0', 'island_lighthouse'], 3, 420, {}),
    'coast_styles': (['cliff_coast_3_hills', 'sand_coast_3', 'marsh_coast_3', 'cliff_coast_1_river_hills', 'island_forest',
                      'island_hills'], 3, 420, {}),
    'terrains2': (['tundra', 'glacier', 'jungle', 'farmland', 'forest_hills', 'forest_mountains'], 3, 420, {}),
    'towns': (['village_0', 'village_farm_2', 'village_2', 'city_1', 'city_2', 'city_3'], 3, 500, {'span': 22.0}),
    'more_paths': (['road_straight_swamp', 'river_curve_gentle_desert', 'river_straight_jungle', 'road_fork_y_farmland',
                    'river_curve_sharp_tundra', 'road_curve_gentle_forest_hills', 'river_straight_swamp',
                    'road_straight_forest_mountains'], 4, 400, {}),
    'mirrored_coasts': (['coast_2_plains', 'coast_2_plains_mirror', 'coast_3_hills', 'coast_3_hills_mirror',
                         'coast_2_river_plains', 'coast_2_river_plains_mirror', 'coast_3_mountains', 'coast_3_mountains_mirror'],
                        4, 400, {}),
    'mountains2': (['waterfall_mountains', 'pass_straight_mountains', 'gorge_curve_gentle_mountains',
                    'pass_fork_y_mountains'], 2, 480, {}),
}


def render_one(args):
    path, size, opts = args
    im = render_scene([Tile(path)], size=size, **opts)
    ImageDraw.Draw(im).text((8, 6), os.path.splitext(os.path.basename(path))[0], fill=(235, 235, 235))
    return im


def water_seamless(stl):
    """four water tiles side by side: the ripples run on across the joins"""
    cells = [(0, 0), (1, 0), (0, 1), (-1, 1)]
    tiles = [Tile(os.path.join(stl, 'water.stl'), (40.0 * i + 20.0 * j, 34.641 * j)) for i, j in cells]
    cx = sum(t.V[:, 0].mean() for t in tiles) / 4
    cy = sum(t.V[:, 1].mean() for t in tiles) / 4
    for t in tiles:
        t.V[:, 0] -= cx; t.V[:, 1] -= cy
        t.grid['xs'] = t.grid['xs'] - cx; t.grid['ys'] = t.grid['ys'] - cy
    return render_scene(tiles, size=900, span=38.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sheets', nargs='*')
    ap.add_argument('--stl', default='stl')
    ap.add_argument('--out', default='previews')
    a = ap.parse_args()
    want = a.sheets or list(SHEETS) + ['water_seamless']
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for name in want:
        if name == 'water_seamless':
            continue
        tiles, cols, size, opts = SHEETS[name]
        jobs += [(name, i, (os.path.join(a.stl, t + '.stl'), size, opts)) for i, t in enumerate(tiles)]
    with Pool() as pool:
        res = pool.map(render_one, [j[2] for j in jobs], chunksize=1)
        for name in want:
            if name == 'water_seamless':
                water_seamless(a.stl).save(os.path.join(a.out, name + '.png'))
                print('wrote', name)
                continue
            tiles, cols, size, opts = SHEETS[name]
            ims = [im for (n, i, _), im in zip(jobs, res) if n == name]
            rows = (len(ims) + cols - 1) // cols
            sheet = Image.new('RGB', (cols * size, rows * size))
            for i, im in enumerate(ims):
                sheet.paste(im, ((i % cols) * size, (i // cols) * size))
            sheet.save(os.path.join(a.out, name + '.png'))
            print('wrote', name, sheet.size)


if __name__ == '__main__':
    main()
