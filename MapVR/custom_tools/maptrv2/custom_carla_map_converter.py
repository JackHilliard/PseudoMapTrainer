"""Convert the CARLA road-polyline tile dataset into MapTR's map-annotation
pkl format.

Unlike the nuScenes/AV2 converters, CARLA tiles are static square patches
(not per-timestamp driving-log frames), so there is no ego pose/SE3 transform
and no need to clip polylines against a moving patch -- this script only
needs to shift each tile's world-frame reference-line polylines into a
common frame with its point cloud. By default that frame is the tile's own
centre (`--gt-frame tile_center`, matching the MapTRv2/GeMap benchmark
convention): the polylines are shifted by `tile_center` and each sample
records the `lidar_recenter_shift` that `LoadCarlaPointsFromFile` adds to
the stored (offset-frame) points, so both land in the same, origin-centred
frame. `--gt-frame offset` reproduces the original behaviour (subtract the
block's own `offset`, shift nothing else -- see the comment on the np.load
call below for the trade-off).

Nothing here assumes a particular tile size. The exporter has produced at
least 25m tiles (`tile_radius` 12.5) and 60m ones (`tile_radius` 30.0), and
both layouts it writes are accepted::

    <data_root>/<split>/manifest.json            # split export
    <data_root>/<split>/blocks/<tile>.npz
    <data_root>/<split>/reference_lines/<tile>_reference_lines.json

    <data_root>/grid_manifest.json               # single-town grid export
    <data_root>/blocks/<tile>.npz                # (no <split> level, and
    <data_root>/reference_lines/<tile>_...json   #  tile names carry no town)

Where a directory holds both files, `manifest.json` wins and
`grid_manifest.json` only backfills keys it lacks -- the same rule
`tools/maptrv2/dataset_viewer.py` uses. `manifest.json` is written by a
later packaging step and is the *curated* view: in the Town10HD grid export
it lists 30 tiles where grid_manifest.json lists 33, and the 3 it omits
contain only sidewalk_edge polylines and no driving centerline at all
(`gt_source: driving_lanes`). Its dataset-level *counts* are still not
trustworthy -- the same file reports `n_tiles: 4103` while listing 30 -- so
everything here is derived from the tile entries, never from those counts.

Note the reference-lines json on disk is not filtered to match: it still
holds every class, so a class-carrying export converts all of them unless
--classes says otherwise.

Each tile's LiDAR block is also scanned for points that would actually
survive the training pipeline's filters -- which, in this repo, means only
``--lidar-point-cloud-range``: ``LoadCarlaPointsFromFile`` here does **no**
z filtering (``--z-max`` defaults to off and exists only to mirror a loader
that applies one). Tiles with fewer than ``--min-lidar-points``
such points would voxelize to *zero* voxels and crash ``extract_lidar_feat``,
so they are dropped from the pkl and listed in a sidecar report. Every kept
sample records its own count so the dataset can re-check cheaply. The
default range is derived from the manifest's own tile size rather than
hardcoded, so the scan describes the tiles being converted.

Usage::

    python custom_tools/maptrv2/custom_carla_map_converter.py \\
        --data-root /path/to/carla --out-dir data/carla/ --split test
"""

import argparse
import json
import os

import mmcv
import numpy as np

# z half of the range the training config's LiDAR branch uses
# (`lidar_point_cloud_range` in projects/configs/maptrv2/pmt_carla_lidar.py).
# Wide on purpose: this pipeline does no z filtering anywhere, so the
# voxelizer range is the only thing that can drop a point on the z axis, and
# these tiles legitimately span a lot of z (highway overpasses inside a
# 25 x 25 m footprint). Measured exactly over all 4103 train + 259 test tiles
# of the reference export: z in [-97.09, +91.43]. Re-measure for a different
# export -- see the sweep documented in projects/configs/carla/carlasim_map.py.
# The xy half is NOT a constant: it follows the tile size read from the
# manifest (see default_pc_range), because a range narrower than the tile
# silently crops it and a wider one just wastes BEV cells. The whole range is
# recorded into the pkl so the dataset can warn if config and pkl drift apart.
DEFAULT_Z_RANGE = (-98.0, 92.0)
# No loader-side z filter in this repo (LoadCarlaPointsFromFile drops its
# z_max argument entirely), so nothing to mirror here by default.
DEFAULT_Z_MAX = None
# Only used when a manifest carries no tile geometry at all; matches the
# original 25m export this converter was written against.
FALLBACK_TILE_RADIUS = 12.5
# Slack allowed before a polyline is reported as leaving its own tile. The
# exporter samples arcs at a fixed arc_gap, so a vertex can land slightly
# past the boundary without anything being wrong.
BOUNDS_MARGIN = 1.0

MANIFEST_NAMES = ('manifest.json', 'grid_manifest.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description='CARLA map data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='root of the CARLA tile dataset (contains <split>/manifest.json, '
        'or a manifest.json/grid_manifest.json directly)')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/carla/',
        help='output directory for the generated pkl')
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split subdirectory under --data-root to convert; also used as '
        'the label in the output filename. If no such subdirectory exists, '
        '--data-root itself is read and this is only the output label')
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=None,
        metavar='CLASS',
        help="optional subset of polyline classes to keep as divider "
        "instances, as class_lookup ids or names (e.g. `0` or "
        "`driving_centerline`). Default: keep every polyline. Only exports "
        "whose polylines carry a class can be filtered")
    parser.add_argument(
        '--lane-types',
        type=str,
        nargs='+',
        default=None,
        metavar='CLASS',
        help='deprecated alias for --classes (it never matched anything: it '
        "compared lane_type_lookup ids against each polyline's `type`, which "
        "holds a geometry kind such as 'arc'/'straight')")
    parser.add_argument(
        '--lidar-point-cloud-range',
        type=float,
        nargs=6,
        default=None,
        metavar=('X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX'),
        help='range the LiDAR voxelizer will use; points outside it are '
        'dropped before voxelization, so a tile with none inside produces '
        'zero voxels. Default: +/-tile_radius in xy (read from the manifest) '
        f'and {DEFAULT_Z_RANGE[0]}..{DEFAULT_Z_RANGE[1]} in z')
    parser.add_argument(
        '--z-max',
        type=float,
        default=DEFAULT_Z_MAX,
        help='optional early z filter to mirror when counting in-range '
        'points. This repo does no z filtering, so it defaults to off; set '
        'it only if you reinstate one in LoadCarlaPointsFromFile')
    parser.add_argument(
        '--min-lidar-points',
        type=int,
        default=1,
        help='drop tiles with fewer than this many in-range LiDAR points; '
        '1 drops only genuinely zero-voxel tiles (default: 1)')
    parser.add_argument(
        '--no-lidar-check',
        action='store_true',
        help='skip the per-tile LiDAR scan entirely (faster, but no tile is '
        'dropped and no point counts are recorded)')
    parser.add_argument(
        '--gt-frame',
        type=str,
        choices=('offset', 'tile_center'),
        default='tile_center',
        help="frame the GT polylines (and, via the recorded per-sample "
        "shift, the LiDAR points) are expressed in. 'tile_center' (default) "
        "centres each tile on its own geometric centre, matching the "
        "MapTRv2/GeMap benchmark convention -- the model's origin-centred "
        "pc_range then covers the whole tile instead of a displaced copy of "
        "it. 'offset' reproduces this converter's original behaviour "
        "(the block's own offset frame) for comparison with old runs")
    return parser.parse_args()


def load_manifest(data_root, split):
    """Locate and load the manifest describing one set of tiles.

    Accepts either export layout: ``<data_root>/<split>/`` (the split
    export) or ``<data_root>`` itself (a grid export, which has no split
    level). Returns ``(tile_dir, manifest)``; `tile_dir` is where `blocks/`
    and `reference_lines/` live.
    """
    for candidate in ([os.path.join(data_root, split)] if split else
                      []) + [data_root]:
        present = [
            n for n in MANIFEST_NAMES
            if os.path.isfile(os.path.join(candidate, n))
        ]
        if not present:
            continue
        manifest = {}
        # reversed so the preferred name (first in MANIFEST_NAMES) is applied
        # last and its keys win, while the other only backfills. A null value
        # never overwrites a real one -- grid_manifest.json spells absent
        # options as `null` rather than omitting them.
        for name in reversed(present):
            with open(os.path.join(candidate, name), encoding='utf-8') as f:
                loaded = json.load(f)
            manifest.update({
                k: v
                for k, v in loaded.items()
                if v is not None or k not in manifest
            })
        if not manifest.get('tiles'):
            raise ValueError(
                f'{candidate}: manifest lists no tiles')
        return candidate, manifest
    raise FileNotFoundError(
        f'no {" or ".join(MANIFEST_NAMES)} under {data_root!r}'
        f'{f" or {os.path.join(data_root, split)!r}" if split else ""}')


def manifest_tile_radius(manifest):
    """Half a tile's side, in metres, from whichever key the export used.

    `tile_radius` (25m export) and `tile_side` (some grid manifests) are
    dataset-level; failing both, the widest per-tile `bounds` gives the same
    answer for a uniform grid and an upper bound otherwise, which is the
    safe direction for sizing a point-cloud range. Returns None only when an
    export states its geometry nowhere -- the per-tile bounds check can
    still fall back to each reference-lines json, so that is not fatal.
    """
    if manifest.get('tile_radius') is not None:
        return float(manifest['tile_radius'])
    if manifest.get('tile_side') is not None:
        return float(manifest['tile_side']) / 2.0
    radii = []
    for tile in manifest.get('tiles', []):
        bounds = tile.get('bounds')
        if bounds is None:
            continue
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:
            radii.append(np.max(bounds[2:] - bounds[:2]) / 2.0)
        elif bounds.size == 6:
            radii.append(np.max(bounds[3:5] - bounds[:2]) / 2.0)
    return float(max(radii)) if radii else None


def tile_footprint(tile, ref, tile_radius):
    """World-frame xy footprint of one tile, as ``(lo, hi)`` arrays.

    Prefers the tile's own `bounds` (exact, and the only source that would
    describe a non-square tile), then its centre plus the export's tile
    radius. The reference-lines json repeats all three keys, so it backs up
    a manifest whose tile entries are sparse. Returns None when nothing
    supplies a footprint, which only disables the bounds warning below.
    """
    bounds = tile.get('bounds') or ref.get('tile_bounds')
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:  # [x_min, y_min, x_max, y_max]
            return bounds[:2], bounds[2:]
        if bounds.size == 6:  # [x_min, y_min, z_min, x_max, y_max, z_max]
            return bounds[:2], bounds[3:5]

    center = tile.get('center') or ref.get('tile_center')
    radius = ref.get('tile_radius')
    radius = float(radius) if radius is not None else tile_radius
    if center is None or radius is None:
        return None
    center = np.asarray(center, dtype=np.float64)[:2]
    return center - radius, center + radius


def tile_center_origin(tile, ref, offset, tile_radius):
    """The tile's own geometric centre, as a 3-vector in world coordinates.

    Prefers the manifest tile's `center` (backfilled by the reference-lines
    json's `tile_center`), falling back to the midpoint of the tile's
    footprint. The z rule is GeMap's and is load-bearing: a 2D centre keeps
    the block's own z (``offset[2]``), so the recentring shift has an exactly
    zero z component and the point cloud's z is left untouched -- GT
    polylines are XY-only downstream (`code_size=2`), so z never reaches a
    regression target either way.
    """
    center = tile.get('center')
    if center is None:
        center = ref.get('tile_center')
    if center is None:
        footprint = tile_footprint(tile, ref, tile_radius)
        if footprint is None:
            raise ValueError(
                f"{tile.get('name')}: --gt-frame tile_center needs a tile "
                "centre, but this export states neither `center`, "
                "`tile_center`, nor any bounds to derive one from")
        center = (np.asarray(footprint[0], dtype=np.float64) +
                  np.asarray(footprint[1], dtype=np.float64)) / 2.0
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    if center.size == 2:
        center = np.array([center[0], center[1], offset[2]], dtype=np.float32)
    return center[:3]


def default_pc_range(tile_radius):
    """The range a training config for this tile size would use: square in
    xy, matching the tile, with the config's z span."""
    radius = tile_radius if tile_radius is not None else FALLBACK_TILE_RADIUS
    z_min, z_max = DEFAULT_Z_RANGE
    return [-radius, -radius, z_min, radius, radius, z_max]


def resolve_classes(selected, manifest):
    """Map ``--classes`` entries (ids or names) onto class_lookup ids.

    Returns a set of int ids, or None for "keep everything". Raises if the
    export has no class taxonomy or a name/id is not in it -- silently
    keeping nothing is how the old --lane-types flag behaved, and it
    produced an empty pkl with no indication why.
    """
    if not selected:
        return None
    lookup = manifest.get('class_lookup') or {}
    if not lookup:
        raise ValueError(
            'this export has no `class_lookup`, so its polylines carry no '
            'class and cannot be filtered; re-run without --classes')
    by_name = {name: int(cid) for cid, name in lookup.items()}
    out = set()
    for entry in selected:
        entry = str(entry)
        if entry in by_name:
            out.add(by_name[entry])
        elif entry.lstrip('-').isdigit() and str(int(entry)) in lookup:
            out.add(int(entry))
        else:
            known = ', '.join(
                f'{cid}={name}' for cid, name in sorted(lookup.items()))
            raise ValueError(
                f'unknown class {entry!r}; this export has: {known}')
    return out


def polyline_class_id(poly):
    """The polyline's class_lookup id, or None on a class-free export.

    Deliberately does not fall back to `type`: that key exists in every
    export and holds a geometry kind ('arc'/'straight'), not a class.
    """
    if poly.get('class_id') is not None:
        return int(poly['class_id'])
    return None


def count_points_in_range(lidar_path, pc_range, z_max, recenter_shift=None):
    """Count a block's points that would survive the training pipeline.

    Mirrors what the model actually sees: ``LoadCarlaPointsFromFile``'s
    recentring (``recenter_shift``, applied when the pkl records one) and
    ``z <= z_max`` filter, then the LiDAR ``Voxelization``'s own
    range filtering. ``GridSamplePoints`` sits between the two but only
    *clamps* grid coordinates -- it never drops points and never moves
    them -- so it cannot change this count.

    Returns:
        tuple[int, int]: raw point count, and count surviving both filters.
    """
    with np.load(lidar_path) as block:
        xyz = np.asarray(block['features'][:, :3], dtype=np.float32)
    n_raw = int(xyz.shape[0])
    if recenter_shift is not None:
        xyz = xyz + np.asarray(recenter_shift, dtype=np.float32)
    if z_max is not None:
        xyz = xyz[xyz[:, 2] <= z_max]
    lo = np.asarray(pc_range[:3], dtype=np.float32)
    hi = np.asarray(pc_range[3:], dtype=np.float32)
    # `>= lo` / `< hi` mirrors hard_voxelize's own
    # `floor((p - lo) / voxel) in [0, grid)` convention rather than
    # PointsRangeFilter.in_range_3d's strict-both-sides test. The two differ
    # only for points exactly on a boundary plane, which can never flip a
    # zero/non-zero verdict in practice.
    n_in_range = int(np.all((xyz >= lo) & (xyz < hi), axis=1).sum())
    return n_raw, n_in_range


def convert_carla_tiles(data_root,
                        split,
                        classes=None,
                        pc_range=None,
                        z_max=DEFAULT_Z_MAX,
                        min_lidar_points=1,
                        lidar_check=True,
                        gt_frame='tile_center'):
    tile_dir, manifest = load_manifest(data_root, split)
    tile_radius = manifest_tile_radius(manifest)
    keep_classes = resolve_classes(classes, manifest)
    if pc_range is None:
        pc_range = default_pc_range(tile_radius)
        if tile_radius is None:
            print(f'[warn] this export states no tile size; assuming '
                  f'tile_radius={FALLBACK_TILE_RADIUS}m for the point-cloud '
                  'range. Pass --lidar-point-cloud-range if that is wrong')
    print(f'{tile_dir}: {len(manifest["tiles"])} tiles, tile_radius='
          f'{tile_radius if tile_radius is not None else "unknown"}, '
          f'gt_frame={gt_frame}, '
          f'pc_range={[round(v, 2) for v in pc_range]}')

    samples = []
    dropped = []
    out_of_bounds = []
    coverage = []
    total_instances = 0
    prog_bar = mmcv.ProgressBar(len(manifest['tiles']))
    for idx, tile in enumerate(manifest['tiles']):
        prog_bar.update()
        name = tile['name']

        # Stored relative to --data-root so the pkl stays valid across
        # containers and mounts; the dataset joins it against its own
        # raw_data_root at load time.
        abs_lidar_path = os.path.join(tile_dir, 'blocks', f'{name}.npz')
        if not os.path.isfile(abs_lidar_path):
            raise FileNotFoundError(abs_lidar_path)
        lidar_path = os.path.relpath(abs_lidar_path, data_root)

        # The polylines below are in WORLD coordinates and must be shifted
        # into the same frame as the LiDAR points the model actually sees.
        # The points as *stored* are in the block's `offset` frame
        # (`points - offset == features[:, :3]` holds exactly), which is NOT
        # the tile centre: the two differ by a mean of ~2.4m (max >7m) on the
        # 25m train split, and by up to ~17m on the 60m grid export.
        #
        # gt_frame='offset' subtracts `offset` from the polylines and shifts
        # nothing else -- GT and points align, but the tile sits displaced
        # from the model's origin-centred pc_range by that same 2-17m, which
        # crops it (measured upstream: 651/4103 tiles under 90% point
        # coverage on the 25m export, far worse on the 30m one).
        #
        # gt_frame='tile_center' (the default, matching the MapTRv2/GeMap
        # benchmark convention) subtracts the tile's own centre from the
        # polylines instead, and records `lidar_recenter_shift =
        # offset - tile_center` on each sample; LoadCarlaPointsFromFile adds
        # that shift to the stored points. Both are translated by the SAME
        # vector, so their relative alignment is untouched -- only the tile's
        # placement relative to the origin changes, to dead centre. Do not
        # "fix" one half without the other: shifting the polylines alone to a
        # different frame than the points is the misalignment this comment
        # used to warn about.
        #
        # np.load is lazy, so this reads only the small `offset` array --
        # it does not pull the full point cloud into memory. (The separate
        # count_points_in_range() call below does read the full `features`
        # array when --no-lidar-check is off; that is the expensive part of
        # this loop, ~14ms for a 110K-point tile.)
        with np.load(abs_lidar_path) as block:
            offset = np.asarray(block['offset'], dtype=np.float32)

        # Loaded before the lidar check (it used to sit after) because
        # tile_center_origin falls back to the json's tile_center/tile_bounds
        # when the manifest tile states no centre.
        ref_path = os.path.join(tile_dir, 'reference_lines',
                                f'{name}_reference_lines.json')
        with open(ref_path, encoding='utf-8') as f:
            ref = json.load(f)

        if gt_frame == 'tile_center':
            origin = tile_center_origin(tile, ref, offset, tile_radius)
            recenter_shift = offset - origin
        else:
            origin = offset
            recenter_shift = None

        n_raw = n_in_range = None
        if lidar_check:
            n_raw, n_in_range = count_points_in_range(abs_lidar_path, pc_range,
                                                      z_max, recenter_shift)
            if n_in_range < min_lidar_points:
                # This tile would voxelize to zero (or near-zero) voxels and
                # crash extract_lidar_feat mid-run. Drop it before its
                # polylines are counted, so the reported instance totals
                # describe what training will actually see.
                dropped.append(
                    dict(
                        name=name,
                        town=tile.get('town'),
                        n_points=n_raw,
                        n_points_in_range=n_in_range))
                continue
            if n_raw:
                coverage.append((n_in_range / n_raw, name))

        divider = []
        for poly in ref['polylines']:
            if keep_classes is not None and \
                    polyline_class_id(poly) not in keep_classes:
                continue
            pts = np.array(poly['points'], dtype=np.float32)
            if pts.shape[0] < 2:
                continue
            divider.append(pts - origin)
            total_instances += 1

        # Sanity check only (tiles are asserted to already be the patch, so
        # no clipping is applied) -- collect, don't crash, if this fires.
        #
        # Compared against the tile's own world-frame footprint shifted into
        # the annotation frame (`footprint - origin`), NOT against
        # +/-tile_radius around zero. Under gt_frame='offset' the origin is
        # the block's `offset`, which is not the tile centre, so a bare
        # radius test fired on most tiles of the 25m export (~1-2m
        # displacement) and essentially all of the 60m one (up to ~17m) --
        # pure noise. Under 'tile_center' the two tests coincide, but the
        # footprint form stays exact for non-square tiles either way.
        footprint = tile_footprint(tile, ref, tile_radius)
        if footprint is not None and divider:
            lo, hi = (footprint[0] - origin[:2] - BOUNDS_MARGIN,
                      footprint[1] - origin[:2] + BOUNDS_MARGIN)
            pts = np.concatenate([p[:, :2] for p in divider])
            overshoot = float(
                np.max(np.maximum(lo - pts, pts - hi), initial=0.0))
            if overshoot > 0:
                out_of_bounds.append(dict(name=name, overshoot=overshoot))

        sample = dict(
            lidar_path=lidar_path,
            sample_idx=name,
            token=name,
            timestamp=idx,
            town=tile.get('town'),
            tile_center=tile.get('center'),
            # the origin the annotation below is actually relative to --
            # recorded explicitly so the frame isn't ambiguous when
            # reading the pkl back. It is `offset` under gt_frame='offset'
            # and the tile centre under 'tile_center'.
            annotation_origin=np.asarray(origin).tolist(),
            gt_frame=gt_frame,
            tile_bounds=tile.get('bounds'),
            # None when --no-lidar-check was passed; the dataset treats
            # a missing count as "unknown" and keeps the sample.
            num_lidar_points=n_raw,
            num_lidar_points_in_range=n_in_range,
            annotation=dict(divider=divider),
        )
        if recenter_shift is not None:
            # What LoadCarlaPointsFromFile must ADD to the stored points
            # (which are always written in the offset frame) to land them in
            # the annotation's frame: stored + (offset - origin) ==
            # world - origin. Same key name and sign convention as the
            # MapTRv2 benchmark repo's converter, so the pkls stay
            # interoperable with its tooling (GeMap's `recenter_shift` is the
            # SAME vector with the opposite sign, subtracted).
            sample['lidar_offset'] = offset.tolist()
            sample['lidar_recenter_shift'] = recenter_shift.tolist()
        samples.append(sample)

    n = max(len(samples), 1)
    print()
    if lidar_check:
        print(f'{split}: {len(samples)} tiles kept, {len(dropped)} dropped '
              f'(<{min_lidar_points} in-range LiDAR point'
              f'{"" if min_lidar_points == 1 else "s"}), {total_instances} '
              f'divider instances ({total_instances / n:.1f} per tile)')
        for entry in dropped[:20]:
            print(f'  [drop] {entry["name"]}: {entry["n_points"]} raw points, '
                  f'{entry["n_points_in_range"]} in range')
        if len(dropped) > 20:
            # A long list usually means the range is wrong, not the data --
            # the sidecar json below has the rest.
            print(f'  [drop] ... and {len(dropped) - 20} more')
    else:
        print(f'{split}: {len(samples)} tiles, {total_instances} divider '
              f'instances ({total_instances / n:.1f} per tile) '
              '[LiDAR check skipped]')
    report_coverage(coverage, pc_range)
    report_out_of_bounds(out_of_bounds)
    return samples, dropped, dict(
        tile_dir=tile_dir,
        tile_radius=tile_radius,
        tile_side=manifest.get('tile_side'),
        gt_frame=gt_frame,
        pc_range=list(pc_range),
        class_lookup=manifest.get('class_lookup') or {},
        classes_kept=sorted(keep_classes) if keep_classes else None,
        n_out_of_bounds=len(out_of_bounds))


def report_coverage(coverage, pc_range):
    """How much of each tile the configured range actually keeps.

    A range narrower than the tile crops it, and because the frame's origin
    is the block `offset` rather than the tile centre, a square range around
    zero is offset from the tile by an amount that grows with tile size --
    so this is the check that catches a range copied from a smaller export.
    """
    if not coverage:
        return
    fracs = np.array([c for c, _ in coverage])
    worst_frac, worst_name = min(coverage)
    poor = int((fracs < 0.9).sum())
    print(f'  [range] median {np.median(fracs):.1%} of raw points fall inside '
          f'{[round(v, 2) for v in pc_range]}; worst tile {worst_name} '
          f'{worst_frac:.1%}')
    if poor:
        print(f'  [range] {poor}/{len(fracs)} tiles keep <90% of their points '
              '-- check the range covers the tile in the `offset` frame')


def report_out_of_bounds(out_of_bounds):
    if not out_of_bounds:
        return
    print(f'  [warn] {len(out_of_bounds)} tile(s) have polyline points '
          f'outside their own tile bounds (+{BOUNDS_MARGIN}m margin):')
    for entry in sorted(
            out_of_bounds, key=lambda e: -e['overshoot'])[:10]:
        print(f'  [warn]   {entry["name"]}: {entry["overshoot"]:.2f}m past '
              'the boundary')
    if len(out_of_bounds) > 10:
        print(f'  [warn]   ... and {len(out_of_bounds) - 10} more')


def main():
    args = parse_args()
    lidar_check = not args.no_lidar_check
    classes = args.classes
    if args.lane_types:
        print('[warn] --lane-types is deprecated and never matched anything; '
              'treating it as --classes (matched against class_lookup)')
        classes = (classes or []) + args.lane_types
    samples, dropped, meta = convert_carla_tiles(
        args.data_root,
        args.split,
        classes,
        pc_range=args.lidar_point_cloud_range,
        z_max=args.z_max,
        min_lidar_points=args.min_lidar_points,
        lidar_check=lidar_check,
        gt_frame=args.gt_frame)
    # Derived from the manifest's tile size inside convert_carla_tiles when
    # not given on the CLI, so read it back rather than re-deriving it here.
    pc_range = meta['pc_range']
    mmcv.mkdir_or_exist(args.out_dir)
    out_path = os.path.join(args.out_dir, f'carla_map_infos_{args.split}.pkl')
    mmcv.dump(
        dict(
            samples=samples,
            split=args.split,
            data_root=args.data_root,
            # Frame of every annotation in `samples` (also recorded per
            # sample); same key as the MapTRv2 benchmark repo's pkls.
            gt_frame=meta['gt_frame'],
            # The tile geometry these samples were built from, so a later
            # reader can tell a 25m export from a 60m one without reopening
            # the source manifest.
            tile_geometry=dict(
                tile_dir=meta['tile_dir'],
                tile_radius=meta['tile_radius'],
                tile_side=meta['tile_side']),
            class_lookup=meta['class_lookup'],
            classes_kept=meta['classes_kept'],
            # Records the geometry the per-sample counts were measured
            # against, so CustomCarlaLocalMapDataset can warn if the config
            # it is running under no longer matches.
            lidar_check=dict(
                enabled=lidar_check,
                point_cloud_range=list(pc_range),
                z_max=args.z_max,
                min_lidar_points=args.min_lidar_points),
            dropped_tiles=dropped),
        out_path)
    print(f'Saved {out_path}')

    if lidar_check:
        # Sidecar report, so a cluster run can be inspected without
        # unpickling the (large) infos file.
        report_path = os.path.join(
            args.out_dir, f'carla_map_infos_{args.split}_dropped.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(
                dict(
                    split=args.split,
                    data_root=args.data_root,
                    tile_radius=meta['tile_radius'],
                    point_cloud_range=list(pc_range),
                    z_max=args.z_max,
                    min_lidar_points=args.min_lidar_points,
                    n_kept=len(samples),
                    n_dropped=len(dropped),
                    dropped=dropped),
                f,
                indent=2)
        print(f'Saved {report_path}')


if __name__ == '__main__':
    main()
