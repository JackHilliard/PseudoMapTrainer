"""Shared helpers for reading the CARLA road-polyline tile dataset layout.

Expected directory layout::

    <data_root>/<split>/manifest.json
    <data_root>/<split>/blocks/<tile_name>.npz
    <data_root>/<split>/reference_lines/<tile_name>_reference_lines.json
"""

import json
import os


def read_carla_manifest(data_root, split):
    """Load ``<data_root>/<split>/manifest.json``."""
    manifest_path = os.path.join(data_root, split, 'manifest.json')
    with open(manifest_path, encoding='utf-8') as f:
        return json.load(f)
