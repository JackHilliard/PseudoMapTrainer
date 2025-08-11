# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from RoMe (188fd42)
#   (https://github.com/DRosemei/RoMe/blob/188fd42d0bf5324d6dd7a48758eef6c81a04d8fe/scripts/mask2former_infer/nuscenes_scenes.py)
# Copyright (c) 2023 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from os.path import join
from nuscenes.nuscenes import NuScenes

def concat(list_of_lists):
    """Concatenate a list of lists into a single list."""
    return sum(list_of_lists, [])

def crawl_nusc_scenes_paths(root_dir, camera_names, version=None, sel_scenes:list=[]):
    """
    Return absolute paths to NuScenes camera images, chronologically ordered.
    Parameters
    ----------
    root_dir : str
        Directory that contains the `samples` folder.
    camera_names : list[str]
        Cameras of interest (e.g. ["CAM_FRONT", "CAM_BACK", ...]).
    version : str | None
        Which version to use. If `None`, the function will return the paths
        of all camera images of the dataset.
    sel_scenes : list[str]
        Explicit list of selected scene names. If omitted, a percentage
        slice is used.
    Returns
    -------
    list[str]
        Chronologically ordered image paths across all requested scenes.
    """
    if version is None:
        nusc_all = [
            NuScenes(dataroot=root_dir, version="v1.0-trainval"),
            NuScenes(dataroot=root_dir, version="v1.0-test")
        ]
    else:
        nusc_all = [NuScenes(dataroot=root_dir, version=f"v1.0-{version}")]
    scenes_all = concat([[(scene, nusc) for scene in nusc.scene] for nusc in nusc_all])
    if len(sel_scenes) == 0:
        sel_scenes = [
            scene["name"] for scene, _ in scenes_all
        ]
    
    paths = []
    for scene, nusc in scenes_all:
        scene_name = scene["name"]
        if scene_name not in sel_scenes:
            continue
        records = [samp for samp in nusc.sample if
                    nusc.get("scene", samp["scene_token"])["name"] in scene_name]
        # sort by timestamp (only to make chronological viz easier)
        records.sort(key=lambda x: (x['timestamp']))
        # interpolate images from 2HZ to 12 HZ
        for index in range(len(records)):
            rec = records[index]
            for cam in camera_names:
                # compute camera key frame poses
                rec_token = rec["data"][cam]
                samp = nusc.get("sample_data", rec_token)
                flag = True  
                # compute first key frame and framse between first frame and second frame
                while flag or not samp["is_key_frame"]: 
                    flag = False
                    rel_camera_path = samp["filename"]
                    camera_path = join(root_dir, rel_camera_path)
                    paths.append(camera_path)
                    if samp["next"] != "":
                        samp = nusc.get('sample_data', samp["next"])
                    else:
                        break
    return paths


if __name__ == "__main__":
    # locations "boston-seaport", "boston-seaport", "singapore-queensto", "singapore-hollandv"
    root_dir = "#####/Nuscenes"
    version = "trainval"
    scenes = ["scene-0546", "scene-0556", "scene-0558", "scene-0769"]
    camera_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    paths = crawl_nusc_scenes_paths(root_dir, version, camera_names, scenes)