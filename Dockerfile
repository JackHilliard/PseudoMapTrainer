# PseudoMapTrainer (ICCV 2025) -- "Learning Online Mapping without HD Maps"
#
# One image, three conda environments -- one per pipeline stage, mirroring the
# three isolated environments the README sets up. Nothing in this repo imports
# across the three directories, so they share only the CUDA/PyTorch base:
#
#   base ("maptr")  MapVR/        MapTRv2 + MapVR online mapping model
#   rogs            RoGS/         2D-Gaussian road reconstruction -> pseudo-labels
#   mask2former     Mask2Former/  perspective-view semantic segmentation
#
# The base (default) environment is the MapVR one, so `docker run` lands
# straight in the training environment; the other two are conda envs:
#
#   conda activate rogs          # or: conda run -n rogs python ...
#   conda activate mask2former
#
# --- Why this deviates from the README's pinned versions ---
# Every documented environment predates Hopper: MapVR pins torch 1.10.0+cu111,
# RoGS 1.13.1+cu116, Mask2Former 1.10.1+cu113. CUDA 11.8 is the first release
# with real sm_90 support, so no TORCH_CUDA_ARCH_LIST tweak can rescue any of
# those toolchains on an H100 -- all three are bumped onto one torch
# 2.1.0+cu118 base here, following the same fix already applied to the sibling
# MapTRv2 and GeMap codebases. The source changes that fallout required
# (THC/THC.h removal, C++17, spconv2, --local-rank, FORCE_CUDA) are committed
# alongside this Dockerfile.
#
# --- Build / run ---
#   docker build -t pseudomaptrainer .
#   docker run --gpus all --shm-size=16g -it \
#     -v /path/to/nuscenes:/data/nuscenes \
#     -v /path/to/m2f_infer:/data/m2f_infer \
#     -v /path/to/outputs:/workspace/PseudoMapTrainer/RoGS/output \
#     pseudomaptrainer
#
# The paths inside the configs are hard-coded and not derived from each other
# (Mask2Former --save_dir == RoGS label_dir == MapVR data_root_seg, etc.), so
# mount the host directories and then point the configs at the mount points.

ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 8.6 covers Ampere (e.g. RTX 3070/3090); 9.0+PTX covers Hopper (H100) and,
# via PTX forward-compat JIT, anything newer released after this image (e.g.
# Blackwell RTX 50-series).
ENV TORCH_CUDA_ARCH_LIST="8.6 9.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        nano \
        ninja-build \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the two extra environments off the pristine base *before* anything is
# installed into it, so they inherit torch/CUDA (hardlinked from the shared
# conda package cache, so nearly free) but none of MapVR's mmcv/mmdet stack.
RUN conda create -y -n rogs --clone base \
    && conda create -y -n mask2former --clone base \
    && conda init bash \
    && conda clean --all -y

# =============================================================================
# 1) base env -- MapVR (MapTRv2 + MapVR), see MapVR/docs/install.md
# =============================================================================

# Install MMCV, MMDetection and MMSegmentation. Bumped from the versions in
# MapVR/docs/install.md (mmcv-full==1.4.0/mmdet==2.14.0/mmsegmentation==0.14.1,
# built for CUDA 11.1) to support Hopper (H100) GPUs, which need CUDA>=11.8 --
# the first CUDA release with real sm_90 support. mmdet/mmsegmentation are
# bumped alongside mmcv because both hard-assert an mmcv upper bound in their
# own __init__.py that the old pins don't cover; 2.28.2/0.30.0 are the last
# mmdet 2.x/mmsegmentation 0.x releases, with mmcv upper bounds of 1.8.0,
# comfortably covering 1.7.2. (MapVR/mmdetection3d/mmdet3d/__init__.py asserts
# its own mmcv upper bound too -- raised to 1.8.0 in this commit.)
#
# mmcv-full MUST be built from source here, not installed from OpenMMLab's
# prebuilt wheel index. Their prebuilt cu118/torch2.1.0 wheel was compiled
# with OpenMMLab's own (unknown, non-configurable) TORCH_CUDA_ARCH_LIST,
# which does not include Hopper (sm_90) and has no +PTX fallback baked in --
# this is invisible on Ampere and only surfaces on an actual H100 as
# `RuntimeError: CUDA error: no kernel image is available for execution on
# the device` inside mmcv's own CUDA ops (e.g. ms_deform_attn,
# sigmoid_focal_loss) the first time they run. Building from source makes
# mmcv's setup.py pick up this Dockerfile's own TORCH_CUDA_ARCH_LIST (set
# above, "8.6 9.0+PTX"), same as every other custom op in this image.
RUN MMCV_WITH_OPS=1 pip install --no-cache-dir --no-binary mmcv-full "mmcv-full==1.7.2"
# mmcv 1.7.2's single-GPU MMDataParallel path (mmcv.parallel.Scatter.forward,
# used by the non-distributed branch of custom_tools/{train,test}.py) calls
# PyTorch's private torch.nn.parallel._functions._get_stream() with a raw int
# device id; torch>=2.x's version of that function requires a torch.device
# object instead (`AttributeError: 'int' object has no attribute 'type'`).
# Patch the installed file to wrap the device id.
RUN sed -i \
    "s/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device('cuda', device) if isinstance(device, int) else device) for device in target_gpus]/" \
    /opt/conda/lib/python3.10/site-packages/mmcv/parallel/_functions.py
RUN pip install --no-cache-dir mmdet==2.28.2 mmsegmentation==0.30.0
# Unpinned `timm` pulls in the latest release, which needs torch.fx APIs not
# present in older torch; pin to a version contemporaneous with this codebase.
RUN pip install --no-cache-dir timm==0.6.13

# spconv2 (maintained fork) replaces mmdetection3d's vendored spconv 1.x,
# whose CUDA kernels are incompatible with CUDA 11.8/sm_86+90 (crash with
# "cuda execution failed with error 2" even on trivial inputs) -- see
# MapVR/mmdetection3d/mmdet3d/ops/spconv/__init__.py, which re-exports this
# instead of the vendored implementation.
RUN pip install --no-cache-dir spconv-cu118==2.3.8

# The rest of MapVR/docs/install.md step (e), with two deviations:
#   * networkx==2.2 is dropped -- it still does `from collections import
#     Mapping`, removed in Python 3.10 (mmdetection3d/requirements/runtime.txt
#     now pins the last 2.x release, 2.8.8, instead).
#   * scikit-image==0.19.0 -> 0.19.3: same minor series, but the first with
#     cp310 wheels.
# PuLP is the linear-program solver behind MaskedMapTRAssigner's allow_split
# matching (projects/mmdet3d_plugin/maptr/assigners/solver.py).
RUN pip install --no-cache-dir \
        PuLP==2.9.0 \
        ipython \
        ninja \
        pandas \
        scikit-image==0.19.3 \
        scikit-learn==1.3.0 \
        yapf==0.40.1

# nuScenes data prep / evaluation needs the devkit; descartes/seaborn are
# imported by the map rendering and log-analysis tools respectively.
RUN pip install --no-cache-dir nuscenes-devkit descartes seaborn

# =============================================================================
# 2) rogs env -- RoGS pseudo-label generation (README section 1.2)
# =============================================================================

# README pins for this environment, minus torch (2.1.0+cu118 comes from the
# cloned base) and minus mayavi, which is only used by RoGS's optional 3D
# GUI visualisation and is already import-guarded in train.py/models/road.py.
# opencv is installed headless: cv2 is only used for image IO here, and the
# GUI build links libGL.so.1, which breaks on Singularity/Apptainer clusters
# with --nv (the host's newer libGL gets bind-mounted over the container's).
RUN /opt/conda/envs/rogs/bin/pip install --no-cache-dir \
        addict==2.4.0 \
        numpy==1.24.4 \
        nuscenes-devkit==1.1.11 \
        opencv-contrib-python-headless==4.11.0.86 \
        opencv-python-headless==4.11.0.86 \
        plyfile==1.0.3 \
        pyquaternion==0.9.9 \
        pyrotation==0.0.2 \
        pytz \
        pyyaml==6.0.2 \
        scikit-learn==1.3.0 \
        scipy==1.10.1 \
        shapely \
        tensorboard==2.14.0 \
        tqdm==4.66.5

# pytorch3d built from source rather than installed from the prebuilt wheel
# index, for the same reason mmcv-full is: those wheels' baked-in arch list is
# not ours and cannot be assumed to carry sm_90 or PTX. v0.7.5 rather than the
# README's 0.7.2 -- 0.7.2 predates torch 2.0 and does not compile against
# torch 2.1's C++17 ATen headers. MAX_JOBS caps peak build memory (each nvcc
# TU here is heavy; 16 in parallel OOMs a 32 GB machine).
RUN /opt/conda/envs/rogs/bin/pip install --no-cache-dir fvcore iopath \
    && git clone --depth 1 --branch v0.7.5 https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d \
    && cd /tmp/pytorch3d \
    && MAX_JOBS=8 /opt/conda/envs/rogs/bin/python setup.py install \
    && cd / && rm -rf /tmp/pytorch3d

# The two Gaussian rasterizers RoGS renders with (README section 1.2, steps
# 2-3): the stock one for RGB, and a second copy compiled with NUM_CHANNELS=6
# for the 6-class BEV label space (RoGS/datasets/label_mappings/
# mapillary_vistas.py: 0=mask, 1=lane marking, 2=crosswalk, 3=road,
# 4=boundary, 5=background).
RUN git clone --recursive https://github.com/fzhiheng/diff-gs-depth-alpha.git /tmp/diff-gs-rgb \
    && cd /tmp/diff-gs-rgb \
    && git checkout 486d1882497d8890888222ea8252a59964ec5dfc \
    && git submodule update --init --recursive \
    && /opt/conda/envs/rogs/bin/python setup.py install \
    && cd / && rm -rf /tmp/diff-gs-rgb

# The label variant is renamed to `diff_gs_label`, which is what RoGS actually
# imports (RoGS/utils/render.py). Note the README says to rename the package
# directory to `diff-gs-label`; a hyphenated directory is not importable, so
# the underscore spelling the code uses is used here. There is deliberately no
# third `diff_gs_label2` build: RoGS/utils/render.py only reaches for it when
# the label tensor has 5 channels, and this pipeline always has 6.
RUN git clone --recursive https://github.com/fzhiheng/diff-gs-depth-alpha.git /tmp/diff-gs-label \
    && cd /tmp/diff-gs-label \
    && git checkout 486d1882497d8890888222ea8252a59964ec5dfc \
    && git submodule update --init --recursive \
    && sed -i -E 's/^#define NUM_CHANNELS[[:space:]]+[0-9]+.*$/#define NUM_CHANNELS 6/' cuda_rasterizer/config.h \
    && grep -q '^#define NUM_CHANNELS 6$' cuda_rasterizer/config.h \
    && mv diff_gaussian_rasterization diff_gs_label \
    && sed -i 's/diff_gaussian_rasterization/diff_gs_label/g' setup.py \
    && /opt/conda/envs/rogs/bin/python setup.py install \
    && cd / && rm -rf /tmp/diff-gs-label

# =============================================================================
# 3) mask2former env -- PV segmentation (README section 1.1)
# =============================================================================

# detectron2 has no release supporting torch 2.x (v0.6 is from 2021), so it is
# built from a pinned main-branch commit. Built from source anyway -- the
# prebuilt index only ever covered old torch/CUDA pairs, and its CUDA ops need
# this image's arch list like everything else here.
RUN /opt/conda/envs/mask2former/bin/pip install --no-cache-dir \
        "git+https://github.com/facebookresearch/detectron2.git@b4a4a3bd136852dae5fb1de37978dee412653e31"
# panopticapi is imported lazily by the panoptic dataset mappers; cheap enough
# to have present. (cityscapesScripts, also in Mask2Former/INSTALL.md, is only
# needed for the Cityscapes datasets this pipeline never touches.)
RUN /opt/conda/envs/mask2former/bin/pip install --no-cache-dir \
        "git+https://github.com/cocodataset/panopticapi.git"

# =============================================================================
# 4) the repo itself, and every in-tree CUDA op
# =============================================================================

COPY . /workspace/PseudoMapTrainer
WORKDIR /workspace/PseudoMapTrainer

# --- MapVR (base env) -------------------------------------------------------
# Pre-install mmdetection3d's runtime deps via pip (wheels) first: `setup.py
# develop` otherwise falls back to setuptools' legacy easy_install for unmet
# deps, and current scikit-image no longer ships an sdist it can build.
RUN pip install --no-cache-dir -r MapVR/mmdetection3d/requirements/runtime.txt
RUN cd MapVR/mmdetection3d && python setup.py develop
# MapTRv2's Geometric Kernel Attention op.
RUN cd MapVR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn \
    && python setup.py build install
# MapVR's differentiable rasterizer, behind RenderedMaskDiceLoss.
RUN cd MapVR/projects/mmdet3d_plugin/maptr/losses/diff_ras \
    && python setup.py develop
RUN pip install --no-cache-dir -r MapVR/requirement.txt

# av2/mmdet/mmcv-full still use the np.float and np.int aliases removed in
# numpy 1.24. Pinned after everything else in this env so it sticks.
RUN pip install --no-cache-dir "numpy==1.23.5"

# ...which makes the pyarrow that av2 pulls in unusable. av2 requires pyarrow
# with no version bound, so it resolves to the latest (25.x), whose wheel is
# built against the numpy 2 C-API and declares no numpy requirement of its own
# -- so pip reports no conflict, and the mismatch only shows up at import time
# as a bare SIGSEGV with no Python traceback (reached here via
# pandas.compat.pyarrow while importing projects.mmdet3d_plugin). 12.0.1 is a
# numpy-1.x-era release and satisfies av2's unbounded requirement.
RUN pip install --no-cache-dir "pyarrow==12.0.1"

# MapVR/requirement.txt pins shapely==1.8.5.post1, but nuscenes-devkit
# (unpinned shapely dependency) silently upgrades it to 2.x. shapely 2.0
# changed STRtree.query() to return integer indices instead of geometry
# objects -- a breaking change that crashes the chamfer-distance evaluation
# code this repo's map metrics rely on. Re-pin as the last install so it
# sticks.
RUN pip install --no-cache-dir "shapely==1.8.5.post1"

# --- Mask2Former (mask2former env) ------------------------------------------
RUN /opt/conda/envs/mask2former/bin/pip install --no-cache-dir \
        -r Mask2Former/requirements.txt
# MSDeformAttn CUDA kernel (Mask2Former/INSTALL.md's `sh make.sh`). Its
# setup.py honours FORCE_CUDA, which is set image-wide above -- `docker build`
# has no GPU, so torch.cuda.is_available() is False here.
RUN cd Mask2Former/mask2former/modeling/pixel_decoder/ops \
    && /opt/conda/envs/mask2former/bin/python setup.py build install

# =============================================================================
# 5) headless opencv everywhere
# =============================================================================

# mmcv-full/mmdet/nuscenes-devkit/detectron2 all transitively pull in
# non-headless opencv-python, which links its GUI backend against libGL.so.1.
# On Singularity/Apptainer clusters with --nv, the host's newer libGL.so.1 gets
# bind-mounted over the container's own and fails to load against this image's
# older glibc (GLIBC_ABI_DT_RELR, added in glibc 2.36, missing here). Swap to
# the headless build -- same OpenCV, no GL/X11 linkage at all -- so cv2 never
# touches libGL.so.1 regardless of what the host injects.
#
# Pinned to 4.11.0.86 (the version RoGS's own environment pins) rather than
# mirroring whatever opencv-python the transitive resolution landed on:
# unpinned, that is now opencv-python 5.x, which requires numpy>=2 and so
# cannot coexist with the numpy==1.23.5 that mmcv-full/mmdet still need for
# their np.float/np.int aliases.
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python \
    && pip install --no-cache-dir "opencv-python-headless==4.11.0.86"
RUN /opt/conda/envs/mask2former/bin/pip uninstall -y \
        opencv-python opencv-python-headless opencv-contrib-python \
    && /opt/conda/envs/mask2former/bin/pip install --no-cache-dir \
        "opencv-python-headless==4.11.0.86"

# utils/dataset_viewer.py is a standalone flask viewer (no torch, no mmdet3d,
# no GPU) that is normally run outside the container; matplotlib and numpy are
# already here transitively, so only flask is missing.
RUN pip install --no-cache-dir flask

# Mount points for the things that do not belong in an image: the nuScenes
# dataset + CAN bus expansion, the Mask2Former inference output, the RoGS
# pseudo-labels, and pretrained/trained checkpoints (MapVR/docs/install.md
# step h: resnet50-19c8e357.pth and resnet18-f37072fd.pth go in MapVR/ckpts).
RUN mkdir -p MapVR/data MapVR/ckpts RoGS/output Mask2Former/output

CMD ["/bin/bash"]
