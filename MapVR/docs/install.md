# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n maptr python=3.8 -y
conda activate maptr
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# Code requries: torch>=1.10
```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-5 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install setuptools==59.5.0
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install other packages.**
```shell
pip install timm
pip install networkx==2.2
pip install scikit-image==0.19.0
pip install numpy==1.23.5
pip install scikit-learn==1.3.0
pip install yapf==0.40.1
pip install PuLP==2.9.0 # our linear program solver
pip install pandas
pip install ipython
pip install ninja
```

**f. Install mmdet3d, GKT, and the MapVR rasterizer**
```shell
cd ~/PseudoMapTrainer/MapVR/mmdetection3d
python setup.py develop

cd ~/PseudoMapTrainer/MapVR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install

cd ~/PseudoMapTrainer/MapVR/projects/mmdet3d_plugin/maptr/losses/diff_ras/
python setup.py develop
```

**g. Install other requirements.**
```shell
cd  ~/PseudoMapTrainer/MapVR
pip install -r requirement.txt
```

**h. Prepare pretrained models.**
```shell
cd ~/PseudoMapTrainer/MapVR
mkdir ckpts

cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

