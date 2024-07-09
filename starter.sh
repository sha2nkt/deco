#!/bin/bash

eval "$(conda shell.bash hook)"
apt-get install libnvidia-gl-515-server -y
apt-get install nvidia-utils-515 -y


conda create -n deco python=3.9 -y

conda activate deco
# pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7  -c pytorch -c nvidia
pip install fvcore
# git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d 
# pip install .
# cd ..

# pip install -r requirements.txt
pip install opencv-python loguru monai==1.0.1  pyrender==0.1.33 smplx==0.1.28 scikit-learn scikit-image jpeg4py matplotlib pandas configargparse flatten_dict chumpy numpy==1.23.1 yacs networkx==2.2 charset-normalizer==3.1.0
bash fetch_data.sh

echo "Please register and agree to the license of DECO on: “https://deco.is.tue.mpg.de/"

# Prompt for username
echo "Enter username for DECO:"
read username

# Prompt for password without showing it
echo "Enter password for DECO:"
read -s password


wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=deco&resume=1&sfile=Release_Damon.tar.gz' -O Release_Damon.tar.gz

tar -xvf Release_Damon.tar.gz  && rm -r Release_Damon.tar.gz

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=deco&resume=1&sfile=damon_segmentations.tar.gz' -O damon_segmentations.tar.gz

tar -xvf damon_segmentations.tar.gz && rm -r damon_segmentations.tar.gz

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=deco&resume=1&sfile=hot_polygon_contact.tar.gz' -O hot_polygon_contact.tar.gz

tar -xvf hot_polygon_contact.tar.gz && rm -r hot_polygon_contact.tar.gz

###############

echo "Please register and agree to the license of the HOT dataset on: “https://hot.is.tue.mpg.de/"
# Prompt for username
echo "Enter username for the HOT dataset:"
read username

# Prompt for password without showing it
echo "Enter password for the HOT dataset:"
read -s password

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=hot&resume=1&sfile=HOT-Annotated.zip' -O HOT-Annotated.zip



# we install SMPL-X
pip install opencv-python loguru monai==1.0.1  pyrender==0.1.33 smplx==0.1.28 scikit-learn scikit-image jpeg4py matplotlib pandas configargparse flatten_dict chumpy numpy==1.23.1 yacs networkx==2.5 charset-normalizer==3.1.0


mv Release_Datasets/damon/ datasets/Release_Datasets/damon

# we install MKL
conda install  mkl mkl-service

# Pytorch3d is needed
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia 
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install .
cd ..
pip install configargparse
pip install chardet==4.0.0 idna==2.10 requests==2.25.1 urllib3==1.26.19
pip install networkx==2.5

# Login to SMPL-X Website
########### SMPL-X ###########

######
#
#
# We need the last SMPL-X
#
cd ..
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


read -p "Username (SMPl-X Website - https://smpl-x.is.tue.mpg.de/):" username
read -p "Password (SMPl-X Website - https://smpl-x.is.tue.mpg.de/):" -s password

username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip&resume=1' -O 'models_smplx_v1_1.zip' --no-check-certificate --continue
unzip models_smplx_v1_1.zip


# we need to move this into the data
cp models/smplx/SMPLX_NEUTRAL.npz deco/data/smplx/SMPLX_NEUTRAL.npz

########### SMPL ###########
echo "Downloading SMPL Models"

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

username=$(urle $username)
password=$(urle $password)

# Login to SMPL Website
# If you don't have an account, create one here: https://smpl.is.tue.mpg.de/
read -p "Username (SMPL Website - https://smpl.is.tue.mpg.de/):" username
read -p "Password (SMPL Website - https://smpl.is.tue.mpg.de/):" -s password

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip&resume=1' -O 'models_smpl.zip' --no-check-certificate --continue
unzip models_smpl.zip
mv SMPL_python_v.1.1.0 smpl
cp smpl/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl deco/data/smpl/SMPL_NEUTRAL.pkl


####################################
# 
# we launch the code.
#

cd deco

# we move the HOT datasets in the proper paths
unzip HOT-Annotated.zip
mv HOT-Annotated datasets/HOT-Annotated
mv polygon_contact datasets/HOT-Annotated/polygon_contact
mv parts datasets/HOT-Annotated/parts
mv segmentation_masks datasets/HOT-Annotated/segmentation_masks


# we launch the the inference

echo "Part 1: inference launch"

python inference.py \
    --img_src example_images \
    --out_dir demo_out

echo "Part 2: Tester Script"

python tester.py --cfg configs/cfg_test.yml