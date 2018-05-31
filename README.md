# cell-fine-tune
## requirements
python==3.6 <br>
keras>=2.1.6<br>
scikit-image>=0.13.1 <br>
## instruction
### 0.
download weights in h5 format and put them in folder ./weights<br>
the current best weights are: ch1_ch2_new100_150epochs_coco_berkeley.h5
https://drive.google.com/drive/folders/1_qL4oyVcPaOiHFTCogo3kXgs5g9FsLuG
### 1.
inside infer.py set:<br>
MODEL_PATH<br>
IMAGE_DIR<br>
SAVE_DIR
### 2.
run python infer.py
