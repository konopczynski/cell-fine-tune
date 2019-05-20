# cell-fine-tune

## instruction
### 0.
install the requirements:
```
pip install -r requirements.txt
```
### 1.
download weights from the folder final_weights and put them in folder ```(../)cell-fine-tune/weights```<br>
https://drive.google.com/drive/folders/1Ngp34zO8bx39z0ZSqwtCpZoCjRCEpc24?usp=sharing
### 2.
Run mask prediction:
```
python infer.py (--model <hybrid/pure>) (--image_folder <>) (--save_folder <>)
```
```--model hybrid``` runs our Hybrid model

```--model pure``` runs MaskRCNN
### Based on
https://github.com/Confusezius/unet-lits-2d-pipeline

https://github.com/matterport/Mask_RCNN
