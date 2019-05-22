# cell-fine-tune

## instruction
### Installaion
install the requirements:
```
pip install -r requirements.txt
```
### Setup
Download all files (weights) from the folder final_weights from the google drive and put them in the folder ```(../)cell-fine-tune/weights```<br>
https://drive.google.com/drive/folders/1Ngp34zO8bx39z0ZSqwtCpZoCjRCEpc24?usp=sharing
### Running
Run mask prediction:
```
python infer.py (--model <hybrid/pure>) (--image_folder <>) (--save_folder <>)
```
```--model hybrid``` runs our Hybrid model

```--model pure``` runs MaskRCNN

### Evaluation
Basic statistics can be generated using the notebook: evaluation/evaluate_prediction.ipynb

## Issues
In case of problems with libomp on Mac (https://github.com/pytorch/pytorch/issues/20030), try:
```
brew install libomp
```
### Based on
https://github.com/Confusezius/unet-lits-2d-pipeline <br>
https://github.com/matterport/Mask_RCNN
