# cell-fine-tune
## requirements
python==3.6 <br>
keras>=2.2.0<br>
scikit-image>=0.13.1 <br>
## instruction
### 0.
install the requirements:
```
pip install -r requirements.txt
```
### 1.
download weights from the folder final_weights and put them in folder ./weights<br>
https://drive.google.com/drive/folders/1Ngp34zO8bx39z0ZSqwtCpZoCjRCEpc24?usp=sharing
### 2.
Run mask prediction:
```
python infer.py --model <hybrid/pure>
```
