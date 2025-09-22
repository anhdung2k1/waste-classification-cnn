## Requirements
You should use python stable version (3.9 <-> 3.11) for using the GPU for training the model
Latest python version or less is not support the GPU selection automatically yet.
If you manage to run in MacOS

pip install tensorflow-macos tensorflow-metal
## Notebook
The .ipynb file you can run either in local or google colab
If you manage to run in local. You should create venv folder
```bash
python -m venv venv
source venv/bin/activate
```
Install the libs required
```bash
pip install -r requirements.txt
```
If you start the jupyter notebook locally
```bash
jupyter lab
```
When you running all cells in the notebook, you will have two models for resnet50 and vgg16 for waste classification
```
resnet50_garbage_classifier_model.keras
vgg16_garbage_classifier_model.keras
```

## Run real-time detection
```bash
source venv/bin/activate
python detection.py
```
It used the YOLO as object detection, once the object is detected will output the bbox image and then called the classifier model which was trained by the notebook to classify the waste name in real-time.