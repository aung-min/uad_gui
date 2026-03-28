# UAD-DEMO
## Unsupervised Industrial Anomaly Detection using MVTec AD <br>


## Inspection of bottle<br>
![App Demo (Inspection of bottle)](assets/UAD-bottle.gif)

## Inspection of transistor and evaluation dashboard <br>
![App Demo (Inspection of transistor and Evaluation Dashboard)](assets/UAD-Transistor.gif)

## Evaluation dashboard <br>
![App Demo (Evaluation Dashboard)](assets/UAD-Evaluation-Dashboard.gif)


### Feature Memory Bank (Static Photo)
![UAD Results Analysis](assets/UAD-Feature-Memory-Flow.png)
#### Figure 1. Feature-Based Anomaly Detection Pipeline Using a ResNet18 Multi-Layer Memory Bank <br>


### Custom CNN Autoencoder (Static Photo)
![UAD Results Analysis](assets/UAD-Autoencoder-Flow.png)
#### Figure 2. Reconstruction-Based Anomaly Detection Pipeline Using a Convolutional Autoencoder <br>


### Hybrid Detector (Static Photo)
![UAD Results Analysis](assets/UAD-Hybrid-Flow.png)
#### Figure 3. Hybrid Anomaly Detection Pipeline Combining Feature Memory Bank and Autoencoder-Based Reconstruction <br>




## Refer to the following steps to run UAD-GIU App locally! <br>

### Dataset setup
MVTec AD dataset: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

### Create this folder
<pre>
data/
  mvtec_ad/
</pre>

### Download the MVTec AD dataset, extract the zip file, and copy these folders into data/mvtec_ad
<pre>
data/
  mvtec_ad/
    bottle/
    screw/
    metal_nut/
    capsule/
    cable/
    transistor/
src/
</pre>

### Environment setup (Windows)
<pre>
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
</pre>

### Run from the project root:
<pre>
python -m src.ui.tkinter_app
</pre>

