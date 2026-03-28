##  Tkinter desktop app for unsupervised anomaly detection on selected MVTec AD categories.

### Dataset setup

MVTec AD dataset: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

### Create this folder
<pre>
data/
  mvtec_ad/
Download the MVTec AD dataset, extract the zip file, and copy these folders into data/mvtec_ad
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

Environment setup
Windows
<pre>
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
</pre>

###Run
From the project root:
<pre>
python -m src.ui.tkinter_app
</pre>

Your main issue was using ``text` instead of triple backticks like:

