Use this corrected README.md:

# UAD_GUI_APP

Tkinter desktop app for unsupervised anomaly detection on selected MVTec AD categories.

## Dataset setup

MVTec AD dataset: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

### Create this folder

```text
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


Environment setup
Windows
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.ui.tkinter_app
Run

From the project root:

python -m src.ui.tkinter_app

