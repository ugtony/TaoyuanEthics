# Taoyuan Ethics

## Installization
1. Install python 
- https://www.python.org/downloads/
2. Install packages
- download [this repository](https://github.com/ugtony/TaoyuanEthics)
- open terminal in project folder "TaoyuanEthics" by `cmd`, and then execute
   `pip install -r requirements.txt`

## Run Program
This repository contains three programs:

1. **`detect_video_sreamlit.py`**  
    A streamlit website demo. 
    `streamlit run detect_video_streamlit.py --server.maxUploadSize 2048`.  
    set server.maxUploadSize to 2048 to increase upload size limit to 2GB


2. **`detect_video.ipynb`**  
    A Jupyter Notebook demo.
    Open Jupyter Lab interface by `jupyter lab`, and open detect_video.ipynb in browser.

3. **`run_app.py/app.py`**  
    ready to be packaged by pyinstaller.
    `pyinstaller run_app.spec`

We would like to express our sincere appreciation to [Pexels](https://www.pexels.com/) for the high-quality video footage that enhances this project.