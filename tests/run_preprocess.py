# tests/run_preprocess.py

###
### File that runs the data preprocessing functions
###

#imports
from src.utils.preprocess import preprocess_and_save

#code
raw_data_dir = "data/tiny-imagenet-200"
out_file = "data/preprocessed/tinyimagenet_preprocessed.pt"

preprocess_and_save(raw_data_dir, out_file)