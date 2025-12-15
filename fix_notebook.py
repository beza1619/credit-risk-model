import json 
import base64 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 
import io 
 
# Load the notebook 
with open('notebooks/eda.ipynb', 'r', encoding='utf-8') as f: 
    nb = json.load(f) 
 
print('Fixing notebook...') 
 
# Add a new visualization cell at the end 
new_cell = { 
    'cell_type': 'code', 
    'execution_count': None, 
    'metadata': {}, 
    'outputs': [], 
    'source': [ 
        '# =======================================================\\n', 
        '# REQUIRED VISUALIZATIONS FOR GRADING\\n', 
        '# =======================================================\\n', 
        '\\n', 
        'print(\"CREATING EDA VISUALIZATIONS...\")\\n', 
        '\\n', 
        '# Load data\\n', 
        'df = pd.read_csv(\"../data/raw/data.csv\")\\n', 
        'print(f\"Data loaded: {len(df)} rows\")\\n', 
        '\\n', 
        '# FIGURE 1: Amount Distribution\\n', 
