import json 
with open('notebooks/eda.ipynb', 'r', encoding='utf-8') as f: 
    data = json.load(f) 
 
print('=== NOTEBOOK ANALYSIS ===') 
print(f'Total cells: {len(data["cells"])}') 
 
viz_keywords = ['plt.', 'sns.', 'plot(', 'hist(', 'boxplot', 'heatmap', 'bar('] 
viz_cells = 0 
 
for i, cell in enumerate(data['cells']): 
    if cell['cell_type'] == 'code': 
        source = ''.join(cell['source']).lower() 
        if any(keyword in source for keyword in viz_keywords): 
            viz_cells += 1 
            print(f'  Cell {i}: Has visualization code') 
            print(f'  Cell {i}: Has visualization code') 
    print('??  PROBLEM: Notebook has few or no visualizations!') 
    print('This matches feedback: "no visualizations provided"') 
else: 
    print('? Notebook appears to have visualizations') 
