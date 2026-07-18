import json

def extract_code():
    with open('/Users/ali/workspace/polarimetry_final/2025-09-28/ngc7538_polarimetry.ipynb') as f:
        nb = json.load(f)
        
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code_cells.append("".join(cell['source']))
            
    print(f"Extracted {len(code_cells)} code cells.")
    
    with open('extracted_notebook_code.py', 'w') as f_out:
        for i, code in enumerate(code_cells):
            f_out.write(f"\n# ==========================================\n# CELL {i}\n# ==========================================\n")
            f_out.write(code)
            f_out.write("\n")
            
    print("Saved code to extracted_notebook_code.py")

if __name__ == "__main__":
    extract_code()
