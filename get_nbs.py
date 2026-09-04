import os, json
out = ''
for f in sorted(os.listdir('notebooks')):
    if f.endswith('.ipynb'):
        out += f'\n--- {f} ---\n'
        try:
            nb = json.load(open('notebooks/'+f, encoding='utf-8'))
            for cell in nb.get('cells', []):
                if cell['cell_type'] == 'markdown':
                    out += ''.join(cell.get('source', [])) + '\n'
                elif cell['cell_type'] == 'code':
                    for o in cell.get('outputs', []):
                        if o.get('output_type') == 'stream':
                            out += 'OUT: ' + ''.join(o.get('text', '')) + '\n'
        except Exception as e:
            out += str(e)
with open('notebooks_summary.txt', 'w', encoding='utf-8') as f:
    f.write(out)
