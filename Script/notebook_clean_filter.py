"""Git clean filter of the tutorial notebooks (.gitattributes: docs/tutorials/*.ipynb filter=notebook): what is
committed has no outputs and no paths of this computer; the notebook on disk keeps both.
  - code cells: outputs and execution counts removed;
  - a FILL IN constant set to an absolute path, NAME = Path('/...'), becomes Path('/path/to/<file name>').
Set up once in every clone (git does not take filters from the repository):
    git config filter.notebook.clean "python3 Script/notebook_clean_filter.py"
    git config filter.notebook.required true
"""
import json
import re
import sys
from pathlib import PurePosixPath

PATH = re.compile(r"""^(\s*[A-Z][A-Z0-9_]*\s*=\s*Path\(\s*)(['"])(/[^'"]*)\2""")


def placeholder(m):
    return f"{m.group(1)}{m.group(2)}/path/to/{PurePosixPath(m.group(3)).name}{m.group(2)}"


nb = json.loads(sys.stdin.buffer.read().decode('utf-8'))
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'], cell['execution_count'] = [], None
    src = cell['source'].splitlines(keepends=True) if isinstance(cell['source'], str) else cell['source']
    cell['source'] = [PATH.sub(placeholder, line) for line in src]
sys.stdout.buffer.write((json.dumps(nb, indent=1, ensure_ascii=False) + '\n').encode('utf-8'))
