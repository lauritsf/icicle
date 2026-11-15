"""The tuning and ONNX API must not load experiment or accelerator dependencies."""

import subprocess
import sys


def test_runtime_imports_without_research_dependencies():
    code = '''
import importlib.abc
import sys

class BlockResearchImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'reproduction', 'pandas', 'pyarrow', 'jax', 'torch', 'ray'}:
            raise ImportError(f'Unexpected runtime dependency: {fullname}')

sys.meta_path.insert(0, BlockResearchImports())
import icicle.hpo
import icicle.serving
'''
    subprocess.run([sys.executable, "-c", code], check=True)
