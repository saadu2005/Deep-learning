"""Cheap repository smoke checks suitable for CI and a new learner's laptop."""
from pathlib import Path
import ast
import runpy

ROOT = Path(__file__).resolve().parents[1]


def test_all_python_sources_parse():
    for path in ROOT.rglob('*.py'):
        if any(part.startswith('.') for part in path.parts):
            continue
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def test_intro_and_basic_math_examples_execute(capsys):
    for chapter in ('01_Introduction', '02_Neural_Network_Basics'):
        for source in sorted((ROOT / chapter).glob('[0-9]*.py')):
            runpy.run_path(str(source), run_name='__main__')
    out = capsys.readouterr().out
    assert 'Weighted Sum' in out
    assert 'Updated Weight' in out
