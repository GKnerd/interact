"""
Checks the two rules of ``interact/constants.py``: it imports nothing from ``interact``, and it
holds only plain, immutable values.
"""
import ast
import importlib.util
from pathlib import Path
from types import MappingProxyType


def test_imports_nothing_from_interact() -> None:
    tree = ast.parse(_constants_path().read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f'line {node.lineno}: relative import'
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        for name in names:
            assert name.split('.')[0] != 'interact', f'line {node.lineno}: imports {name}'


def test_values_are_plain_and_immutable() -> None:
    # Loaded from its file, not through the package, so the check needs neither torch nor interact
    spec = importlib.util.spec_from_file_location('constants', _constants_path())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [name for name in vars(module) if name.isupper()]
    assert names, 'no constants found'
    for name in names:
        assert _is_plain(getattr(module, name)), f'{name} is not a plain, immutable value'


def _constants_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'interact' / 'constants.py'


def _is_plain(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, slice):
        return all(_is_plain(part) for part in (value.start, value.stop, value.step))
    if isinstance(value, (tuple, frozenset)):
        return all(_is_plain(item) for item in value)
    if isinstance(value, MappingProxyType):
        return all(_is_plain(key) and _is_plain(item) for key, item in value.items())
    return False
