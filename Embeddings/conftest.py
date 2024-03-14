"""
@file conftest.py

@details Guard access to resources with fixtures.

From
https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
this makes it so that pytest looks for the conftest and pytest adds the parent
directory of conftest.py to the sys.path (in this case, /T1000).

No need to write custom code for mangling sys.path or remember to drag
PYTHONPATH along, or placing __init__.py into dirs where it doesn't belong.
"""

import pytest

# Import modules from else where in this repository:

from pathlib import Path
import sys

number_of_parents_to_project_path = 1

current_filepath = Path(__file__).resolve() # Resolve to the absolute path.
project_path = \
	current_filepath.parents[number_of_parents_to_project_path].resolve()
sys.path.append(project_path / "CoreCode/")
