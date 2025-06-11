from pathlib import Path
import pytest
import sys

# To obtain modules from CoreCode
corecode_path = Path(__file__).resolve().parents[4] / "CoreCode"

if corecode_path.exists():
	sys.path.append(str(corecode_path))
else:
	print(f"CoreCode path does not exist: {corecode_path}")

morepydanticai_path = Path(__file__).resolve().parents[1]

if morepydanticai_path.exists():
	sys.path.append(str(morepydanticai_path))
else:
	print(f"MorePydanticAI path does not exist: {morepydanticai_path}")

# This is to be able to import TestSetup
tests_path = Path(__file__).resolve().parents[0]

if tests_path.exists():
	sys.path.append(str(tests_path))
else:
	print(f"Tests path does not exist: {tests_path}")