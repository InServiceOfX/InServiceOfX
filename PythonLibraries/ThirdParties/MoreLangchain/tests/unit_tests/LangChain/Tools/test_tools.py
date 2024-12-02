# See langchain/tools/__init__.py which then points to
# from langchain_core.tools.convert import tool as tool i.e.
# langchain_core/tools/convert.py and then
# @overload
# def tool(..) -> BaseTool

from langchain_core.tools import tool

import pytest

"""
https://python.langchain.com/docs/concepts/tools/#create-tools-using-the-tool-decorator
Recommended way to create tools is to use the @tool decorator. 
"""

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def test_multiply_has_attributes():
    assert multiply.name == "multiply"
    assert multiply.description == "Multiply two numbers."
    assert isinstance(multiply.args, dict)
    assert 'a' in multiply.args.keys()
    assert 'b' in multiply.args.keys()
    assert multiply.args['a'] == {'title': 'A', 'type': 'integer'}
    assert multiply.args['b'] == {'title': 'B', 'type': 'integer'}

def test_multiply_type_checking():
    # Test that it enforces integer types
    with pytest.raises(AttributeError):
        multiply(2.5, 3)
    with pytest.raises(AttributeError):
        multiply("2", 3)

def test_multiply_invokes():
    assert multiply.invoke({"a": 2, "b": 3}) == 6

@tool
async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def test_amultiply_has_attributes():
    assert amultiply.name == "amultiply"
    assert amultiply.description == "Multiply two numbers."
    assert isinstance(amultiply.args, dict)
    assert 'a' in amultiply.args.keys()
    assert 'b' in amultiply.args.keys()
    assert amultiply.args['a'] == {'title': 'A', 'type': 'integer'}
    assert amultiply.args['b'] == {'title': 'B', 'type': 'integer'}

def test_amultiply_invokes():
    with pytest.raises(NotImplementedError):
        amultiply.invoke({"a": 2, "b": 3})

