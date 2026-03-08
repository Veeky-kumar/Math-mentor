import sympy
from typing import Any
from langchain.tools import tool

@tool
def evaluate_expression(expression: str) -> str:
    '''Evaluate a mathematical expression safely using SymPy.'''
    try:
        # Define some common variables
        x, y, z = sympy.symbols('x y z')
        # Parse the string and evaluate
        expr = sympy.sympify(expression)
        result = sympy.simplify(expr)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def get_sympy_tools():
    return [evaluate_expression]
