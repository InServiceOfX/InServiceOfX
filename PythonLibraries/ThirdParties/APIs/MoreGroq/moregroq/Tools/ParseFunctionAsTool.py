from inspect import signature, Parameter
from moregroq.Wrappers.ChatCompletionConfiguration import (
    FunctionDefinition,
    FunctionParameters,
    ParameterProperty,
    Tool)

from typing import Callable

class ParseFunctionAsTool:

    @staticmethod
    def parse_for_docstring_arguments_name(input_function: Callable):
        docstring = input_function.__doc__
        
        arguments = input_function.__code__.co_varnames[
            :input_function.__code__.co_argcount]

        sig = signature(input_function)

        param_info = {}
    
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            param_info[name] = {
                'default': None \
                    if param.default is Parameter.empty else param.default,
                'annotation': 'Any' \
                    if param.annotation is Parameter.empty \
                        else param.annotation.__name__,
                'kind': str(param.kind)
            }

        return docstring, arguments, param_info, input_function.__name__

    @staticmethod
    def parse_for_function_definition(input_function: Callable):
        docstring, arguments, param_info, name = \
            ParseFunctionAsTool.parse_for_docstring_arguments_name(
                input_function)
        
        function_properties = [
            ParameterProperty(
                name=argument_name,
                type="string" \
                    if param_info[argument_name]['annotation'] == 'Any' \
                        else param_info[argument_name]['annotation'],
                # TODO: Ask user for the description.
                description="",
                required=True
            )
            for argument_name in arguments
        ]

        function_definition = FunctionDefinition(
            name=name,
            description=docstring,
            parameters=FunctionParameters(
                properties=function_properties
            )
        )

        return function_definition
        
