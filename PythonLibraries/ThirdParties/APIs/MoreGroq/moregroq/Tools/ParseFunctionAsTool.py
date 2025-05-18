from inspect import signature, Parameter, cleandoc
import re
from moregroq.Wrappers.ChatCompletionConfiguration import (
    FunctionDefinition,
    FunctionParameters,
    ParameterProperty)

from typing import Callable
import typing

class ParseFunctionAsTool:

    @staticmethod
    def _convert_Python_type_for_Groq_API_JSON_schema(type_name: str):
        """This function was necessitated by this error:

        groq.BadRequestError: Error code: 400 - {'error': {'message': 'schema is not valid JSON Schema for tool calculate parameters: jsonschema file:///home/di/params.json compilation failed: \'/properties/expression/type\' does not validate with https://json-schema.org/draft/2020-12/schema#/allOf/1/$ref/properties/properties/additionalProperties/$dynamicRef/allOf/3/$ref/properties/type/anyOf/0/$ref/enum:
        value must be one of "array", "boolean", "integer", "null", "number", "object", "string"', 'type': 'invalid_request_error'}}
        """
        if type_name == "int":
            return "integer"
        elif type_name == "float":
            return "number"
        elif type_name == "str":
            return "string"
        elif type_name == "bool":
            return "boolean"
        elif type_name == "list" or type_name == "tuple" \
            or type_name == "List" or type_name.startswith("List["):
            return "array"
        else:
            return type_name
 
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
    def _get_detailed_type_info(input_function: Callable):
        """Extract detailed type information from function annotations."""
        sig = signature(input_function)
        type_info = {}
        type_annotation = {}
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue

            if param.annotation is not param.empty:
                # Handle typing module's complex types
                if hasattr(param.annotation, '__origin__'):
                    origin = param.annotation.__origin__
                    args = getattr(param.annotation, '__args__', [])
                    
                    if origin is typing.Union:
                        type_info[name] = \
                            f"Union[{', '.join(arg.__name__ for arg in args)}]"
                    elif origin is list:
                        type_info[name] = f"List[{args[0].__name__}]"
                    elif origin is dict:
                        type_info[name] = \
                            f"Dict[{args[0].__name__}, {args[1].__name__}]"
                    else:
                        type_info[name] = str(param.annotation)
                    
                    type_annotation[name] = origin
                else:
                    type_info[name] = param.annotation.__name__
                    type_annotation[name] = param.annotation
            # Then param.annotation is param.empty
            else:
                type_info[name] = "Any"
                type_annotation[name] = param.annotation

        return type_info, type_annotation
        
    @staticmethod
    def _parse_docstring_sections(input_function: Callable):
        """Parse docstring into structured sections (Args, Returns, etc.)."""
        if not input_function.__doc__:
            return {}
        
        docstring = cleandoc(input_function.__doc__)
        sections = {}
        current_section = 'description'
        sections[current_section] = []
        
        for line in docstring.split('\n'):
            section_match = re.match(r'^(\w+):', line)
            if section_match and section_match.group(1) in [
                'Args',
                'Returns',
                'Raises',
                'Examples',
                'Usage',
                'Notes',
                'See Also',
                'References']:
                current_section = section_match.group(1)
                sections[current_section] = []
            else:
                sections[current_section].append(line)
        
        # Join the lines in each section
        for section in sections:
            sections[section] = '\n'.join(sections[section]).strip()
        
        return sections

    @staticmethod
    def _extract_parameter_descriptions(input_function: Callable):
        """
        Extract parameter descriptions from the Args section of a docstring.
        
        Args:
            input_function: The function to extract parameter descriptions from
            
        Returns:
            dict: A dictionary mapping parameter names to their descriptions
        """
        arguments = signature(input_function).parameters.keys()

        if 'self' in arguments:
            arguments.remove('self')

        # Parse docstring sections
        sections = ParseFunctionAsTool._parse_docstring_sections(input_function)
        
        # If no Args section, return empty descriptions
        if 'Args' not in sections:
            return {arg: "" for arg in arguments}
        
        param_descriptions = {}
        args_section = sections['Args']
        
        # Split the Args section by lines and process
        lines = args_section.split('\n')
        current_param = None
        current_description = []
        
        for line in lines:
            # Check if this line defines a new parameter
            param_match = re.match(
                r'^\s*([a-zA-Z0-9_]+)\s*(\(.*\))?\s*:\s*(.*)',
                line)
            
            if param_match:
                # If we were processing a parameter, save it
                if current_param and current_param in arguments:
                    param_descriptions[current_param] = \
                        '\n'.join(current_description).strip()

                # Start processing new parameter
                current_param = param_match.group(1)
                current_description = [param_match.group(3)]
            elif line.strip() and current_param:
                # Continue with current parameter description
                current_description.append(line.strip())
        
        # Save the last parameter being processed
        if current_param and current_param in arguments:
            param_descriptions[current_param] = \
                '\n'.join(current_description).strip()
        
        # Ensure all arguments have descriptions (even if empty)
        for arg in arguments:
            if arg not in param_descriptions:
                param_descriptions[arg] = ""
        
        return param_descriptions

    @staticmethod
    def parse_for_function_definition(input_function: Callable):
        _, _, param_info, name = \
            ParseFunctionAsTool.parse_for_docstring_arguments_name(
                input_function)

        type_info, type_annotation = \
            ParseFunctionAsTool._get_detailed_type_info(input_function)

        sections = ParseFunctionAsTool._parse_docstring_sections(input_function)

        param_descriptions = \
            ParseFunctionAsTool._extract_parameter_descriptions(input_function)

        function_properties = [
            ParameterProperty(
                name=argument_name,
                type=ParseFunctionAsTool._convert_Python_type_for_Groq_API_JSON_schema(
                    type_info[argument_name]),
                actual_type=type_annotation[argument_name],
                description=param_descriptions[argument_name],
                required=True
            )
            for argument_name in param_info.keys()
        ]

        function_definition = FunctionDefinition(
            name=name,
            description=sections['description'],
            parameters=FunctionParameters(
                properties=function_properties
            )
        )

        return function_definition
