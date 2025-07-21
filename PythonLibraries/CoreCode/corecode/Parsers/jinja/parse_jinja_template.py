# Try to import jinja2, if not available, set to None
try:
    from jinja2 import Environment
    from jinja2.exceptions import TemplateSyntaxError
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Environment = None
    Template = None
    TemplateSyntaxError = None

def parse_jinja_template(template_content: str) -> dict:
    """
    Parse a Jinja template and extract information about variables, blocks, etc.
    
    If jinja2 is not available, returns the template content as-is.
    
    Args:
        template_content: The Jinja template content to parse
        
    Returns:
        dict: Information about the parsed template, or template content if
        jinja2 unavailable
    """
    # Check if jinja2 is available
    if not JINJA2_AVAILABLE:
        # Return template content as-is when jinja2 is not available
        return {
            'template_content': template_content,
            'jinja2_available': False,
            'note': 'jinja2 library not available - template returned as-is'
        }

    try:
        # Create environment and parse template
        env = Environment()
        ast = env.parse(template_content)
        
        # Extract information from the AST
        template_info = {
            'variables': set(),
            'blocks': set(),
            'macros': set(),
            'includes': set(),
            'extends': None,
            'is_valid': True,
            'errors': [],
            'jinja2_available': True
        }
        
        # Walk through the AST to find variables, blocks, etc.
        for node in ast.find_all():
            # Find variables ({{ variable }})
            if hasattr(node, 'name'):
                template_info['variables'].add(node.name)
            
            # Find blocks ({% block name %})
            if hasattr(node, 'name') and hasattr(node, 'type'):
                if node.type == 'block':
                    template_info['blocks'].add(node.name)
                elif node.type == 'macro':
                    template_info['macros'].add(node.name)
            
            # Find includes ({% include 'file.html' %})
            if hasattr(node, 'template'):
                template_info['includes'].add(str(node.template))
            
            # Find extends ({% extends 'base.html' %})
            if hasattr(node, 'template') and hasattr(node, 'type'):
                if node.type == 'extends':
                    template_info['extends'] = str(node.template)
        
        return template_info
        
    except TemplateSyntaxError as e:
        return {
            'is_valid': False,
            'errors': [str(e)],
            'variables': set(),
            'blocks': set(),
            'macros': set(),
            'includes': set(),
            'extends': None,
            'jinja2_available': True
        }
