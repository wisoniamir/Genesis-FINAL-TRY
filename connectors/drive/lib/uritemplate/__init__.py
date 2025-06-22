
# Minimal uritemplate implementation for GENESIS Sync Beacon
import re

def expand(template, variables, **kwargs):
    # This is a very simplified version that just handles basic templates
    for var_name, var_value in variables.items():
        pattern = r'{' + re.escape(var_name) + r'}'
        template = re.sub(pattern, str(var_value), template)
    return template
