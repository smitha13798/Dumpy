import json
import re
import shutil
import os
    from tree_sitter import Language, Parser
import tree_sitter_python
    from BlockFunction import FunctionType

# Load the Tree-sitter language for Python
    PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser()
parser.set_language(PY_LANGUAGE)

# File paths
src_file_to_copy = os.path.join(os.path.dirname(__file__), '../projectsrc/projectsrc.py')
destination_file_path = os.path.join(os.path.dirname(__file__), '../projectsrc/projectskeleton2.py')

# Copy the src file to generate the skeleton
shutil.copy(src_file_to_copy, destination_file_path)
print(f"File '{src_file_to_copy}' has been copied to '{destination_file_path}'.")


# Function to read source code from a file
def read_file(file_path):
with open(file_path, "r", encoding="utf-8") as file:
    return file.read()


# Function to write to a file
def write_file(file_path, content):
with open(file_path, "w", encoding="utf-8") as file:
    file.write(content)


def extract_function_name(node):
if node.type == "attribute":
attribute_node = node.child_by_field_name('attribute')
if attribute_node.type == "identifier":
return src[attribute_node.start_byte:attribute_node.end_byte].strip()
if node.type == "call":
attribute_node = node.child_by_field_name('function')
if attribute_node.type == "attribute":
identifier_node = attribute_node.child_by_field_name('attribute')
if identifier_node.type == "identifier":
return src[identifier_node.start_byte:identifier_node.end_byte].strip()

return ""


def is_simple_value(node):
"""
Checks if the given node represents a simple value like a number, string, or basic arithmetic expression.
    This ensures we're capturing basic assignments and simple expressions.
"""
# Add more conditions to capture arithmetic or other operations
"""functionNode = node.child_by_field_name('function');
if (functionNode):
attributeNode = functionNode.child_by_field_name('attribute')
if (attributeNode):
functionName = src[attributeNode.start_byte:attributeNode.end_byte].strip()
if is_function_in_enum(
    functionName) == False:  # Function which is not in enum is most likely a custom function

return True"""
isClass = node.child_by_field_name('function');
if (isClass):
attributeNode = isClass.child_by_field_name('identifier')
if (attributeNode):
functionName = src[attributeNode.start_byte:attributeNode.end_byte].strip()
if is_function_in_enum(
    functionName) == False:  # Function which is not in enum is most likely a custom function

return True

return node.type in {"integer", "float", "string", "true", "false", "none", "attribute", "list"}


def extract_class_attribute(node):
"""
Extracts class attributes (e.g., d_model: int) from an expression_statement or assignment.
"""
if node.type in {"expression_statement", "assignment"}:
left_node = node.child_by_field_name("left")
right_node = node.child_by_field_name("right")

# Handle class attributes with type annotations
if node.type == "expression_statement" and left_node:
    # Extract attribute name and type annotation
left_text = src[left_node.start_byte:left_node.end_byte].strip()
# Type annotations might be part of the expression
# For simplicity, we'll treat everything as a potential attribute
attribute_name = left_text.split(":")[0].strip()  # Get the name before ":"
attribute_type = left_text.split(":")[1].strip() if ":" in left_text else ""  # Get the type after ":"
return {
    'function': attribute_name,
    'parameters': attribute_type,
    'assigned': "",
    'index': node.start_point[0] + 1
}

elif node.type == "assignment" and left_node and right_node:
    # Extract attribute name and type from assignment
attribute_name = src[left_node.start_byte:left_node.end_byte].strip()
attribute_type = src[right_node.start_byte:right_node.end_byte].strip()  # This may need improvement
return {
    'function': attribute_name,
    'parameters': attribute_type,
    'assigned': "",
    'index': node.start_point[0] + 1
}

return None


# Use the file path defined above to read the file
src = read_file(src_file_to_copy)

# Convert the source code to bytes
src_bytes = src.encode("utf-8")

# Parse the source code
tree = parser.parse(src_bytes)


# Function to find functions, their calls, and assignments
# Function to find functions, their calls, and assignments
def find_functions_and_calls(root_node):
scope_data = []
current_function = None
loop_context = {}  # To keep track of the current loop context

def traverse(node, current_scope=None, current_class=None):

nonlocal current_function
nonlocal loop_context
token = 1
rowcheck = 0

if node.type in {"class_definition", "function_definition"}:
scope_name_node = node.child_by_field_name("name")
if scope_name_node:
scope_name = src[scope_name_node.start_byte:scope_name_node.end_byte]
else:
scope_name = src[node.start_byte:node.end_byte].strip()

row_number = node.start_point[0] + 1
start_byte = node.start_byte
end_byte = node.end_byte

has_return_statement = False
return_value = None

for child in node.children:
if child.type == "block":
return_info = find_return_statements(child)
if return_info["has_return"]:
has_return_statement = True
return_value = return_info['return_value']

if node.type == "class_definition":
current_class = scope_name
parameters_node = node.child_by_field_name("parameters")
base_classes = []

base_class_node = node.child_by_field_name("superclasses")
if base_class_node:
base_classes = src[base_class_node.start_byte:base_class_node.end_byte].strip()

if parameters_node:
parameters = src[parameters_node.start_byte:parameters_node.end_byte].strip()
else:
parameters = ""
declaration_calls = []
attributes = []
child = node.child_by_field_name("body")
if child.type == 'block':
for e in child.children:

if e.type == 'decorated_definition':
declaration = ""
for j in e.children:
if j.type == "decorator":
declaration = src[j.start_byte:j.end_byte].strip()
declaration_call = {
    'function': "Declaration",
    'parameters': declaration,
    'assigned': node.start_point[0] + 1,

}
declaration_calls.append(declaration_call)
for children in child.children:
if children.type == "expression_statement":

# member = src[children.start_byte:children.end_byte].strip()
# attributes.append(member)
for e in children.children:
if (e.type == "assignment"):
left = e.child_by_field_name("left")
right = e.child_by_field_name("type")

member = {
    'function': "member",
    'parameters': src[right.start_byte:right.end_byte].strip(),
    'assigned': src[left.start_byte:left.end_byte].strip(),
    'index': node.start_point[0] + 1,
# Initialize list for function calls within the loop
}
attributes.append(member)
# print(member)

class_data = {
    "scope": current_class,
    "attributes": attributes,
    "class_declaration": declaration_calls,  # Add attributes list above functions
"functions": [],
    "translate": True,
    "row_number": row_number,
    "parameters": parameters,
    "base_classes": base_classes,
    "start_byte": start_byte,
    "end_byte": end_byte
}

scope_data.append(class_data)

elif node.type == "function_definition":

current_scope = scope_name
parameters_node = node.child_by_field_name("parameters")

if parameters_node:
parameters = src[parameters_node.start_byte:parameters_node.end_byte].strip()
else:
parameters = ""

function_data = {
    "functionName": current_scope,
    "parameters": parameters,
    "functionCalls": [],
    "scope": current_class if current_class else "global",  # Handle global functions
"row": str(row_number),
    "returns": return_value,
    "translate": True,
    "start_byte": start_byte,
    "end_byte": end_byte
}

if current_class:
class_entry = next((item for item in scope_data if item["scope"] == current_class), None)
if class_entry:
class_entry['functions'].append(function_data)
else:
# Handle global functions or methods outside a class
global_entry = next((item for item in scope_data if item["scope"] == "global"), None)
if not global_entry:
    global_entry = {
        "scope": "global",
        "functions": [],
        "translate": False
    }
scope_data.append(global_entry)
global_entry['functions'].append(function_data)

# Set the current function
current_function = function_data

if node.type == 'for_statement' and current_function:
    iterable_node = node.child_by_field_name('right')
walker_node = node.child_by_field_name('left')
element = "element"
if walker_node:
element = src[walker_node.start_byte:walker_node.end_byte].strip()
value = src[iterable_node.start_byte:iterable_node.end_byte].strip()

loop_entry = {
    'function': "python_loop",
    'parameters': value,
    'assigned': element,
    'index': node.start_point[0] + 1,
    'functionCallsLoop': []  # Initialize list for function calls within the loop
}

# Update the loop context
loop_context = loop_entry

if current_function:
current_function['functionCalls'].append(loop_entry)

# Traverse children of the loop
for child in node.children:
traverse(child, current_scope, current_class)

# Clear loop context after processing
loop_context = {}

return  # Skip further processing for the loop node

if node.type == "assignment" and current_function:
    assigned_variable_node = node.child_by_field_name("left")
assigned_value_node = node.child_by_field_name("right")
if assigned_variable_node and assigned_value_node:
    isClass = assigned_value_node.child_by_field_name('function')

if isClass:
if isClass.type == "identifier":  # Check if assignment is a class name/class member
parametersnode = assigned_value_node.child_by_field_name('arguments')
parameters = ""
if parametersnode:
parameters = src[parametersnode.start_byte:parametersnode.end_byte].strip()
className = src[isClass.start_byte:isClass.end_byte].strip()
variable_name = src[assigned_variable_node.start_byte:assigned_variable_node.end_byte].strip()
row_number = node.start_point[0] + 1
variable_assignment_call = {
    'function': "Variable",
    'parameters': className + '(' + parameters + ')',
    'assigned': variable_name,
    'index': row_number
}
if loop_context:
loop_context['functionCallsLoop'].append(variable_assignment_call)
elif current_function:
    current_function['functionCalls'].append(variable_assignment_call)
token = 0

if assigned_variable_node and assigned_value_node:
    variable_name = src[assigned_variable_node.start_byte:assigned_variable_node.end_byte].strip()
assigned_value = src[assigned_value_node.start_byte:assigned_value_node.end_byte].strip()

if is_simple_value(assigned_value_node):
row_number = node.start_point[0] + 1
rowcheck = row_number
variable_assignment_call = {
    'function': "Variable",
    'parameters': assigned_value,
    'assigned': variable_name,
    'index': row_number
}

if loop_context:
loop_context['functionCallsLoop'].append(variable_assignment_call)
elif current_function:
    current_function['functionCalls'].append(variable_assignment_call)
token = 0
if node.type == "call":
called_function_node = node.child_by_field_name("function")  # function: call
special_input_text = ""
function_name = ""
parameters_text = ""
assigned_variable =""
parameter_node = node.child_by_field_name('arguments')
function_name = extract_function_name(called_function_node)
if parameter_node.type == "argument_list":
parameters_text = src[parameter_node.start_byte:parameter_node.end_byte].strip()
if called_function_node.type == "call":  # function: call
special_parameter_node = called_function_node.child_by_field_name('arguments')
if special_parameter_node.type == "argument_list":
special_input_text = src[special_parameter_node.start_byte:special_parameter_node.end_byte].strip()


assigned_variable = find_assignment_for_call(node)
row_number = node.start_point[0] + 1

if is_function_in_enum(function_name):
function_call = {
    'function': function_name,
    'parameters': special_input_text + parameters_text,
    'assigned': assigned_variable,
    'index': row_number
}

if function_name != "":
token = 0
if loop_context and current_scope!="":
loop_context['functionCallsLoop'].append(function_call)
elif current_function:
    current_function['functionCalls'].append(function_call)

if not is_function_in_enum(function_name):
current_function['translate'] = False
if current_class:
class_entry = next((item for item in scope_data if item["scope"] == current_class),
    None)
if class_entry:
class_entry['translate'] = False
else:
global_entry = next((item for item in scope_data if item["scope"] == "global"), None)
if global_entry:
global_entry['translate'] = False

if not is_function_in_enum(function_name):
function_call = {
    'function': "cheatblock",
    'parameters': function_name,
    'assigned': assigned_variable,
    'index': row_number,
# If replace is true, we cheeck how many functions are present as "cheat blocks"
}
if function_name != "":
token = 0
if loop_context:
loop_context['functionCallsLoop'].append(function_call)
elif current_function:
    current_function['functionCalls'].append(function_call)

for child in node.children:
traverse(child, current_scope, current_class)

traverse(root_node)
return scope_data

# This ensures that functions in the global scope are assigned to "global".

    def filter_translatable_scopes(scope_data):
filtered_scope_data = []
for scope in scope_data:
if scope['scope'] == "global":
# Set global translate to True if there are any translatable functions
if any(func['translate'] for func in scope['functions'] if 'translate' in func):
scope['translate'] = True
else:
scope['translate'] = False

# Keep only translatable functions
translatable_functions = [
    func for func in scope['functions'] if func['translate']
    ]
if translatable_functions or scope['scope'] == "global":
# If there are translatable functions, include the scope
filtered_scope_data.append({
    **scope,
    'functions': translatable_functions
})
return filtered_scope_data

traverse(root_node)
return filter_translatable_scopes(scope_data)


def find_return_statements(node):
"""Find return statements within a function body and capture return values."""
has_return = False
return_value = None

def traverse_returns(n):
nonlocal has_return, return_value
if n.type == "return_statement":
has_return = True
# The return statement usually contains an expression as its first child
if len(n.children) > 1:
return_value_node = n.children[1]  # The second child is often the return expression
return_value = src[return_value_node.start_byte:return_value_node.end_byte].strip()

for child in n.children:
traverse_returns(child)

traverse_returns(node)
return {
    "has_return": has_return,
    "return_value": return_value  # Return the value here
}


def extract_function_and_parameters(full_call_text):
"""Extract function name and parameters from the full call text."""
function_name_end = full_call_text.find('(')
if function_name_end != -1:
function_name = full_call_text[:function_name_end].strip()
parameters_text = full_call_text[function_name_end:].strip()
else:
function_name = full_call_text.strip()
parameters_text = ""

# Remove any leading module or class name prefixes
function_name = function_name.split('.')[-1]  # Get the last part after '.'

return function_name, parameters_text


def find_assignment_for_call(node):
"""Find the variable to which the function call is assigned."""
parent = node.parent
while parent:
if parent.type == "assignment":
assigned_to_node = parent.child_by_field_name("left")
if assigned_to_node:
return src[assigned_to_node.start_byte:assigned_to_node.end_byte].strip()
parent = parent.parent
return ""


# Check if a function is in the enum
def is_function_in_enum(function_name):
try:

FunctionType[function_name.upper()]
return True
except KeyError:
    return False


# Find functions and calls categorized by their parent nodes
scope_data = find_functions_and_calls(tree.root_node)

# Write the output data to a JSON file
with open('output.json', 'w', encoding='utf-8') as json_file:
    json.dump(scope_data, json_file, indent=4)

print("JSON file 'output.json' has been created with the translatable class and function data.")

# Read the copied skeleton file
skeleton_src = read_file(destination_file_path)

# Initialize an empty string to store the modified skeleton
modified_skeleton = ""
last_index = 0

# Insert #Doable comments at the head of translatable functions and classes, and remove the translatable part
for scope in scope_data:
print(scope['scope'])
print('\n')
if scope:
if scope['scope'] != 'global' and scope['translate']:

# If the scope is a class, remove the entire class definition
start_byte = scope['start_byte']  # Start byte of the class definition
end_byte = scope['end_byte']  # End byte of the class definition

# Append the source code up to the start of the class definition
modified_skeleton += skeleton_src[last_index:start_byte]

# Insert the #Doable comment at the head of the class
modified_skeleton += f"#{scope['scope']}+\n"
modified_skeleton += f"#{scope['scope']}-\n"
# Update last_index to skip the entire class definition
last_index = end_byte
else:

# If the scope is global, handle functions within the global scope
for function in scope['functions']:
if function['translate']:
start_byte = function['start_byte']  # Start byte of the function definition
end_byte = function['end_byte']  # End byte of the function definition

# Append the source code up to the start of the function definition
modified_skeleton += skeleton_src[last_index:start_byte]

# Insert the #Doable comment at the head of the function
modified_skeleton += f"#{function['functionName']}+\n"
modified_skeleton += f"#{function['functionName']}-\n"
# Update last_index to skip the translatable function
last_index = end_byte

# Add the remaining part of the file after the last processed scope
modified_skeleton += skeleton_src[last_index:]

# Write the modified skeleton back to the file
write_file(destination_file_path, modified_skeleton)

print(
    f"Translatable scopes and return statements have been marked with #Doable and removed from '{destination_file_path}'.")
