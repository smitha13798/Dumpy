import json
import shutil
from tree_sitter import Language, Parser
import tree_sitter_python
from BlockFunction import FunctionType

# Load the Tree-sitter language for Python
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser()
parser.set_language(PY_LANGUAGE)

# File paths
src_file_to_copy = 'C:/Users/siebe/Desktop/Dumpy/Electron-Blockly_File/projectsrc/projectsrc.py'
destination_file_path = 'C:/Users/siebe/Desktop/Dumpy/Electron-Blockly_File/projectsrc/projectskeleton2.py'

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

# Use the file path defined above to read the file
src = read_file(src_file_to_copy)

# Convert the source code to bytes
src_bytes = src.encode("utf-8")

# Parse the source code
tree = parser.parse(src_bytes)
# Function to check if a node contains flax operations and extract them
def extract_flax_operations(node):
    operations = []
    prefixes = ('nn.', 'jax.', 'jnp.', 'random.');
    if node.type == 'call':
        function_name_node = node.child_by_field_name('function')
        if function_name_node:
            function_name = function_name_node.text.decode('utf-8')
            if function_name.startswith(prefixes):
                parameters_node = node.child_by_field_name('arguments')
                parameters_text = src[parameters_node.start_byte:parameters_node.end_byte].strip() if parameters_node else ""
                operations.append({
                    "function": function_name.split('.')[-1],  
                    "parameters": parameters_text
                })
    for child in node.children:
        operations.extend(extract_flax_operations(child))
    return operations

# Function to find functions and classes with flax operations
def find_functions_with_flax(root_node):
    scope_data = []

    def traverse(node, current_scope=None, current_class=None):
        if node.type in {"class_definition", "function_definition"}:
            scope_name_node = node.child_by_field_name("name")
            if scope_name_node:
                scope_name = src[scope_name_node.start_byte:scope_name_node.end_byte]
            else:
                scope_name = src[node.start_byte:node.end_byte].strip()

            # Capture the row (line number) and starting index
            row_number = node.start_point[0] + 1  # Tree-sitter uses 0-indexed lines
            start_index = node.start_byte  # Capture the starting index
            end_index = node.end_byte  # Capture the ending index

            if node.type == "class_definition":
                current_class = scope_name
                parameters_node = node.child_by_field_name("parameters")
                base_classes = []

                # Capture the base classes (like nn.Module)
                base_class_node = node.child_by_field_name("superclasses")
                if base_class_node:
                    base_classes = src[base_class_node.start_byte:base_class_node.end_byte].strip()

                if parameters_node:
                    parameters = src[parameters_node.start_byte:parameters_node.end_byte].strip()
                else:
                    parameters = ""

                # Create a new scope entry for the class
                class_data = {
                    "scope": current_class,
                    "functions": [],
                    "translate": True,
                    "start_index": start_index,
                    "end_index": end_index,
                    "row_number": row_number,
                    "parameters": parameters,
                    "base_classes": base_classes,
                }
                scope_data.append(class_data)

            elif node.type == "function_definition":
                current_scope = scope_name
                parameters_node = node.child_by_field_name("parameters")

                if parameters_node:
                    parameters = src[parameters_node.start_byte:parameters_node.end_byte].strip()
                else:
                    parameters = ""

                # Extract flax operations within the function
                flax_operations = extract_flax_operations(node)

                if flax_operations:
                    function_data = {
                        "functionName": current_scope,
                        "parameters": parameters,
                        "functionCalls": flax_operations,
                        "scope": current_class if current_class else "",
                        "row": str(row_number),
                        "start_index": start_index,
                        "end_index": end_index,
                        "translate": True
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
                        }
                        scope_data.append(global_entry)
                    global_entry['functions'].append(function_data)

        # Detect and handle function calls
        if node.type == "call":
            called_function_node = node.child_by_field_name("function")
            if called_function_node:
                # Extract the full function call including nested calls
                full_call_text = src[node.start_byte:node.end_byte].strip()

                # Extract function name and parameters
                function_name, parameters_text = extract_function_and_parameters(full_call_text)

                if current_scope:
                    if current_class:
                        class_entry = next((item for item in scope_data if item["scope"] == current_class), None)
                        if class_entry:
                            current_function = next(
                                (f for f in class_entry['functions'] if f['functionName'] == current_scope),
                                None)
                    else:
                        global_entry = next((item for item in scope_data if item["scope"] == "global"), None)
                        if global_entry:
                            current_function = next(
                                (f for f in global_entry['functions'] if f['functionName'] == current_scope), None)

                    if current_function:
                        # Normalize function calls to avoid duplicates
                        if parameters_text.endswith('(x)'):
                            # Remove the base version if it exists
                            current_function['functionCalls'] = [
                                call for call in current_function['functionCalls']
                                if call['function'] != function_name or call['parameters'] != parameters_text.rstrip('(x)')
                            ]
                            # Add the call with `(x)` if it's not already in the list
                            if not any(call['function'] == function_name and call['parameters'] == parameters_text
                                       for call in current_function['functionCalls']):
                                current_function['functionCalls'].append({
                                    'function': function_name,
                                    'parameters': parameters_text
                                })
                        else:
                            # Add the call if not already present
                            if not any(call['function'] == function_name and call['parameters'] == parameters_text
                                       for call in current_function['functionCalls']):
                                current_function['functionCalls'].append({
                                    'function': function_name,
                                    'parameters': parameters_text
                                })

                        # Check if the function is translatable
                        if not is_function_in_enum(function_name):
                            current_function['translate'] = False
                            if current_class:
                                class_entry['translate'] = False
                            else:
                                global_entry['translate'] = False

        # Traverse all children
        for child in node.children:
            traverse(child, current_scope, current_class)

    traverse(root_node)
    return scope_data

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

print("JSON file 'output.json' has been created with the class and function data.")

# Read the copied skeleton file
skeleton_src = read_file(destination_file_path)

# Initialize an empty string to store the modified skeleton
modified_skeleton = ""
last_index = 0

# Delete translatable scopes and insert #Doable comments
for scope in scope_data:
    if scope['translate']:
        # Insert #Doable comment at the start index
        modified_skeleton += skeleton_src[last_index:scope['start_index']]
        modified_skeleton += '#Doable\n'
        last_index = scope['end_index']

# Add the remaining part of the file after the last deleted scope
modified_skeleton += skeleton_src[last_index:]

# Write the modified skeleton back to the file
write_file(destination_file_path, modified_skeleton)

print(f"Translatable scopes have been deleted and #Doable comments have been added to '{destination_file_path}'.")