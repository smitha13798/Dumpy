import json
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

# Use the file path defined above to read the file
src = read_file(src_file_to_copy)

# Convert the source code to bytes
src_bytes = src.encode("utf-8")

# Parse the source code
tree = parser.parse(src_bytes)

# Function to find functions, their calls, and assignments
def find_functions_and_calls(root_node):
    scope_data = []

    def traverse(node, current_scope=None, current_class=None):
        if node.type in {"class_definition", "function_definition"}:
            scope_name_node = node.child_by_field_name("name")
            if scope_name_node:
                scope_name = src[scope_name_node.start_byte:scope_name_node.end_byte]
            else:
                scope_name = src[node.start_byte:node.end_byte].strip()

            # Capture the row (line number)
            row_number = node.start_point[0] + 1  # Tree-sitter uses 0-indexed lines
            start_byte = node.start_byte  # Capture the start_byte for the function/class
            end_byte = node.end_byte  # Capture the end_byte for the function/class

            has_return_statement = False
            return_value = None  # Initialize return_value

            # Traverse the function body to find return statements and assignments
            for child in node.children:
                if child.type == "block":
                    return_info = find_return_statements(child)
                    if return_info["has_return"]:
                        has_return_statement = True
                        return_value = return_info['return_value']  # Capture return_value

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
                    "scope": current_class if current_class else "",
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

        # Detect and handle function calls
        if node.type == "call":
            called_function_node = node.child_by_field_name("function")
            if called_function_node:
                # Extract the full function call including nested calls
                full_call_text = src[node.start_byte:node.end_byte].strip()

                # Extract function name and parameters
                function_name, parameters_text = extract_function_and_parameters(full_call_text)

                # Check if the function call is assigned to a variable
                assigned_variable = find_assignment_for_call(node)

                # Capture the row number (index) where the function call occurs
                row_number = node.start_point[0] + 1  # Tree-sitter uses 0-indexed lines

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
                        # Always add the function call, even if it is a duplicate
                        current_function['functionCalls'].append({
                            'function': function_name,
                            'parameters': parameters_text,
                            'assigned': assigned_variable,
                            'index': row_number  # Include the index (row number)
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

    # Filter out non-translatable scopes and functions
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
    if 'translate' in scope and scope['translate']:
        if scope['scope'] != 'global':
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

print(f"Translatable scopes and return statements have been marked with #Doable and removed from '{destination_file_path}'.")
