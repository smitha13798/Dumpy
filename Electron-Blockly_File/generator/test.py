import json
from tree_sitter import Language, Parser
import tree_sitter_python
from BlockFunction import FunctionType

# Load the language grammar


PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser()
parser.set_language(PY_LANGUAGE)
# Define the query to get the parameter list of functions


# Define the query to get the parameter list of function definitions and function calls

# Define the query string
query_string = """
(function_definition
  name: (identifier) @function_name
  parameters: (parameters) @param_list
  body: (block
          (function_definition) @nested_function
          (call) @call_function
        )
)
"""

# Create the query object using the language
query = PY_LANGUAGE.query(query_string)


# Function to parse the Python file and extract parameter lists and nested functions
def extract_function_scope_details(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Parse the code
    tree = parser.parse(bytes(code, "utf8"))

    # Apply the query to the parse tree
    captures = query.captures(tree.root_node)

    # Organize results
    function_scopes = []
    current_scope = {}

    for capture in captures:
        node, capture_name = capture
        extracted_code = code[node.start_byte:node.end_byte]

        if capture_name == "function_name":
            if current_scope:  # Save the previous scope if any
                function_scopes.append(current_scope)
            current_scope = {"name": extracted_code, "parameters": "", "calls": [], "nested_functions": []}
 
        elif capture_name == "param_list":
            current_scope["parameters"] = extracted_code

        elif capture_name == "call_function":
            current_scope["calls"].append(extracted_code)

        elif capture_name == "nested_function":
            current_scope["nested_functions"].append(extracted_code)

    if current_scope:  # Add the last function scope
        function_scopes.append(current_scope)

    return function_scopes


# Example usage
file_path = "../projectsrc/projectsrc .py"  # Replace with the path to your Python file
function_scopes = extract_function_scope_details(file_path)
for scope in function_scopes:
    print(f"Function: {scope['name']}")
    print(f"  Parameters: {scope['parameters']}")
    print(f"  Nested Functions: {scope['nested_functions']}")
    print(f"  Calls: {scope['calls']}")
    print()