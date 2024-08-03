from tree_sitter import Language, Parser

# Load the compiled language grammar
Language.build_library(
    'build/my-languages.so',
    [
        './tree-sitter-javascript'
    ]
)

# Load the language
JAVASCRIPT_LANGUAGE = Language('build/my-languages.so', 'javascript')

# Create the parser
parser = Parser()
parser.set_language(JAVASCRIPT_LANGUAGE)

# Parse some JavaScript code
source_code = b"""
function add(a, b) {
  return a + b;
}

function subtract(a, b) {
  return a - b;
}

function multiply(a, b) {
  return a * b;
}
"""

tree = parser.parse(source_code)

# Function to traverse the AST and find function declarations
def get_functions_from_ast(node, functions=[]):
    if node.type == 'function_declaration':
        function_name = node.child_by_field_name('name').text.decode('utf-8')
        functions.append(function_name)
    for child in node.children:
        get_functions_from_ast(child, functions)
    return functions

# Extract functions
root_node = tree.root_node
functions = get_functions_from_ast(root_node)

print("Functions found:", functions)
