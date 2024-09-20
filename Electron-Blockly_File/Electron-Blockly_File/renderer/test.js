function removeFirstAndLastParentheses(str) {
    // Remove the first opening parenthesis
    let result = str.replace(/^\(/, '');
    // Remove the last closing parenthesis
    result = result.replace(/\)$/, '');
    return result;
}

var str = "(3)"
console.log(removeFirstAndLastParentheses(str))