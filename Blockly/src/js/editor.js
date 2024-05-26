import CodeMirror from 'codemirror';
import 'codemirror/mode/python/python';

const editor = CodeMirror.fromTextArea(document.getElementById('codeEditor'), {
    mode: 'python',
    lineNumbers: true,
    autoCloseBrackets: true,
    matchBrackets: true,
    styleActiveLine: true,
    theme: 'monokai'
});

export default editor;
