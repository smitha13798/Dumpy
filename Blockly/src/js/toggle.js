const showBlockly = document.getElementById('showBlockly');
const showPreview = document.getElementById('showPreview');
const blocklyContainer = document.getElementById('blocklyContainer');
const previewContainer = document.getElementById('previewContainer');

showBlockly.addEventListener('click', () => {
    blocklyContainer.style.display = 'block';
    previewContainer.style.display = 'none';
});

showPreview.addEventListener('click', () => {
    blocklyContainer.style.display = 'none';
    previewContainer.style.display = 'block';
});
