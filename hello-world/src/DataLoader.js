document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileContent = document.getElementById('fileContent');

    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (event) => {
        event.preventDefault();
        dropZone.classList.remove('dragover');

        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            const reader = new FileReader();

            reader.onload = (e) => {
                fileContent.textContent = e.target.result;
                console.log(fileContent.textContent);
                document.getElementById('textarea').textContent += fileContent.textContent;

            };

            reader.readAsText(file);
        }
    });
});
