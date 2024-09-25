// static/app.js

document.addEventListener('DOMContentLoaded', () => {
    const cells = document.querySelectorAll('.cell');

    cells.forEach(cell => {
        cell.addEventListener('click', () => {
            const x = parseInt(cell.getAttribute('data-x'));
            const y = parseInt(cell.getAttribute('data-y'));

            fetch('/click', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ x, y }),
            })
            .then(response => response.text())
            .then(data => {
                console.log('Server response:', data);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        });
    });
});
