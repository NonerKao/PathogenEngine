document.addEventListener("DOMContentLoaded", () => {
    const grid6x6 = document.getElementById("grid-6x6");
    const grid5x5 = document.getElementById("grid-5x5");

    // Create the 6x6 grid
    for (let col = 0; col < 6; col++) {
        for (let row = 0; row < 6; row++) {
            const cell = document.createElement("div");
            cell.dataset.row = row;
            cell.dataset.col = col;
            cell.addEventListener("click", () => handleClick6x6(cell));
            grid6x6.appendChild(cell);
        }
    }

    // Create the 5x5 grid
    for (let col = -2; col < 3; col++) {
        for (let row = -2; row < 3; row++) {
            const cell = document.createElement("div");
            cell.dataset.row = row;
            cell.dataset.col = col;
            cell.addEventListener("click", () => handleClick5x5(row, col));
            grid5x5.appendChild(cell);
        }
    }

    // Function to generate a random color
    function getRandomColor() {
        const letters = '0123456789ABCDEF';
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

    // Handle click on the 6x6 grid
    function handleClick6x6(cell) {
        cell.style.backgroundColor = getRandomColor();
        console.log(`6x6 Grid clicked at (${cell.dataset.row}, ${cell.dataset.col})`);
    }

    // Handle click on the 5x5 grid
    function handleClick5x5(row, col) {
        console.log(`5x5 Grid clicked at (${row}, ${col})`);
    }

    // Throttle function to limit the rate at which a function can fire
    function throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }

    // Handle scroll events
    function handleScroll(event) {
        if (event.deltaY < 0) {
            console.log("Scrolled up");
        } else {
            console.log("Scrolled down");
        }
    }

    // Apply throttle to the scroll event
    document.addEventListener("wheel", throttle(handleScroll, 150));

    fetch('/game_state')
        .then(response => response.json())
        .then(data => {
            setupGrid(data.white_positions, data.black_positions);
        });

    function setupGrid(whitePositions, blackPositions) {
        whitePositions.forEach(pos => {
            const [col, row] = convertPosition(pos);
            const cell = document.querySelector(`#grid-6x6 div[data-row="${row}"][data-col="${col}"]`);
            if (cell) cell.style.backgroundColor = 'white';
        });

        blackPositions.forEach(pos => {
            const [col, row] = convertPosition(pos);
            const cell = document.querySelector(`#grid-6x6 div[data-row="${row}"][data-col="${col}"]`);
            if (cell) cell.style.backgroundColor = 'black';
        });
    }

    function convertPosition(pos) {
        // Convert SGF-style coordinates to grid indices
        const col = pos.charCodeAt(0) - 'a'.charCodeAt(0);
        const row = pos.charCodeAt(1) - 'a'.charCodeAt(0);
        return [col, row];
    }
});


