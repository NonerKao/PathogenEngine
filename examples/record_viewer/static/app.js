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

    // Handle click on the 6x6 grid
    function handleClick6x6(cell) {
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
    function handleScroll(direction) {
        clearGridState();

        if (direction === "up" && current_id < steps.length - 1) {
            applyStepForward(steps[current_id], steps[current_id+1]);
            current_id++;
        } else if (direction === "down" && current_id > -1) {
            applyStepBackward(steps[current_id], steps[current_id-1]);
            current_id--;
        }

        drawGrid();
    }

    const gridState = Array.from({ length: 6 }, () => Array(6).fill(null));
    let current_id = -1; // Start before the first step
    steps = [];

    // Function to clear the grid state
    function clearGridState() {
        gridState.forEach((row, rowIndex) => {
            row.forEach((_, colIndex) => {
                gridState[rowIndex][colIndex] = null;
            });
        });
    }

    // Function to draw the grid based on the current gridState
    function drawGrid() {
        gridState.forEach((row, rowIndex) => {
            row.forEach((cellState, colIndex) => {
                const cell = document.querySelector(`#grid-6x6 div[data-row="${rowIndex}"][data-col="${colIndex}"]`);
                if (cellState) {
                    const char1Class = token.char1 === "D" ? "token-blue" : 
                                       token.char1 === "P" ? "token-red" : "";
                    const char2Class = token.color === "blue" ? "token-blue" : 
                                       token.color === "red" ? "token-red" : "";

                    cell.innerHTML = `<span class="${char1Class}">${cellState.char1}</span><span class="${char2Class}">${cellState.char2}</span>`;
                } else {
                    cell.innerHTML = ''; // Clear the cell if there's no state
                }
            });
        });
    }

    function applyStepForward(from, to) {
        const (from_on_map, [fcol, frow]) = convertPosition(from.pos);
        const (to_on_map, [tcol, trow]) = convertPosition(to.pos);
        if (from.char1 != to.char1) {
            // switch player
            // XXX: Remove previous map position, ...
            // XXX: Display `to`
        } else if (from.is_marker != to.is_marker) {
            // from BoardMove to SetMarker, nothing to be done
        } else {
            if (from_on_map != to_on_map) {
                // from SetMap/Lockdown to SetCharacter, nothing to be done
            } else if (from_on_map) {
                // Lockdown
                // XXX: Remove previous plague's map position
                // XXX: Display `to`
            } else {
                if (from.is_marker) {
                    // SetMarker
                    // XXX: add `to.marker`
                } else {
                    // BoardMove
                    // XXX: Remove `from.char1`
                    // XXX: Display `to`
                }
            }
        }
        //gridState[row][col] = { char1: step.char1, char2: step.char2, color: step.color };
    }

    function applyStepBackward(step) {
        const (on_map, [col, row]) = convertPosition(step.pos);
        gridState[row][col] = { char1: step.char1, char2: step.char2, color: step.color };
    }

    // Apply throttle to the scroll event
    document.addEventListener("wheel", throttle((event) => {
        if (event.deltaY < 0) {
            handleScroll("up");
        } else {
            handleScroll("down");
        }
    }, 150));

    fetch('/game_state')
        .then(response => response.json())
        .then(data => {
            setupGrid(data.white_positions, data.black_positions);
            steps = data.steps;
        })

    function setupGrid(whitePositions, blackPositions) {
        // Setup0
        whitePositions.forEach(pos => {
            const (_, [col, row]) = convertPosition(pos);
            const cell = document.querySelector(`#grid-6x6 div[data-row="${row}"][data-col="${col}"]`);
            if (cell) cell.style.backgroundColor = 'white';
        });

        blackPositions.forEach(pos => {
            const (_, [col, row]) = convertPosition(pos);
            const cell = document.querySelector(`#grid-6x6 div[data-row="${row}"][data-col="${col}"]`);
            if (cell) cell.style.backgroundColor = 'black';
        });
    }

    // Return (on_map, [col, row])
    function convertPosition(pos) {
        // Convert SGF-style coordinates to grid indices
        let col = pos.charCodeAt(0) - 'a'.charCodeAt(0);
        let row = pos.charCodeAt(1) - 'a'.charCodeAt(0);
        if col > 5 {
	    return (true, [col - 8, row - 8]); 
        } else {
	    return (false, [col, row]); 
        }
    }

    // Function to apply tokens to the grid
    function applyTokens(tokens) {
        tokens.forEach(token => {
            const [row, col] = token.pos.split("").map(Number);
            const cell = document.querySelector(`#grid-6x6 div[data-row="${row}"][data-col="${col}"]`);
            if (cell) {
                bg_white = cell.style.backgroundColor === 'white';
	        console.log(` (${cell.dataset.row}, ${cell.dataset.col}): (${cell.style.backgroundColor})`);
                const char1 = token.char1 === "D" ? "token-blue" : 
                              token.char1 === "P" ? "token-red" : "";
                const char2 = token.color === "blue" ? "token-blue" : 
                              token.color === "red" ? "token-red" : "";

                cell.innerHTML = `<span class="${char1}">${token.char1}</span><span class="${char2}">${token.char2}</span>`;
            }
        });
    }

    // Apply tokens (use your game state instead of the hardcoded tokens)
    applyTokens(tokens);
});


