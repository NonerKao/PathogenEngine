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

    // Handle key events
    function handleKey(direction) {
        if (direction === "PageUp" && current_id < steps.length - 1) {
            if (current_id == -1) {
                const [_, [tcol, trow]] = convertPosition(steps[0].pos);
                drawMap(trow, tcol, 'P');
            } else {
                applyStepForward();
            }
            current_id++;
        } else if (direction === "PageDown" && current_id > -1) {
            if (current_id == 0) {
                const [_, [fcol, frow]] = convertPosition(steps[0].pos);
                const cell = document.querySelector(`#grid-5x5 div[data-row="${frow}"][data-col="${fcol}"]`);
                cell.innerHTML = `<span class=""></span>`;
                mapState['P'] = [-999, -999];
            } else {
                applyStepBackward();
            }
            current_id--;
        }

    }

    let current_id = -1;
    steps = [];

    const rows = 6;
    const cols = 6;
    
    // Initialize the gridState as a 2D array
    let gridState = Array.from({ length: rows }, () => 
        Array.from({ length: cols }, () => ({ char1: ' ', marker: 0 }))
    );
    let mapState = {};
    mapState['D'] = [-999, -999];
    mapState['P'] = [-999, -999];
    
    // Function to reset or reinitialize the gridState
    function clearGridStateAll() {
        for (let rowIndex = 0; rowIndex < rows; rowIndex++) {
            for (let colIndex = 0; colIndex < cols; colIndex++) {
                gridState[rowIndex][colIndex] = { char1: ' ', marker: 0 };
            }
        }
    }
    clearGridStateAll();
    function clearGridState(row, col, id) {
        gridState[row][col] = { char1: ' ', marker: 0 };
    }
    
    // Function to draw the grid based on the current gridState
    function drawGrid(row, col) {
        const cell = document.querySelector(`#grid-6x6 div[data-row="${row}"][data-col="${col}"]`);
        if (gridState[row][col]) {
            const char1Class = gridState[row][col].char1 === "D" ? "token-blue" : 
                               gridState[row][col].char1 === "P" ? "token-red" : "";
            const char2Class = gridState[row][col].marker > 0 ? "token-blue" : 
                               gridState[row][col].marker < 0 ? "token-red" : "";
                    
            char2 = (gridState[row][col].marker >= 6 || gridState[row][col].marker <= -6) ? '*' : (gridState[row][col].marker != 0) ? String(Math.abs(gridState[row][col].marker)) : '';

            cell.innerHTML = `<span class="${char1Class}">${gridState[row][col].char1}</span><span class="${char2Class}">${char2}</span>`;
        } else {
            cell.innerHTML = ''; // Clear the cell if there's no state
        }
    }

    function drawMap(row, col, char1) {
        const [prev_col, prev_row] = mapState[char1];
        
        const prev_cell = document.querySelector(`#grid-5x5 div[data-row="${prev_row}"][data-col="${prev_col}"]`);
        if (prev_cell) {
            prev_cell.innerHTML = `<span class=""></span>`;
        }

        const cell = document.querySelector(`#grid-5x5 div[data-row="${row}"][data-col="${col}"]`);
        const char1Class = char1 === "D" ? "token-blue" : 
                           char1 === "P" ? "token-red" : "";
        cell.innerHTML = `<span class="${char1Class}">${char1}</span>`;
        mapState[char1] = [col, row];
    }

    function applyStepForward() {
        let from = steps[current_id];
        let to = steps[current_id+1];
        const [from_on_map, [fcol, frow]] = convertPosition(from.pos);
        const [to_on_map, [tcol, trow]] = convertPosition(to.pos);
            
        if (from.char1 != to.char1) {
            // switch player
            drawMap(trow, tcol, to.char1);
        } else if (to.is_marker) {
            // SetMarker
            let sum = gridState[trow][tcol].marker + to.marker;
            if (gridState[trow][tcol].marker >= 6 || gridState[trow][tcol].marker <= -6) {
                // do nothing
            } else {
                gridState[trow][tcol].marker = sum;
            }
            drawGrid(trow, tcol);
        } else {
            if (from_on_map != to_on_map) {
                // from SetMap/Lockdown to SetCharacter, nothing to be done
            } else if (from_on_map) {
                drawMap(trow, tcol, 'P');
            } else {
                if (from.is_marker) {
                    // not possible here
                } else {
                    // BoardMove
                    gridState[frow][fcol].char1 = ' ';
                    drawGrid(frow, fcol);
                    gridState[trow][tcol].char1 = to.char1;
                    drawGrid(trow, tcol);
                }
            }
        }
    }

    function applyStepBackward(step) {
    }

    document.addEventListener("keydown", (event) => {
        handleKey(event.key);
    });

    fetch('/game_state')
        .then(response => response.json())
        .then(data => {
            setupGrid(data.white_positions, data.black_positions);
            data.steps.forEach(s => {
                if (s.id > 8 /* setup1(0~3) + setup2(4~7) + setup3(8) */) {
                    steps.push(s);
                } else {
                    addSetup(s);
                }
            });
            for (let rowIndex = 0; rowIndex < rows; rowIndex++) {
                for (let colIndex = 0; colIndex < cols; colIndex++) {
                    drawGrid(rowIndex, colIndex);
                }
            }
        });

    function addSetup(step) {
        const [on_map, [col, row]] = convertPosition(step.pos);
        if (on_map) {
            mapState[step.char1] = [col, row];
            drawMap(row, col, 'D');
        } else if (step.is_marker) {
            gridState[row][col] = { char1: ' ', marker: -1 };
        } else {
            // taking turn to put the characters
            gridState[row][col] = { char1: step.char1, marker: 0 };
        }
    }

    function setupGrid(whitePositions, blackPositions) {
        // Setup0
        whitePositions.forEach(pos => {
            const [_, [col, row]] = convertPosition(pos);
            const cell = document.querySelector(`#grid-6x6 div[data-row="${row}"][data-col="${col}"]`);
            if (cell) cell.style.backgroundColor = 'white';
        });

        blackPositions.forEach(pos => {
            const [_, [col, row]] = convertPosition(pos);
            const cell = document.querySelector(`#grid-6x6 div[data-row="${row}"][data-col="${col}"]`);
            if (cell) cell.style.backgroundColor = 'black';
        });
    }

    console.log(`(${steps[current_id]} = ${steps}[${current_id}])`);

    function exposeSteps(sgfSteps) {
        steps = sgfSteps;
    }

    // Return [on_map, [col, row]]
    function convertPosition(pos) {
        // Convert SGF-style coordinates to grid indices
        let col = pos.charCodeAt(1) - 'a'.charCodeAt(0);
        let row = pos.charCodeAt(0) - 'a'.charCodeAt(0);
        if (col > 5) {
	    return [true, [col - 8, row - 8]]; 
        } else {
	    return [false, [col, row]]; 
        }
    }

});


