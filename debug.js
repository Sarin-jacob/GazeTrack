// debug.js

(function() {
    'use strict';

    /**
     * This script waits for the main page content to load, then injects a
     * new button and a grid overlay for debugging gaze coordinates.
     */
    const bodyy=document.body
    document.addEventListener('DOMContentLoaded', () => {
        const controlsContainer = document.getElementById('controls');
        const stageContainer = document.getElementById('stage');

        // Exit if required elements aren't found
        if (!controlsContainer || !stageContainer) {
            console.error("Debug Grid: Could not find '#controls' or '#stage' elements.");
            return;
        }

        // --- Create Grid Overlay ---
        const gridOverlay = document.createElement('div');
        gridOverlay.id = 'debug-grid-overlay';
        Object.assign(gridOverlay.style, {
            position: 'absolute',
            inset: '0',
            zIndex: '99',
            display: 'none', // Hidden by default
            gridTemplateColumns: 'repeat(10, 1fr)',
            gridTemplateRows: 'repeat(10, 1fr)',
            pointerEvents: 'none' // Make it non-interactive
        });

        // Populate the grid with numbered cells
        for (let y = 0; y < 10; y++) {
            for (let x = 0; x < 10; x++) {
                const cell = document.createElement('div');
                // const xCoord = ((x + 0.5) / 10).toFixed(2);
                // const yCoord = ((y + 0.5) / 10).toFixed(2);
                
                // cell.textContent = `${xCoord}, ${yCoord}`;
                const cellNumber = y * 10 + x + 1;
                cell.textContent = cellNumber;
                Object.assign(cell.style, {
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '10px',
                    color: 'rgba(255, 255, 255, 0.6)'
                });
                gridOverlay.appendChild(cell);
            }
        }
        bodyy.appendChild(gridOverlay);
        // stageContainer.appendChild(gridOverlay);

        // --- Create Toggle Button ---
        const toggleGridBtn = document.createElement('button');
        toggleGridBtn.textContent = 'Toggle Grid';
        toggleGridBtn.className = 'secondary'; // Use an existing style from your CSS

        // --- Add Toggle Functionality ---
        let isGridVisible = false;
        toggleGridBtn.addEventListener('click', () => {
            isGridVisible = !isGridVisible;
            gridOverlay.style.display = isGridVisible ? 'grid' : 'none';
        });

        // Add the new button to the controls area
        controlsContainer.appendChild(toggleGridBtn);
    });

})();