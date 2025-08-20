   // src/features/history/useHistory.js
   import { useState, useEffect, useCallback } from 'react';
   import { clearCanvas, drawImageOnCanvas } from '../drawing/canvasUtils';

   /**
    * Custom hook for managing undo/redo history for shapes and masks.
    * It centralizes the history stack and provides functions to manipulate it.
    *
    * @param {string|null} originalImageSrc - The Data URL of the original image (needed for redrawing).
    * @param {function(Array<Object>): void} setShapes - Setter for the shapes array (managed by parent, e.g., App.js).
    * @param {function(string|null): void} setColorPickedMaskDataUrl - Setter for the color-picked mask data URL (managed by parent).
    * @param {React.RefObject<HTMLCanvasElement>} maskCanvasRef - Ref to the mask display canvas.
    * @param {React.MutableRefObject<boolean>} isRestoringHistoryRef - Ref to signal if history is being restored.
    * @returns {{
    * undo: function,
    * redo: function,
    * reset: function,
    * canUndo: boolean,
    * canRedo: boolean,
    * saveState: function // Exposed for explicit saves (e.g., on initial load)
    * }}
    */
   const useHistory = (
       originalImageSrc,
       setShapes, // Now received as a prop from App.js
       setColorPickedMaskDataUrl, // Now received as a prop from App.js
       maskCanvasRef,
       isRestoringHistoryRef
   ) => {
       // History stacks are internal to useHistory
       const [undoStack, setUndoStack] = useState([]);
       const [redoStack, setRedoStack] = useState([]);
       const [historyIndex, setHistoryIndex] = useState(-1); // -1 means no history yet, 0 is the first state

       // Function to apply a historical state
       const applyHistoryState = useCallback((state) => {
           isRestoringHistoryRef.current = true; // Set flag to prevent saving this state to history
           setShapes(state.shapes);
           setColorPickedMaskDataUrl(state.colorPickedMaskDataUrl);

           // Redraw canvases based on the restored state
           const maskCtx = maskCanvasRef.current?.getContext('2d');
           if (maskCtx) {
               clearCanvas(maskCtx);
               // If the mask was part of the state, draw it
               if (state.activeDisplayMaskSrc) {
                   drawImageOnCanvas(maskCtx, state.activeDisplayMaskSrc);
               }
           }

           // After state updates, reset the flag
           // Use a timeout to ensure React has processed the state updates
           setTimeout(() => {
               isRestoringHistoryRef.current = false;
               console.log("useHistory: applyHistoryState - isRestoringHistory flag reset to false.");
           }, 0);
       }, [setShapes, setColorPickedMaskDataUrl, maskCanvasRef, isRestoringHistoryRef]);


       // Save current state to history
       // Now explicitly takes shapes, colorPickedMaskDataUrl, and activeDisplayMaskSrc as arguments
       const saveState = useCallback((currentShapes, currentColorPickedMaskDataUrl, currentActiveDisplayMaskSrc) => {
           // Only save if not currently restoring history
           if (isRestoringHistoryRef.current) {
               console.log("useHistory: saveState - Skipping save because history is being restored.");
               return;
           }

           const currentState = {
               shapes: currentShapes,
               colorPickedMaskDataUrl: currentColorPickedMaskDataUrl,
               activeDisplayMaskSrc: currentActiveDisplayMaskSrc
           };

           setUndoStack(prevStack => {
               const newStack = prevStack.slice(0, historyIndex + 1); // Truncate redo history
               return [...newStack, currentState];
           });
           setHistoryIndex(prevIndex => prevIndex + 1);
           setRedoStack([]); // Clear redo stack on new action
           console.log("useHistory: saveState - State saved. New history index:", historyIndex + 1);

       }, [historyIndex, isRestoringHistoryRef]); // Dependencies only include internal state that affects history stack management


       // Undo function
       const undo = useCallback((currentShapes, currentColorPickedMaskDataUrl, currentActiveDisplayMaskSrc) => {
           if (historyIndex > 0) {
               const prevState = undoStack[historyIndex - 1];
               // Before applying previous state, save current state to redo stack
               setRedoStack(prevStack => [{
                   shapes: currentShapes,
                   colorPickedMaskDataUrl: currentColorPickedMaskDataUrl,
                   activeDisplayMaskSrc: currentActiveDisplayMaskSrc
               }, ...prevStack]);
               setHistoryIndex(prevIndex => prevIndex - 1);
               applyHistoryState(prevState);
               console.log("useHistory: Undo - Applied state at index:", historyIndex - 1);
           } else {
               console.log("useHistory: Undo - No more history to undo.");
           }
       }, [historyIndex, undoStack, applyHistoryState]);


       // Redo function
       const redo = useCallback((currentShapes, currentColorPickedMaskDataUrl, currentActiveDisplayMaskSrc) => {
           if (redoStack.length > 0) {
               const nextState = redoStack[0];
               // Before applying next state, save current state to undo stack
               setUndoStack(prevStack => [...prevStack, {
                   shapes: currentShapes,
                   colorPickedMaskDataUrl: currentColorPickedMaskDataUrl,
                   activeDisplayMaskSrc: currentActiveDisplayMaskSrc
               }]);
               setHistoryIndex(prevIndex => prevIndex + 1);
               setRedoStack(prevStack => prevStack.slice(1)); // Remove from redo stack
               applyHistoryState(nextState);
               console.log("useHistory: Redo - Applied state at index:", historyIndex + 1);
           } else {
               console.log("useHistory: Redo - No more history to redo.");
           }
       }, [redoStack, undoStack, applyHistoryState]);


       // Reset function
       const reset = useCallback((currentShapes, currentColorPickedMaskDataUrl, currentActiveDisplayMaskSrc) => {
           isRestoringHistoryRef.current = true; // Set flag to prevent saving empty state to history immediately

           setUndoStack([]);
           setRedoStack([]);
           setHistoryIndex(-1);
           setShapes([]); // Clear shapes (from App.js)
           setColorPickedMaskDataUrl(null); // Clear color-picked mask (from App.js)

           // Crucial: Reset flags *after* the state updates have a chance to propagate.
           // Then, explicitly save the new empty state to history.
           setTimeout(() => {
               isRestoringHistoryRef.current = false;
               console.log("useHistory: Reset - isRestoringHistory flag reset to false.");
               saveState([], null, null); // Save the cleared state
           }, 0);


           if (maskCanvasRef.current) {
               clearCanvas(maskCanvasRef.current.getContext('2d'));
           }
           console.log("useHistory: Reset - All drawing states cleared.");
       }, [setShapes, setColorPickedMaskDataUrl, maskCanvasRef, isRestoringHistoryRef, saveState]);

       const canUndo = historyIndex > 0;
       const canRedo = historyIndex < undoStack.length - 1;

       return {
           undo,
           redo,
           reset,
           canUndo,
           canRedo,
           saveState // Export saveState for App.js to call explicitly on initial load
       };
   };

   export default useHistory;
   