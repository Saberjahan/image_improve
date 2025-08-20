   import { useRef, useEffect, useState, useCallback } from 'react';
   import { drawImageOnCanvas, clearCanvas } from '../features/drawing/canvasUtils';
   import { MAX_CANVAS_DISPLAY_WIDTH, MAX_CANVAS_DISPLAY_HEIGHT } from '../config/constants';

   /**
    * Custom hook for managing HTML canvas elements, their 2D rendering contexts,
    * and dynamically setting their dimensions based on loaded image aspect ratios
    * and predefined maximum display sizes from constants.
    *
    * @param {React.RefObject<HTMLCanvasElement>} originalCanvasRef - Ref to the main canvas for the original image.
    * @param {React.RefObject<HTMLCanvasElement>} maskCanvasRef - Ref to the canvas for displaying masks or repaired images.
    * @param {React.RefObject<HTMLCanvasElement>} outputCanvasRef - Ref to a hidden canvas for off-screen processing.
    * @param {string|null} originalImageSrc - The Data URL of the original image to be displayed.
    * @param {function(boolean): void} setIsOriginalCanvasReadyForPixels - Setter to indicate if original canvas has loaded pixel data.
    * @returns {{canvasWidth: number, canvasHeight: number, originalCtx: CanvasRenderingContext2D|null, maskCtx: CanvasRenderingContext2D|null, outputCtx: CanvasRenderingContext2D|null}}\
    * An object containing the calculated canvas dimensions and their 2D contexts.
    */
   const useCanvas = (originalCanvasRef, maskCanvasRef, outputCanvasRef, originalImageSrc, setIsOriginalCanvasReadyForPixels) => {
       const [canvasWidth, setCanvasWidth] = useState(0);
       const [canvasHeight, setCanvasHeight] = useState(0);

       // Contexts are managed internally by the hook and returned
       const originalCtx = originalCanvasRef.current ? originalCanvasRef.current.getContext('2d') : null;
       const maskCtx = maskCanvasRef.current ? maskCanvasRef.current.getContext('2d') : null;
       const outputCtx = outputCanvasRef.current ? outputCanvasRef.current.getContext('2d') : null;


       // Function to calculate and set canvas dimensions based on image aspect ratio
       const setCanvasDimensions = useCallback((imgWidth, imgHeight) => {
           if (imgWidth === 0 || imgHeight === 0) {
               setCanvasWidth(0);
               setCanvasHeight(0);
               return;
           }

           const imageAspectRatio = imgWidth / imgHeight;
           let newWidth = MAX_CANVAS_DISPLAY_WIDTH;
           let newHeight = MAX_CANVAS_DISPLAY_HEIGHT;

           // Scale to fit within max dimensions while maintaining aspect ratio
           if (imageAspectRatio > (MAX_CANVAS_DISPLAY_WIDTH / MAX_CANVAS_DISPLAY_HEIGHT)) {
               // Image is wider than the max display aspect ratio
               newHeight = MAX_CANVAS_DISPLAY_WIDTH / imageAspectRatio;
           } else {
               // Image is taller or has similar aspect ratio
               newWidth = MAX_CANVAS_DISPLAY_HEIGHT * imageAspectRatio;
           }

           setCanvasWidth(Math.floor(newWidth));
           setCanvasHeight(Math.floor(newHeight));
       }, []);


       // Effect to draw the original image and set canvas dimensions
       useEffect(() => {
           if (originalImageSrc) {
               const img = new Image();
               img.onload = () => {
                   setCanvasDimensions(img.width, img.height); // Set dimensions based on actual image size
                   const ctx = originalCanvasRef.current.getContext('2d');
                   drawImageOnCanvas(ctx, originalImageSrc); // Draw image scaled to fit
                   // Safely call setIsOriginalCanvasReadyForPixels
                   if (typeof setIsOriginalCanvasReadyForPixels === 'function') {
                       console.log("useCanvas: Calling setIsOriginalCanvasReadyForPixels(true) on image load success.");
                       setIsOriginalCanvasReadyForPixels(true);
                   } else {
                       console.warn("useCanvas: setIsOriginalCanvasReadyForPixels is not a function when trying to set true on image load success.");
                   }
               };
               img.onerror = () => {
                   console.error("useCanvas: Failed to load original image.");
                   // Clear canvas and reset dimensions on error
                   setCanvasDimensions(0, 0);
                   if (originalCanvasRef.current) {
                       clearCanvas(originalCanvasRef.current.getContext('2d'));
                   }
                   // Safely call setIsOriginalCanvasReadyForPixels
                   if (typeof setIsOriginalCanvasReadyForPixels === 'function') {
                       console.log("useCanvas: Calling setIsOriginalCanvasReadyForPixels(false) on image load error.");
                       setIsOriginalCanvasReadyForPixels(false);
                   } else {
                       console.warn("useCanvas: setIsOriginalCanvasReadyForPixels is not a function when trying to set false on image load error.");
                   }
               };
               img.src = originalImageSrc;
           } else {
               console.log("useCanvas: originalImageSrc is null, clearing canvas and setting ready state to false.");
               setCanvasDimensions(0, 0);
               if (originalCanvasRef.current) {
                   clearCanvas(originalCanvasRef.current.getContext('2d'));
               }
               // Safely call setIsOriginalCanvasReadyForPixels
               if (typeof setIsOriginalCanvasReadyForPixels === 'function') {
                   console.log("useCanvas: Calling setIsOriginalCanvasReadyForPixels(false) because originalImageSrc is null.");
                   setIsOriginalCanvasReadyForPixels(false);
               } else {
                   console.warn("useCanvas: setIsOriginalCanvasReadyForPixels is not a function when trying to set false because originalImageSrc is null.");
               }
           }
       }, [originalImageSrc, setCanvasDimensions, originalCanvasRef, setIsOriginalCanvasReadyForPixels]);

       return {
           canvasWidth,
           canvasHeight,
           originalCtx,
           maskCtx,
           outputCtx
       };
   };

   export default useCanvas;
   