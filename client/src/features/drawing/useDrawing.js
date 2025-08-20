// src/features/drawing/useDrawing.js
import { useState, useEffect, useCallback, useRef } from 'react';
import {
    getCanvasCoordinates,
    clearCanvas,
    getPixelColor,
    rgbaToString,
    createMaskFromShapesAndColorMask, // This function is now async
    generateContiguousColorMask,
    generateGlobalColorMask,
    drawImageOnCanvas,
} from './canvasUtils';
import { MAX_CANVAS_DISPLAY_WIDTH, MAX_CANVAS_DISPLAY_HEIGHT } from '../../config/constants';

// --- Helper Functions (specific to useDrawing hook logic) ---

/**
 * Draws a shape on the canvas, applying rotation if specified for supported types.
 * @param {CanvasRenderingContext2D} ctx - The 2D rendering context.
 * @param {Object} shape - The shape object to draw.
 * @param {string} color - Stroke or fill color.
 * @param {number} size - Line width for stroke.
 * @param {boolean} fill - Whether to fill the shape.
 */
const drawShape = (ctx, shape, color, size, fill = false) => {
    if (!ctx || !shape) return;

    ctx.save(); // Save the current canvas state

    // Apply transformations if shape has a rotation and is a type that can rotate
    if (shape.rotation && (shape.type === 'rectangle' || shape.type === 'circle')) {
        let centerX, centerY;
        if (shape.type === 'rectangle') {
            centerX = shape.x + shape.width / 2;
            centerY = shape.y + shape.height / 2;
        } else if (shape.type === 'circle') {
            centerX = shape.x; // x,y is center for circle
            centerY = shape.y;
        }
        ctx.translate(centerX, centerY);
        ctx.rotate(shape.rotation);
        ctx.translate(-centerX, -centerY);
    }

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = size;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (shape.type === 'brush' || shape.type === 'free_shape') {
        if (shape.points && shape.points.length > 0) {
            ctx.moveTo(shape.points[0].x, shape.points[0].y);
            shape.points.forEach(p => ctx.lineTo(p.x, p.y));
        }
        if (fill && shape.type === 'free_shape') {
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
        }
    } else if (shape.type === 'rectangle') {
        ctx.rect(shape.x, shape.y, shape.width, shape.height);
        if (fill) {
            ctx.fillStyle = color;
            ctx.fill();
        }
    } else if (shape.type === 'circle') {
        ctx.arc(shape.x, shape.y, shape.radius, 0, Math.PI * 2);
        if (fill) {
            ctx.fillStyle = color;
            ctx.fill();
        }
    } else if (shape.type === 'triangle') {
        if (shape.points && shape.points.length === 3) {
            ctx.moveTo(shape.points[0].x, shape.points[0].y);
            ctx.lineTo(shape.points[1].x, shape.points[1].y);
            ctx.lineTo(shape.points[2].x, shape.points[2].y);
            ctx.closePath();
            if (fill) {
                ctx.fillStyle = color;
                ctx.fill();
            }
        }
    }

    if (!fill || shape.type === 'brush') { // Always stroke brush, and stroke other shapes if not filled
        ctx.stroke();
    }

    ctx.restore(); // Restore the canvas state
};

/**
 * Calculates the unrotated bounding box of a shape, and its center.
 * This is kept for isPointInShape, even if not used for handles.
 * @param {Object} shape - The shape object.
 * @returns {{ minX: number, minY: number, maxX: number, maxY: number, width: number, height: number, centerX: number, centerY: number }}
 */
const getShapeBoundingBox = (shape) => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    let centerX, centerY;

    if (shape.type === 'rectangle') {
        minX = shape.x;
        minY = shape.y;
        maxX = shape.x + shape.width;
        maxY = shape.y + shape.height;
        centerX = shape.x + shape.width / 2;
        centerY = shape.y + shape.height / 2;
    } else if (shape.type === 'circle') {
        minX = shape.x - shape.radius;
        minY = shape.y - shape.radius;
        maxX = shape.x + shape.radius;
        maxY = shape.y + shape.radius;
        centerX = shape.x;
        centerY = shape.y;
    } else { // Fallback for unknown or malformed shapes
        minX = shape.x || 0;
        minY = shape.y || 0;
        maxX = (shape.x || 0) + (shape.width || 0);
        maxY = (shape.y || 0) + (shape.height || 0);
        centerX = (minX + maxX) / 2;
        centerY = (minY + maxY) / 2;
    }
    return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY, centerX, centerY };
};

/**
 * Checks if a point (x, y) is inside a given shape, accounting for rotation.
 * @param {CanvasRenderingContext2D} ctx - The 2D rendering context.
 * @param {Object} shape - The shape object.
 * @param {number} x - X coordinate of the point.
 * @param {number} y - Y coordinate of the point.
 * @returns {boolean} - True if the point is inside the shape, false otherwise.
 */
const isPointInShape = (ctx, shape, x, y) => {
    if (!ctx || !shape) return false;

    ctx.save();
    ctx.beginPath();

    // Apply inverse transformations for hit testing on rotated shapes
    if (shape.rotation && (shape.type === 'rectangle' || shape.type === 'circle')) {
        const { centerX, centerY } = getShapeBoundingBox(shape); // Use bbox center for rotation point
        const cos = Math.cos(-shape.rotation); // Inverse rotation
        const sin = Math.sin(-shape.rotation);
        const translatedX = x - centerX;
        const translatedY = y - centerY;
        x = translatedX * cos - translatedY * sin + centerX;
        y = translatedX * sin + translatedY * cos + centerY;
    }

    if (shape.type === 'rectangle') {
        ctx.rect(shape.x, shape.y, shape.width, shape.height);
    } else if (shape.type === 'circle') {
        ctx.arc(shape.x, shape.y, shape.radius, 0, Math.PI * 2);
    } else if (shape.type === 'triangle') {
        if (shape.points && shape.points.length === 3) {
            ctx.moveTo(shape.points[0].x, shape.points[0].y);
            ctx.lineTo(shape.points[1].x, shape.points[1].y);
            ctx.lineTo(shape.points[2].x, shape.points[2].y);
            ctx.closePath();
        }
    } else if (shape.type === 'free_shape') {
        if (shape.points && shape.points.length > 0) {
            ctx.moveTo(shape.points[0].x, shape.points[0].y);
            shape.points.forEach(p => ctx.lineTo(p.x, p.y));
            ctx.closePath(); // Close path for fillable free shapes
        }
    } else if (shape.type === 'brush') {
        // For brush, use a simple bounding box check for selection
        // Calculate min/max x/y from points
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        if (shape.points && shape.points.length > 0) {
            shape.points.forEach(p => {
                minX = Math.min(minX, p.x);
                minY = Math.min(minY, p.y);
                maxX = Math.max(maxX, p.x);
                maxY = Math.max(maxY, p.y);
            });
        }
        const padding = shape.size / 2 + 5; // Add padding for easier selection
        ctx.rect(minX - padding, minY - padding, maxX - minX + 2 * padding, maxY - minY + 2 * padding);
    } else {
        ctx.restore();
        return false; // Unknown shape type
    }

    const isInPath = ctx.isPointInPath(x, y);
    ctx.restore();
    return isInPath;
};

/**
 * Finds the topmost shape at a given position.
 * @param {Array<Object>} shapes - Array of all drawn shapes.
 * @param {number} x - X coordinate of the point.
 * @param {number} y - Y coordinate of the point.
 * @param {CanvasRenderingContext2D} ctx - The 2D rendering context.
 * @returns {Object|null} The found shape object or null.
 */
const findShapeAtPosition = (shapes, x, y, ctx) => {
    // Iterate shapes in reverse order to select the topmost one
    for (let i = shapes.length - 1; i >= 0; i--) {
        const shape = shapes[i];
        // Exclude color masks from selection
        if (shape && shape.type !== 'color_mask' && isPointInShape(ctx, shape, x, y)) { // Use the robust isPointInShape
            return shape;
        }
    }
    return null;
};


/**
 * Draws selection handles (resize, rotate, delete) around a shape.
 * @param {CanvasRenderingContext2D} ctx - The 2D rendering context.
 * @param {Object} shape - The shape object.
 */
const drawSelectionHandles = (ctx, shape) => {
    if (!ctx || !shape) return;

    const { minX, minY, maxX, maxY, width, height, centerX, centerY } = getShapeBoundingBox(shape);
    const handleSize = 8;
    const padding = 5; // Padding around the bounding box for handles

    ctx.save(); // Save the current canvas state for handle drawing

    // Apply shape's rotation to the context for handle positioning relative to rotated shape
    if (shape.rotation && (shape.type === 'rectangle' || shape.type === 'circle')) {
        ctx.translate(centerX, centerY);
        ctx.rotate(shape.rotation);
        ctx.translate(-centerX, -centerY);
    }

    ctx.strokeStyle = 'rgba(0, 150, 255, 0.8)'; // Blue for selection
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]); // Dashed line for bounding box

    // Draw bounding box (relative to the potentially rotated context)
    ctx.strokeRect(minX - padding, minY - padding, width + 2 * padding, height + 2 * padding);
    ctx.setLineDash([]); // Reset line dash

    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 1;

    // Helper to draw a handle
    const drawHandle = (x, y, style = 'rect', fillColor = 'white', strokeColor = 'blue') => {
        ctx.beginPath();
        if (style === 'rect') {
            ctx.rect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize);
        } else if (style === 'circle') {
            ctx.arc(x, y, handleSize / 2, 0, Math.PI * 2);
        }
        ctx.fillStyle = fillColor;
        ctx.strokeStyle = strokeColor;
        ctx.fill();
        ctx.stroke();
    };

    // Resize Handles (8 for rectangles/circles)
    if (shape.type === 'rectangle' || shape.type === 'circle') {
        // Corners
        drawHandle(minX - padding, minY - padding); // NW
        drawHandle(maxX + padding, minY - padding); // NE
        drawHandle(minX - padding, maxY + padding); // SW
        drawHandle(maxX + padding, maxY + padding); // SE

        // Mid-points
        drawHandle(centerX, minY - padding); // N
        drawHandle(maxX + padding, centerY); // E
        drawHandle(centerX, maxY + padding); // S
        drawHandle(minX - padding, centerY); // W
    }

    // Rotation Handle (for rectangles/circles)
    if (shape.type === 'rectangle' || shape.type === 'circle') {
        const rotateHandleY = minY - padding - 20; // Position above the top-center handle
        drawHandle(centerX, rotateHandleY, 'circle', 'lightgray', 'blue'); // Rotation handle
        ctx.beginPath();
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 1;
        ctx.moveTo(centerX, minY - padding);
        ctx.lineTo(centerX, rotateHandleY + handleSize / 2);
        ctx.stroke(); // Line connecting to shape
    }

    // Delete Handle (for all shapes)
    // Position it relative to the top-right corner of the bounding box
    const deleteHandleX = maxX + padding + 15;
    const deleteHandleY = minY - padding;
    drawHandle(deleteHandleX, deleteHandleY, 'circle', 'red', 'darkred');
    ctx.beginPath();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.moveTo(deleteHandleX - 3, deleteHandleY - 3);
    ctx.lineTo(deleteHandleX + 3, deleteHandleY + 3);
    ctx.moveTo(deleteHandleX + 3, deleteHandleY - 3);
    ctx.lineTo(deleteHandleX - 3, deleteHandleY + 3);
    ctx.stroke(); // 'X' for delete

    ctx.restore(); // Restore the canvas state
};


/**
 * Checks if a point (x, y) is within a specific handle area.
 * Returns the handle name ('nw', 'n', 'rotate', 'delete', etc.) or null.
 */
const getHandleAtPoint = (x, y, shape) => {
    const { minX, minY, maxX, maxY, centerX, centerY } = getShapeBoundingBox(shape);
    const handleSize = 8;
    const padding = 5; // Must match padding in drawSelectionHandles

    // Helper to check if point is within a square handle area
    const isPointInHandleArea = (px, py, handleCx, handleCy) => {
        return px >= handleCx - handleSize / 2 && px <= handleCx + handleSize / 2 &&
               py >= handleCy - handleSize / 2 && py <= handleCy + handleSize / 2;
    };

    // Transform mouse coordinates to the shape's local, unrotated space
    let transformedX = x;
    let transformedY = y;
    if (shape.rotation && (shape.type === 'rectangle' || shape.type === 'circle')) {
        const cos = Math.cos(-shape.rotation); // Inverse rotation
        const sin = Math.sin(-shape.rotation);
        const translatedX = x - centerX;
        const translatedY = y - centerY;
        transformedX = translatedX * cos - translatedY * sin + centerX;
        transformedY = translatedX * sin + translatedY * cos + centerY;
    }

    // Delete Handle (for all shapes) - check first as it's outside bounding box
    const deleteHandleX = maxX + padding + 15;
    const deleteHandleY = minY - padding;
    if (isPointInHandleArea(transformedX, transformedY, deleteHandleX, deleteHandleY)) {
        return 'delete';
    }

    // Rotation Handle (for rectangles/circles)
    if (shape.type === 'rectangle' || shape.type === 'circle') {
        const rotateHandleY = minY - padding - 20;
        if (isPointInHandleArea(transformedX, transformedY, centerX, rotateHandleY)) {
            return 'rotate';
        }
    }

    // Resize Handles (8 for rectangles/circles)
    if (shape.type === 'rectangle' || shape.type === 'circle') {
        if (isPointInHandleArea(transformedX, transformedY, minX - padding, minY - padding)) return 'nw';
        if (isPointInHandleArea(transformedX, transformedY, centerX, minY - padding)) return 'n';
        if (isPointInHandleArea(transformedX, transformedY, maxX + padding, minY - padding)) return 'ne';
        if (isPointInHandleArea(transformedX, transformedY, maxX + padding, centerY)) return 'e';
        if (isPointInHandleArea(transformedX, transformedY, maxX + padding, maxY + padding)) return 'se';
        if (isPointInHandleArea(transformedX, transformedY, centerX, maxY + padding)) return 's';
        if (isPointInHandleArea(transformedX, transformedY, minX - padding, maxY + padding)) return 'sw';
        if (isPointInHandleArea(transformedX, transformedY, minX - padding, centerY)) return 'w';
    }

    return null;
};


/**
 * Custom hook for managing drawing operations on a canvas, including
 * brush, shapes, color selection, and shape editing.
 *
 * @param {React.RefObject<HTMLCanvasElement>} originalCanvasRef - Ref to the main canvas for the original image.
 * @param {React.RefObject<HTMLCanvasElement>} drawingOverlayCanvasRef - Ref to the overlay canvas for drawing.
 * @param {React.RefObject<HTMLCanvasElement>} maskCanvasRef - Ref to the canvas where the mask is displayed.
 * @param {React.RefObject<HTMLCanvasElement>} outputCanvasRef - Ref to a hidden canvas for generating masks.
 * @param {string} drawingMode - Current drawing mode ('brush', 'rectangle', 'circle', 'triangle', 'free_shape', 'color_select', 'edit_shape').
 * @param {number} penSize - Size of the drawing pen/brush.
 * @param {string} penColor - Color of the drawing pen/brush.
 * @param {string|null} originalImageSrc - Data URL of the original image.
 * @param {string|null} selectedColor - RGBA string of the color picked for matching.
 * @param {function(string|null): void} setSelectedColor - Setter for selectedColor.
 * @param {number} colorTolerance - Tolerance for color matching.
 * @param {Array<Object>} shapes - Array of all drawn shapes.
 * @param {function(Array<Object>): void} setShapes - Setter for shapes array.
 * @param {Array<{x: number, y: number}>} freeShapePoints - Points for the current free shape being drawn.
 * @param {function(Array<{x: number, y: number}>): void} setFreeShapePoints - Setter for freeShapePoints.
 * @param {string|null} selectedShapeId - ID of the currently selected shape.
 * @param {function(string|null): void} setSelectedShapeId - Setter for selectedShapeId.
 * @param {React.MutableRefObject<boolean>} isRestoringHistoryRef - Ref to signal if history is being restored.
 * @param {React.MutableRefObject<boolean>} shouldSaveHistoryRef - Ref to signal when a new state should be saved.
 * @param {function(boolean): void} setIsDrawingLoading - Setter for drawing loading state.
 * @param {boolean} totalLoading - Overall loading state of the app.
 * @param {boolean} isOriginalCanvasReadyForPixels - Indicates if the original canvas has loaded pixel data.
 * @returns {{
 * activeDisplayMaskSrc: string|null,
 * closeFreeShape: function,
 * deleteSelectedShape: function,
 * handleMouseDown: function,
 * handleMouseMove: function,
 * handleMouseUp: function,
 * handleMouseLeave: function,
 * handleDoubleClick: function,
 * handleContiguousColorMatch: function,
 * handleGlobalColorMatch: function,
 * selectedColorRgbString: string|null
 * }}
 */
const useDrawing = (
    originalCanvasRef,
    drawingOverlayCanvasRef,
    maskCanvasRef,
    outputCanvasRef,
    drawingMode,
    penSize,
    penColor,
    originalImageSrc,
    selectedColor,
    setSelectedColor,
    colorTolerance,
    shapes,
    setShapes,
    freeShapePoints,
    setFreeShapePoints,
    selectedShapeId,
    setSelectedShapeId,
    isRestoringHistoryRef,
    shouldSaveHistoryRef,
    setIsDrawingLoading,
    totalLoading,
    isOriginalCanvasReadyForPixels
) => {
    const [isDrawing, setIsDrawing] = useState(false); // True when mouse is down and an interaction is active
    const [activeDisplayMaskSrc, setActiveDisplayMaskSrc] = useState(null);
    const [startPoint, setStartPoint] = useState({ x: 0, y: 0 }); // Mouse coordinates when drag started for new shapes
    const dragStartPointRef = useRef({ x: 0, y: 0 }); // Mouse coordinates when drag started for editing existing shapes
    const currentShapeRef = useRef(null); // The shape being drawn or actively manipulated (temporary state)
    const initialShapeOnDragStartRef = useRef(null); // State of the selected shape when interaction began

    const [interactionMode, setInteractionMode] = useState('none'); // 'none', 'moving', 'resizing', 'rotating', 'deleting', 'selecting'
    const [activeHandle, setActiveHandle] = useState(null); // Stores the name of the handle being dragged (e.g., 'nw', 'rotate', 'delete')

    // Get canvas contexts
    const originalCtx = originalCanvasRef.current?.getContext('2d');
    const drawingCtx = drawingOverlayCanvasRef.current?.getContext('2d');
    const maskCtx = maskCanvasRef.current?.getContext('2d');
    const outputCtx = outputCanvasRef.current?.getContext('2d');

    // Derived state for display
    const selectedColorRgbString = selectedColor ? rgbaToString(selectedColor) : null;

    // --- Drawing & Mask Generation Effects ---

    // Effect to redraw the drawing overlay and mask canvas whenever shapes or activeDisplayMaskSrc changes
    useEffect(() => {
        if (!drawingCtx || !maskCtx || !originalImageSrc) return;

        // This inner async function is needed because useEffect itself cannot be async
        const updateMaskAndDrawing = async () => {
            clearCanvas(drawingCtx);

            const validShapes = shapes.filter(s => s != null);

            validShapes.forEach(shape => {
                // Skip drawing the selected shape from the main array if we are in edit mode
                // and it's either currently being manipulated (isDrawing) or just selected.
                // It will be drawn separately below with its handles.
                if (drawingMode === 'edit_shape' && selectedShapeId === shape.id) {
                    return;
                }

                if (shape.type !== 'color_mask') {
                    drawShape(drawingCtx, shape, shape.color || penColor, shape.size || penSize, shape.fill);
                }
            });

            // Draw the current free shape being drawn (if any)
            if (drawingMode === 'free_shape' && freeShapePoints.length > 0) {
                drawingCtx.beginPath();
                drawingCtx.strokeStyle = penColor;
                drawingCtx.lineWidth = penSize;
                drawingCtx.lineCap = 'round';
                drawingCtx.lineJoin = 'round';
                drawingCtx.moveTo(freeShapePoints[0].x, freeShapePoints[0].y);
                freeShapePoints.forEach(p => drawingCtx.lineTo(p.x, p.y));
                drawingCtx.stroke();
            }

            // Draw the temporary shape being drawn (rectangle, circle, triangle, or brush) for NEW shapes
            // This only applies when NOT in edit_shape mode or not actively manipulating a selected shape.
            if (isDrawing && currentShapeRef.current &&
                ['rectangle', 'circle', 'triangle', 'brush'].includes(drawingMode) &&
                interactionMode === 'none') {
                drawShape(drawingCtx, currentShapeRef.current, penColor, penSize, true);
            }

            // --- IMPORTANT: Draw the SELECTED shape with its handles in 'edit_shape' mode ---
            if (drawingMode === 'edit_shape' && selectedShapeId) {
                // Prioritize drawing the actively manipulated shape (from currentShapeRef.current)
                // if an interaction is ongoing and it's the selected shape.
                // Otherwise, draw the selected shape as it exists in the main 'shapes' array.
                const shapeToDraw = (isDrawing && currentShapeRef.current && currentShapeRef.current.id === selectedShapeId)
                    ? currentShapeRef.current
                    : validShapes.find(s => s.id === selectedShapeId);

                if (shapeToDraw) {
                    drawShape(drawingCtx, shapeToDraw, shapeToDraw.color || penColor, shapeToDraw.size || penSize, true); // Draw the shape itself
                    drawSelectionHandles(drawingCtx, shapeToDraw); // Draw handles on it
                }
            }

            // Update the mask canvas with the combined mask
            if (outputCtx) {
                // AWAIT the asynchronous createMaskFromShapesAndColorMask
                const maskDataUrl = await createMaskFromShapesAndColorMask(validShapes, drawingCtx.canvas.width, drawingCtx.canvas.height, outputCtx);
                setActiveDisplayMaskSrc(maskDataUrl); // Update the state for display/download
            }
        };

        updateMaskAndDrawing(); // Call the async function

    }, [shapes, drawingCtx, maskCtx, penColor, penSize, freeShapePoints, drawingMode, selectedShapeId, originalImageSrc, outputCtx, isDrawing, interactionMode, currentShapeRef.current]);


    // Effect to draw the activeDisplayMaskSrc onto the maskCanvasRef
    useEffect(() => {
        if (maskCtx) {
            // Set a dark background before drawing the mask
            maskCtx.clearRect(0, 0, maskCtx.canvas.width, maskCtx.canvas.height); // Clear existing content
            maskCtx.fillStyle = '#1a1a1a'; // Dark grey background
            maskCtx.fillRect(0, 0, maskCtx.canvas.width, maskCtx.canvas.height);

            if (activeDisplayMaskSrc) {
                drawImageOnCanvas(maskCtx, activeDisplayMaskSrc);
            }
        }
    }, [activeDisplayMaskSrc, maskCtx]);


    // --- Event Handlers ---

    const closeFreeShape = useCallback(() => {
        if (freeShapePoints.length >= 2) {
            const newShape = {
                id: Date.now().toString(),
                type: 'free_shape',
                points: [...freeShapePoints],
                size: penSize,
                color: penColor,
                fill: true,
                rotation: 0 // Initialize rotation
            };
            setShapes(prevShapes => [...prevShapes, newShape]);
            setFreeShapePoints([]);
            shouldSaveHistoryRef.current = true;
        }
    }, [freeShapePoints, penSize, penColor, setShapes, setFreeShapePoints, shouldSaveHistoryRef]);

    const deleteSelectedShape = useCallback(() => {
        if (selectedShapeId) {
            setShapes(prevShapes => prevShapes.filter(shape => shape.id !== selectedShapeId));
            setSelectedShapeId(null); // Deselect after deletion
            shouldSaveHistoryRef.current = true;
        }
    }, [selectedShapeId, setShapes, setSelectedShapeId, shouldSaveHistoryRef]);


    const handleMouseDown = useCallback((e) => {
        if (!drawingCtx || totalLoading) return;

        const { x, y } = getCanvasCoordinates(e, drawingCtx.canvas);
        setStartPoint({ x, y }); // Update startPoint state for new shapes
        dragStartPointRef.current = { x, y }; // Set drag start point for current mouse position

        if (drawingMode === 'edit_shape') {
            const validShapes = shapes.filter(s => s != null && s.type !== 'color_mask');
            let clickedOnInteractionTarget = false;

            // Check for handle clicks first (iterate in reverse to prioritize topmost shape's handles)
            for (let i = validShapes.length - 1; i >= 0; i--) {
                const shape = validShapes[i];
                const handleName = getHandleAtPoint(x, y, shape); // Use the comprehensive getHandleAtPoint
                if (handleName) {
                    setSelectedShapeId(shape.id);
                    initialShapeOnDragStartRef.current = { ...shape };
                    currentShapeRef.current = { ...shape }; // Copy for manipulation
                    setActiveHandle(handleName);
                    setIsDrawing(true); // Indicate active interaction (a drag will follow)
                    setInteractionMode(handleName === 'delete' ? 'deleting' : (handleName === 'rotate' ? 'rotating' : 'resizing'));
                    clickedOnInteractionTarget = true;
                    break; // Found a handle, stop checking
                }
            }

            // If no handle was clicked, check if a shape itself was clicked for moving
            if (!clickedOnInteractionTarget) {
                // Iterate through shapes in reverse to select the topmost one
                for (let i = validShapes.length - 1; i >= 0; i--) {
                    const shape = validShapes[i];
                    if (isPointInShape(drawingCtx, shape, x, y)) {
                        setSelectedShapeId(shape.id);
                        initialShapeOnDragStartRef.current = { ...shape };
                        currentShapeRef.current = {
                            ...shape,
                            // Store the offset from shape's origin to mouse click for smooth dragging
                            offsetX: x - shape.x,
                            offsetY: y - shape.y,
                        };
                        setInteractionMode('moving'); // Set mode for moving
                        setActiveHandle(null); // No specific handle for moving the whole shape
                        setIsDrawing(true); // Indicate active interaction (a drag will follow)
                        clickedOnInteractionTarget = true;
                        break; // Found a shape, stop checking
                    }
                }
            }

            // If nothing was clicked, deselect any currently selected shape
            if (!clickedOnInteractionTarget) {
                setSelectedShapeId(null);
                setInteractionMode('none'); // No interaction
                setActiveHandle(null);
                setIsDrawing(false); // No drawing/active interaction
                currentShapeRef.current = null;
                initialShapeOnDragStartRef.current = null;
            }
            return; // Exit early for edit_shape mode
        }

        // --- Logic for other drawing modes (original behavior) ---
        // For drawing modes, we start drawing immediately
        setIsDrawing(true);
        setInteractionMode('none'); // Ensure no editing interaction is active for new shape drawing

        const shapeId = Date.now().toString(); // Using Date.now().toString() for IDs

        if (drawingMode === 'brush') {
            currentShapeRef.current = {
                id: shapeId,
                type: 'brush',
                points: [{ x, y }],
                color: penColor,
                size: penSize,
                fill: false,
            };
            drawingCtx.beginPath();
            drawingCtx.strokeStyle = penColor;
            drawingCtx.lineWidth = penSize;
            drawingCtx.lineCap = 'round';
            drawingCtx.lineJoin = 'round';
            drawingCtx.moveTo(x, y);
        } else if (['rectangle', 'circle', 'triangle'].includes(drawingMode)) {
            currentShapeRef.current = {
                id: shapeId,
                type: drawingMode,
                x,
                y,
                width: 0,
                height: 0,
                color: penColor,
                size: penSize,
                fill: true,
                rotation: 0 // Initialize rotation for new shapes
            };
            if (drawingMode === 'triangle') {
                // For triangle, store points relative to start for drawing
                currentShapeRef.current.points = [{ x, y }, { x, y }, { x, y }];
            }
        } else if (drawingMode === 'free_shape') {
            setFreeShapePoints(prevPoints => [...prevPoints, { x, y }]);
            shouldSaveHistoryRef.current = true;
            setIsDrawing(false); // Free shape points are added on click, not a continuous drag
        } else if (drawingMode === 'color_select' && originalCtx && isOriginalCanvasReadyForPixels) {
            const pixelColor = getPixelColor(originalCtx, x, y);
            setSelectedColor(pixelColor);
            setIsDrawing(false); // Color select is a single click action, not a drag
        }
    }, [drawingCtx, drawingMode, penColor, penSize, setFreeShapePoints, originalCtx, isOriginalCanvasReadyForPixels, shapes, setSelectedColor, setSelectedShapeId, originalImageSrc, totalLoading]);


    const handleMouseMove = useCallback((event) => {
        // Only proceed if an interaction is active
        if (!isDrawing || !drawingCtx || !originalImageSrc || totalLoading || !currentShapeRef.current) return;

        const { x, y } = getCanvasCoordinates(event, drawingCtx.canvas);

        if (drawingMode === 'edit_shape') {
            if (!initialShapeOnDragStartRef.current) return; // Should not happen if interactionMode is set correctly

            let updatedShape = { ...initialShapeOnDragStartRef.current }; // Always start from initial state for calculations

            if (interactionMode === 'moving') {
                // Use the stored offset to calculate the new position relative to the mouse
                const newX = x - currentShapeRef.current.offsetX; // currentShapeRef.current holds offsetX/offsetY from handleMouseDown
                const newY = y - currentShapeRef.current.offsetY;
                if (updatedShape.type === 'rectangle' || updatedShape.type === 'circle') {
                    updatedShape.x = newX;
                    updatedShape.y = newY;
                } else if (updatedShape.points && updatedShape.points.length > 0) {
                    // For shapes with points, calculate the delta from the *initial* shape's origin
                    // and apply it to all points.
                    const initialShapeBBox = getShapeBoundingBox(initialShapeOnDragStartRef.current);
                    // Calculate the delta needed to move the initial shape's top-left to the new desired position
                    const deltaX = newX - initialShapeBBox.minX;
                    const deltaY = newY - initialShapeBBox.minY;

                    updatedShape.points = updatedShape.points.map(p => ({
                        x: p.x + deltaX,
                        y: p.y + deltaY
                    }));
                }
            } else if (interactionMode === 'resizing') {
                if (updatedShape.type === 'rectangle') {
                    let newX = initialShapeOnDragStartRef.current.x;
                    let newY = initialShapeOnDragStartRef.current.y;
                    let newWidth = initialShapeOnDragStartRef.current.width;
                    let newHeight = initialShapeOnDragStartRef.current.height;

                    // Calculate delta from the original drag start point
                    const dx = x - dragStartPointRef.current.x;
                    const dy = y - dragStartPointRef.current.y;

                    // Calculate new dimensions based on active handle and mouse movement
                    switch (activeHandle) {
                        case 'nw': newX = initialShapeOnDragStartRef.current.x + dx; newY = initialShapeOnDragStartRef.current.y + dy; newWidth = initialShapeOnDragStartRef.current.width - dx; newHeight = initialShapeOnDragStartRef.current.height - dy; break;
                        case 'n': newY = initialShapeOnDragStartRef.current.y + dy; newHeight = initialShapeOnDragStartRef.current.height - dy; break;
                        case 'ne': newY = initialShapeOnDragStartRef.current.y + dy; newWidth = initialShapeOnDragStartRef.current.width + dx; newHeight = initialShapeOnDragStartRef.current.height - dy; break;
                        case 'e': newWidth = initialShapeOnDragStartRef.current.width + dx; break;
                        case 'se': newWidth = initialShapeOnDragStartRef.current.width + dx; newHeight = initialShapeOnDragStartRef.current.height + dy; break;
                        case 's': newHeight = initialShapeOnDragStartRef.current.height + dy; break;
                        case 'sw': newX = initialShapeOnDragStartRef.current.x + dx; newWidth = initialShapeOnDragStartRef.current.width - dx; newHeight = initialShapeOnDragStartRef.current.height + dy; break;
                        case 'w': newX = initialShapeOnDragStartRef.current.x + dx; newWidth = initialShapeOnDragStartRef.current.width - dx; break;
                        default: break;
                    }
                    updatedShape = { ...updatedShape, x: newX, y: newY, width: Math.max(1, newWidth), height: Math.max(1, newHeight) };

                } else if (updatedShape.type === 'circle') {
                    const { centerX, centerY } = getShapeBoundingBox(initialShapeOnDragStartRef.current);
                    const newRadius = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
                    updatedShape = { ...updatedShape, radius: Math.max(1, newRadius) };
                }
            } else if (interactionMode === 'rotating') {
                if (updatedShape.type === 'rectangle' || updatedShape.type === 'circle') {
                    const { centerX, centerY } = getShapeBoundingBox(initialShapeOnDragStartRef.current);
                    const startAngle = Math.atan2(dragStartPointRef.current.y - centerY, dragStartPointRef.current.x - centerX);
                    const currentAngle = Math.atan2(y - centerY, x - centerX);
                    const angleDelta = currentAngle - startAngle;

                    updatedShape.rotation = (initialShapeOnDragStartRef.current.rotation || 0) + angleDelta;
                }
            }
            currentShapeRef.current = updatedShape; // Update the temporary shape in ref
            // dragStartPointRef.current is NOT updated here for edit mode, it remains fixed from handleMouseDown
            return; // Exit early for edit_shape mode
        }

        // --- Logic for other drawing modes (original behavior) ---
        const startX = startPoint.x; // Use startPoint state for drawing modes
        const startY = startPoint.y;

        if (drawingMode === 'brush') {
            currentShapeRef.current.points.push({ x, y });
            drawingCtx.lineTo(x, y);
            drawingCtx.stroke();
        } else if (drawingMode === 'rectangle') {
            const width = x - startX;
            const height = y - startY;
            currentShapeRef.current = { ...currentShapeRef.current, x: startX, y: startY, width, height };
        } else if (drawingMode === 'circle') {
            const radius = Math.sqrt(Math.pow(x - startX, 2) + Math.pow(y - startY, 2));
            currentShapeRef.current = { ...currentShapeRef.current, x: startX, y: startY, radius };
        } else if (drawingMode === 'triangle') {
            currentShapeRef.current.points = [
                { x: startX, y: startY },
                { x: x, y: startY },
                { x: (startX + x) / 2, y: y } // Apex of triangle
            ];
        }
    }, [isDrawing, drawingCtx, drawingMode, penSize, originalImageSrc, totalLoading, interactionMode, activeHandle, startPoint]);


    const handleMouseUp = useCallback(() => {
        if (!drawingCtx || totalLoading) return; // Removed `!isDrawing` check here to allow deselection clicks

        // If in edit_shape mode and an interaction was active
        if (drawingMode === 'edit_shape' && selectedShapeId) {
            if (interactionMode === 'deleting') {
                setShapes(prevShapes => prevShapes.filter(s => s && s.id !== selectedShapeId));
                shouldSaveHistoryRef.current = true;
            } else if (isDrawing && currentShapeRef.current) { // Only update if an actual drag/edit occurred
                const updatedShape = { ...currentShapeRef.current };
                // Clean up temporary properties before saving
                delete updatedShape.offsetX;
                delete updatedShape.offsetY;

                setShapes(prevShapes => {
                    const safePrevShapes = Array.isArray(prevShapes) ? prevShapes : [];
                    return safePrevShapes.map(s =>
                        s && s.id === selectedShapeId ? updatedShape : s
                    ).filter(s => s != null);
                });
                shouldSaveHistoryRef.current = true;
            }
            // Reset interaction states for edit mode, but keep selectedShapeId if it was just a click
            setIsDrawing(false); // No longer actively dragging
            setInteractionMode('none'); // No longer in an active interaction mode
            setActiveHandle(null);
            currentShapeRef.current = null; // Clear temporary shape
            initialShapeOnDragStartRef.current = null;
            // Do NOT clear selectedShapeId here if it was just a selection click
            // It should only be cleared if clicking outside a shape, or if the shape is deleted.
            return; // Exit early for edit_shape mode
        }

        // --- Logic for other drawing modes (original behavior) ---
        if (isDrawing && ['brush', 'rectangle', 'circle', 'triangle', 'free_shape'].includes(drawingMode) && currentShapeRef.current) {
            // Free shape points are added on click, not finalized on mouseUp
            if (drawingMode !== 'free_shape' && drawingMode !== 'color_select') {
                const newShape = currentShapeRef.current;
                setShapes(prevShapes => [...prevShapes, newShape]);
                shouldSaveHistoryRef.current = true;
            }
        }

        setIsDrawing(false); // Stop drawing for other modes
        currentShapeRef.current = null; // Clear the temporary shape for other modes
    }, [drawingCtx, drawingMode, penSize, setShapes, selectedShapeId, totalLoading, shouldSaveHistoryRef, interactionMode, isDrawing]);


    const handleMouseLeave = useCallback(() => {
        // If drawing with brush/shape, cancel the current drawing
        if (isDrawing && ['brush', 'rectangle', 'circle', 'triangle'].includes(drawingMode)) {
            setIsDrawing(false);
            currentShapeRef.current = null;
            clearCanvas(drawingCtx); // Clear temporary drawing
        }
        // For edit_shape, if mouse leaves while dragging, finalize the action
        // Ensure it only finalizes if an actual drag interaction was in progress
        if (isDrawing && drawingMode === 'edit_shape' && interactionMode !== 'none' && interactionMode !== 'deleting') {
            handleMouseUp(); // Call mouseUp to finalize current interaction
        }
    }, [isDrawing, drawingMode, drawingCtx, handleMouseUp, interactionMode]);


    const handleDoubleClick = useCallback((event) => {
        if (!drawingCtx || !originalImageSrc || totalLoading) return;

        const { x, y } = getCanvasCoordinates(event, drawingCtx.canvas);

        if (drawingMode === 'free_shape' && freeShapePoints.length > 0) {
            closeFreeShape();
        }
    }, [drawingCtx, drawingMode, freeShapePoints, closeFreeShape, originalImageSrc, totalLoading]);


    const handleContiguousColorMatch = useCallback(async () => {
        if (!originalCtx || !outputCtx || !originalImageSrc || !selectedColor || totalLoading) return;
        setIsDrawingLoading(true);

        try {
            const maskDataUrl = await generateContiguousColorMask(
                originalCtx,
                originalImageSrc,
                rgbaToString(selectedColor),
                colorTolerance,
                outputCtx
            );
            if (maskDataUrl) {
                const newMaskShape = {
                    id: `color_mask_${Date.now()}`,
                    type: 'color_mask',
                    dataUrl: maskDataUrl,
                    x: 0, y: 0,
                    width: originalCtx.canvas.width,
                    height: originalCtx.canvas.height
                };
                setShapes(prevShapes => {
                    const safePrevShapes = Array.isArray(prevShapes) ? prevShapes : [];
                    return [...safePrevShapes.filter(s => s != null && s.type !== 'color_mask'), newMaskShape];
                });
                shouldSaveHistoryRef.current = true;
            }
        } catch (error) {
            console.error("Error generating contiguous color mask:", error);
        } finally {
            setIsDrawingLoading(false);
        }
    }, [originalCtx, outputCtx, originalImageSrc, selectedColor, colorTolerance, setShapes, setIsDrawingLoading, totalLoading, shouldSaveHistoryRef]);


    const handleGlobalColorMatch = useCallback(async () => {
        if (!originalCtx || !outputCtx || !originalImageSrc || !selectedColor || totalLoading) return;
        setIsDrawingLoading(true);

        try {
            const maskDataUrl = await generateGlobalColorMask(
                originalCtx,
                originalImageSrc,
                rgbaToString(selectedColor),
                colorTolerance,
                outputCtx
            );
            if (maskDataUrl) {
                const newMaskShape = {
                    id: `color_mask_${Date.now()}`,
                    type: 'color_mask',
                    dataUrl: maskDataUrl,
                    x: 0, y: 0,
                    width: originalCtx.canvas.width,
                    height: originalCtx.canvas.height
                };
                setShapes(prevShapes => {
                    const safePrevShapes = Array.isArray(prevShapes) ? prevShapes : [];
                    return [...safePrevShapes.filter(s => s != null && s.type !== 'color_mask'), newMaskShape];
                });
                shouldSaveHistoryRef.current = true;
            }
        } catch (error) {
            console.error("Error generating global color mask:", error);
        } finally {
            setIsDrawingLoading(false);
        }
    }, [originalCtx, outputCtx, originalImageSrc, selectedColor, colorTolerance, setShapes, setIsDrawingLoading, totalLoading, shouldSaveHistoryRef]);


    return {
        activeDisplayMaskSrc,
        closeFreeShape,
        deleteSelectedShape,
        handleMouseDown,
        handleMouseMove,
        handleMouseUp,
        handleMouseLeave,
        handleDoubleClick,
        handleContiguousColorMatch,
        handleGlobalColorMatch,
        selectedColorRgbString,
    };
};

export default useDrawing;
