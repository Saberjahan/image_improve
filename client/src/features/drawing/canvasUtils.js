// src/features/drawing/canvasUtils.js
// This file contains utility functions for canvas operations and drawing.

/**
 * Calculates canvas coordinates from a mouse or touch event.
 * @param {Event} event - The mouse or touch event.
 * @param {HTMLCanvasElement} canvas - The canvas element.
 * @returns {{x: number, y: number}} - The calculated coordinates.
 */
export const getCanvasCoordinates = (event, canvas) => {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if (event.touches && event.touches.length > 0) {
        clientX = event.touches[0].clientX;
        clientY = event.touches[0].clientY;
    } else {
        clientX = event.clientX;
        clientY = event.clientY;
    }

    // Calculate coordinates relative to the canvas drawing surface
    const x = (clientX - rect.left) * (canvas.width / rect.width);
    const y = (clientY - rect.top) * (canvas.height / rect.height);

    return { x, y };
};

/**
 * Draws a shape on the given canvas context.
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 * @param {Object} shape - The shape object to draw.
 * @param {string} strokeColor - The color for the shape's stroke.
 * @param {number} strokeSize - The size for the shape's stroke.
 * @param {boolean} fill - Whether to fill the shape.
 */
export const drawShape = (ctx, shape, strokeColor, strokeSize, fill = false) => {
    // Note: This drawShape is primarily for drawing individual shapes like rectangles, circles, brushes.
    // For 'color_mask' shapes, they are handled directly in createMaskFromShapesAndColorMask
    // by drawing their dataUrl as an image, not by this function's switch.
    // This function is called by useDrawing for drawing *editable* shapes on the overlay.

    ctx.beginPath();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeSize;
    ctx.fillStyle = shape.color || 'rgba(255,255,255,0.5)'; // Default fill for color_mask

    switch (shape.type) {
        case 'brush':
        case 'free_shape':
            if (shape.points && shape.points.length > 0) {
                ctx.moveTo(shape.points[0].x, shape.points[0].y);
                shape.points.forEach(p => ctx.lineTo(p.x, p.y));
                if (shape.type === 'free_shape') {
                    ctx.closePath(); // Close polygon for free shape
                }
            }
            break;
        case 'rectangle':
            ctx.rect(shape.x, shape.y, shape.width, shape.height);
            break;
        case 'circle':
            ctx.arc(shape.x, shape.y, shape.radius, 0, Math.PI * 2);
            break;
        case 'triangle':
            if (shape.points && shape.points.length >= 3) {
                ctx.moveTo(shape.points[0].x, shape.points[0].y);
                ctx.lineTo(shape.points[1].x, shape.points[1].y);
                ctx.lineTo(shape.points[2].x, shape.points[2].y);
                ctx.closePath();
            }
            break;
        // 'color_mask' type is handled directly in createMaskFromShapesAndColorMask via drawImage,
        // not intended to be drawn here using paths.
        default:
            console.warn(`Unknown shape type for direct drawing: ${shape.type}`);
            return;
    }

    if (fill) {
        ctx.fill();
    }
    ctx.stroke();
};

/**
 * Clears the entire canvas.
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 */
export const clearCanvas = (ctx) => {
    if (ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }
};

/**
 * Draws an image onto a canvas, scaling it to fit while maintaining aspect ratio.
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 * @param {string} imgSrc - The Data URL of the image to draw.
 */
export const drawImageOnCanvas = (ctx, imgSrc) => {
    const img = new Image();
    img.onload = () => {
        const canvas = ctx.canvas;
        const aspectRatio = img.width / img.height;
        let drawWidth = canvas.width;
        let drawHeight = canvas.height;

        if (canvas.width / canvas.height > aspectRatio) {
            drawWidth = canvas.height * aspectRatio;
        } else {
            drawHeight = canvas.width / aspectRatio;
        }

        const xOffset = (canvas.width - drawWidth) / 2;
        const yOffset = (canvas.height - drawHeight) / 2;

        clearCanvas(ctx);
        ctx.drawImage(img, xOffset, yOffset, drawWidth, drawHeight);
    };
    img.src = imgSrc;
};

/**
 * Gets the RGBA color of a pixel at specified coordinates.
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 * @param {number} x - The x-coordinate.
 * @param {number} y - The y-coordinate.
 * @returns {Array<number>} - An array [R, G, B, A] of the pixel color.
 */
export const getPixelColor = (ctx, x, y) => {
    const imageData = ctx.getImageData(x, y, 1, 1);
    return imageData.data; // This is a Uint8ClampedArray [R, G, B, A]
};

/**
 * Converts an RGBA array to a CSS rgba() string.
 * @param {Array<number>} rgba - An array [R, G, B, A].
 * @returns {string} - The CSS rgba() string.
 */
export const rgbaToString = (rgba) => {
    if (!rgba || rgba.length < 3) return 'rgba(0,0,0,0)';
    // For alpha, imageData.data[3] is 0-255, convert to 0-1 for CSS rgba()
    const alpha = rgba.length === 4 ? rgba[3] / 255 : 1;
    return `rgba(${rgba[0]},${rgba[1]},${rgba[2]},${alpha})`;
};

/**
 * Calculates the Euclidean distance between two colors in RGB space.
 * @param {Array<number>} color1 - [R, G, B] array for the first color.
 * @param {Array<number>} color2 - [R, G, B] array for the second color.
 * @returns {number} - The distance.
 */
const colorDistance = (color1, color2) => {
    return Math.sqrt(
        Math.pow(color1[0] - color2[0], 2) +
        Math.pow(color1[1] - color2[1], 2) +
        Math.pow(color1[2] - color2[2], 2)
    );
};

/**
 * Converts an RGBA string to an RGB array.
 * @param {string} rgbaString - The CSS rgba() string.
 * @returns {Array<number>} - An array [R, G, B].
 */
const rgbaStringToRgbArray = (rgbaString) => {
    const parts = rgbaString.match(/\d+/g);
    if (!parts || parts.length < 3) return [0, 0, 0];
    return [parseInt(parts[0]), parseInt(parts[1]), parseInt(parts[2])];
};

/**
 * Generates a mask for contiguous pixels of a selected color within a tolerance.
 * Uses a flood-fill like algorithm.
 * @param {CanvasRenderingContext2D} originalCtx - Context of the original image.
 * @param {string} originalImageSrc - Data URL of the original image.
 * @param {string} targetColorRgbaString - The RGBA string of the color to match.
 * @param {number} tolerance - The color tolerance (0-255).
 * @param {CanvasRenderingContext2D} outputCtx - Hidden canvas context for generating the mask.
 * @returns {Promise<string|null>} - Data URL of the generated mask.
 */
export const generateContiguousColorMask = async (originalCtx, originalImageSrc, targetColorRgbaString, tolerance, outputCtx) => {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvasWidth = img.width;
            const canvasHeight = img.height;

            // Draw original image onto the originalCtx (if not already there at full resolution)
            // This ensures we have the pixel data to read from
            originalCtx.canvas.width = canvasWidth;
            originalCtx.canvas.height = canvasHeight;
            originalCtx.drawImage(img, 0, 0);

            outputCtx.canvas.width = canvasWidth;
            outputCtx.canvas.height = canvasHeight;
            clearCanvas(outputCtx); // Clear output canvas for the new mask

            const imageData = originalCtx.getImageData(0, 0, canvasWidth, canvasHeight);
            const pixels = imageData.data;
            const maskImageData = outputCtx.createImageData(canvasWidth, canvasHeight);
            const maskPixels = maskImageData.data;

            const targetRgb = rgbaStringToRgbArray(targetColorRgbaString);
            const visited = new Uint8Array(canvasWidth * canvasHeight); // 0 = not visited, 1 = visited

            // Find the starting point (assumed to be where the color was picked)
            // For simplicity, let's assume the first pixel of the target color found
            // In a real app, you'd pass the exact click coordinates.
            let startX = -1, startY = -1;
            for (let i = 0; i < pixels.length; i += 4) {
                const r = pixels[i];
                const g = pixels[i + 1];
                const b = pixels[i + 2];
                if (colorDistance([r, g, b], targetRgb) <= tolerance) {
                    startX = (i / 4) % canvasWidth;
                    startY = Math.floor((i / 4) / canvasWidth);
                    break;
                }
            }

            if (startX === -1) {
                console.warn("No starting pixel found for contiguous match within tolerance.");
                resolve(null);
                return;
            }

            const queue = [{ x: startX, y: startY }];
            visited[startY * canvasWidth + startX] = 1;

            while (queue.length > 0) {
                const { x, y } = queue.shift();
                const pixelIndex = (y * canvasWidth + x) * 4;

                // Set mask pixel to white
                maskPixels[pixelIndex] = 255;
                maskPixels[pixelIndex + 1] = 255;
                maskPixels[pixelIndex + 2] = 255;
                maskPixels[pixelIndex + 3] = 255; // Full opacity

                const checkNeighbor = (nx, ny) => {
                    if (nx >= 0 && nx < canvasWidth && ny >= 0 && ny < canvasHeight) {
                        const neighborIndex = (ny * canvasWidth + nx);
                        if (!visited[neighborIndex]) {
                            const pIndex = neighborIndex * 4;
                            const r = pixels[pIndex];
                            const g = pixels[pIndex + 1];
                            const b = pixels[pIndex + 2];

                            if (colorDistance([r, g, b], targetRgb) <= tolerance) {
                                visited[neighborIndex] = 1;
                                queue.push({ x: nx, y: ny });
                            }
                        }
                    }
                };

                checkNeighbor(x + 1, y);
                checkNeighbor(x - 1, y);
                checkNeighbor(x, y + 1);
                checkNeighbor(x, y - 1);
            }

            outputCtx.putImageData(maskImageData, 0, 0);
            resolve(outputCtx.canvas.toDataURL('image/png'));
        };
        img.onerror = () => {
            console.error("Failed to load image for contiguous color mask generation.");
            resolve(null);
        };
        img.src = originalImageSrc;
    });
};

/**
 * Generates a mask for all pixels of a selected color globally within a tolerance.
 * @param {CanvasRenderingContext2D} originalCtx - Context of the original image.
 * @param {string} originalImageSrc - Data URL of the original image.
 * @param {string} targetColorRgbaString - The RGBA string of the color to match.
 * @param {number} tolerance - The color tolerance (0-255).
 * @param {CanvasRenderingContext2D} outputCtx - Hidden canvas context for generating the mask.
 * @returns {Promise<string|null>} - Data URL of the generated mask.
 */
export const generateGlobalColorMask = async (originalCtx, originalImageSrc, targetColorRgbaString, tolerance, outputCtx) => {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvasWidth = img.width;
            const canvasHeight = img.height;

            originalCtx.canvas.width = canvasWidth;
            originalCtx.canvas.height = canvasHeight;
            originalCtx.drawImage(img, 0, 0);

            outputCtx.canvas.width = canvasWidth;
            outputCtx.canvas.height = canvasHeight;
            clearCanvas(outputCtx); // Clear output canvas for the new mask

            const imageData = originalCtx.getImageData(0, 0, canvasWidth, canvasHeight);
            const pixels = imageData.data;
            const maskImageData = outputCtx.createImageData(canvasWidth, canvasHeight);
            const maskPixels = maskImageData.data;

            const targetRgb = rgbaStringToRgbArray(targetColorRgbaString);

            for (let i = 0; i < pixels.length; i += 4) {
                const r = pixels[i];
                const g = pixels[i + 1];
                const b = pixels[i + 2];

                if (colorDistance([r, g, b], targetRgb) <= tolerance) {
                    maskPixels[i] = 255;
                    maskPixels[i + 1] = 255;
                    maskPixels[i + 2] = 255;
                    maskPixels[i + 3] = 255; // Full opacity
                } else {
                    maskPixels[i] = 0;
                    maskPixels[i + 1] = 0;
                    maskPixels[i + 2] = 0;
                    maskPixels[i + 3] = 255; // Full opacity
                }
            }

            outputCtx.putImageData(maskImageData, 0, 0);
            resolve(outputCtx.canvas.toDataURL('image/png'));
        };
        img.onerror = () => {
            console.error("Failed to load image for global color mask generation.");
            resolve(null);
        };
        img.src = originalImageSrc;
    });
};


/**
 * Draws selection handles around a shape for editing.
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 * @param {Object} shape - The shape object.
 */
export const drawSelectionHandles = (ctx, shape) => {
    ctx.save();
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]); // Dashed line

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    if (shape.type === 'rectangle') {
        minX = shape.x;
        minY = shape.y;
        maxX = shape.x + shape.width;
        maxY = shape.y + shape.height;
    } else if (shape.type === 'circle') {
        minX = shape.x - shape.radius;
        minY = shape.y - shape.radius;
        maxX = shape.x + shape.radius;
        maxY = shape.y + shape.radius;
    } else if (shape.points && shape.points.length > 0) {
        shape.points.forEach(p => {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        });
    }

    const padding = 5; // Small padding around the bounding box
    ctx.strokeRect(minX - padding, minY - padding, (maxX - minX) + 2 * padding, (maxY - minY) + 2 * padding);

    // Draw resize handles (small squares at corners/midpoints)
    const handleSize = 8;
    const handles = [
        { x: minX - padding, y: minY - padding }, // Top-left
        { x: maxX + padding - handleSize, y: minY - padding }, // Top-right
        { x: minX - padding, y: maxY + padding - handleSize }, // Bottom-left
        { x: maxX + padding - handleSize, y: maxY + padding - handleSize }, // Bottom-right
        // Add mid-point handles for more control
        { x: minX - padding + (maxX - minX + 2 * padding) / 2 - handleSize / 2, y: minY - padding }, // Top-middle
        { x: minX - padding + (maxX - minX + 2 * padding) / 2 - handleSize / 2, y: maxY + padding - handleSize }, // Bottom-middle
        { x: minX - padding, y: minY - padding + (maxY - minY + 2 * padding) / 2 - handleSize / 2 }, // Left-middle
        { x: maxX + padding - handleSize, y: minY - padding + (maxY - minY + 2 * padding) / 2 - handleSize / 2 }, // Right-middle
    ];

    handles.forEach(handle => {
        ctx.fillRect(handle.x, handle.y, handleSize, handleSize);
    });

    ctx.restore();
};


/**
 * Generates a combined mask from all drawn shapes and an optional color-picked mask.
 * This function now returns a Promise, ensuring all asynchronous image loading
 * for 'color_mask' shapes is complete before returning the final Data URL.
 *
 * @param {Array<Object>} shapes - Array of shape objects (brush, rectangle, circle, free_shape, color_mask).
 * @param {number} canvasWidth - The width of the target canvas.
 * @param {number} canvasHeight - The height of the target canvas.
 * @param {CanvasRenderingContext2D} outputCtx - The context of a hidden canvas to draw the mask on.
 * @returns {Promise<string>} A Promise that resolves with the data URL of the generated mask.
 */
export const createMaskFromShapesAndColorMask = (shapes, canvasWidth, canvasHeight, outputCtx) => {
    return new Promise(async (resolve, reject) => {
        outputCtx.canvas.width = canvasWidth;
        outputCtx.canvas.height = canvasHeight;
        clearCanvas(outputCtx);

        if (!Array.isArray(shapes)) {
            console.error(
                "CRITICAL ERROR: createMaskFromShapesAndColorMask received 'shapes' that is not an array.",
                "Type:", typeof shapes, "Value:", shapes
            );
            resolve(outputCtx.canvas.toDataURL('image/png'));
            return;
        }

        const imageLoadPromises = [];

        // First, draw all non-color_mask shapes (these are synchronous)
        shapes.forEach(shape => {
            if (shape.type !== 'color_mask') {
                outputCtx.beginPath();
                outputCtx.fillStyle = 'white'; // Mask areas are white
                outputCtx.strokeStyle = 'white';
                outputCtx.lineWidth = shape.size || 1;

                switch (shape.type) {
                    case 'brush':
                    case 'free_shape':
                        if (shape.points && shape.points.length > 0) {
                            outputCtx.moveTo(shape.points[0].x, shape.points[0].y);
                            shape.points.forEach(p => outputCtx.lineTo(p.x, p.y));
                            if (shape.type === 'free_shape') {
                                outputCtx.closePath();
                                outputCtx.fill();
                            } else {
                                outputCtx.lineCap = 'round';
                                outputCtx.lineJoin = 'round';
                                outputCtx.stroke();
                            }
                        }
                        break;
                    case 'rectangle':
                        outputCtx.rect(shape.x, shape.y, shape.width, shape.height);
                        outputCtx.fill();
                        break;
                    case 'circle':
                        outputCtx.arc(shape.x, shape.y, shape.radius, 0, Math.PI * 2);
                        outputCtx.fill();
                        break;
                    case 'triangle':
                        if (shape.points && shape.points.length >= 3) {
                            outputCtx.moveTo(shape.points[0].x, shape.points[0].y);
                            outputCtx.lineTo(shape.points[1].x, shape.points[1].y);
                            outputCtx.lineTo(shape.points[2].x, shape.points[2].y);
                            outputCtx.closePath();
                            outputCtx.fill();
                        }
                        break;
                    default:
                        console.warn(`Unknown shape type for mask generation: ${shape.type}`);
                }
            }
        });

        // Then, load and draw all color_mask images (these are asynchronous)
        shapes.forEach(shape => {
            if (shape.type === 'color_mask' && shape.dataUrl) {
                const img = new Image();
                const promise = new Promise((imgResolve, imgReject) => {
                    img.onload = () => {
                        // Use globalCompositeOperation to blend the new mask on top of existing shapes
                        // 'source-over' draws the new image on top, 'lighter' could combine white areas
                        // For a mask, 'source-over' is usually fine as white pixels will add to the mask.
                        outputCtx.globalCompositeOperation = 'source-over';
                        outputCtx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
                        imgResolve();
                    };
                    img.onerror = () => {
                        console.error("Failed to load color_mask image for mask generation:", shape.dataUrl);
                        imgReject(new Error("Failed to load color_mask image"));
                    };
                    img.src = shape.dataUrl;
                });
                imageLoadPromises.push(promise);
            }
        });

        // Wait for all color_mask images to load and draw
        try {
            await Promise.all(imageLoadPromises);
            // Now that all images are drawn, get the final Data URL
            resolve(outputCtx.canvas.toDataURL('image/png'));
        } catch (error) {
            console.error("Error loading one or more color_mask images:", error);
            resolve(outputCtx.canvas.toDataURL('image/png')); // Still resolve to avoid hanging
        }
    });
};
