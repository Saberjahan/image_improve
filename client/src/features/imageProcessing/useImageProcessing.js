// src/features/imageProcessing/useImageProcessing.js
import { useState, useCallback } from 'react';
import { clearCanvas, drawImageOnCanvas } from '../drawing/canvasUtils';
import { processImage } from './imageApiUtils'; // Import the API utility function

/**
 * Custom hook for handling image processing tasks like corruption detection and pixel repair.
 * It interacts with a hypothetical AI backend (simulated here) and manages related states.
 *
 * @param {string|null} originalImageSrc - The Data URL of the original image.
 * @param {string|null} activeDisplayMaskSrc - The Data URL of the currently active mask.
 * @param {function(string|null): void} setColorPickedMaskDataUrl - Setter for the color-picked mask data URL.
 * @param {function(Array<Object>): void} setShapes - Setter for the shapes array.
 * @param {React.RefObject<HTMLCanvasElement>} originalCanvasRef - Ref to the original image canvas.
 * @param {React.MutableRefObject<boolean>} shouldSaveHistoryRef - Ref to signal when a new state should be saved.
 * @param {function(boolean): void} setIsImageProcessingLoading - Setter for image processing loading state.
 * @returns {{
 * repairedImageSrc: string|null,
 * setRepairedImageSrc: function,
 * showMaskOnSecondCanvas: boolean,
 * setShowMaskOnSecondCanvas: function,
 * detectCorruption: function,
 * detectCorruptionFromMask: function,
 * repairPixels: function,
 * onGenerateKeywords: function
 * }}
 */
const useImageProcessing = (
    originalImageSrc,
    activeDisplayMaskSrc,
    setColorPickedMaskDataUrl,
    setShapes,
    originalCanvasRef,
    shouldSaveHistoryRef,
    setIsImageProcessingLoading
) => {
    const [repairedImageSrc, setRepairedImageSrc] = useState(null);
    const [showMaskOnSecondCanvas, setShowMaskOnSecondCanvas] = useState(true);

    /**
     * Calls the backend to detect corruption on the full image and returns a mask.
     */
    const detectCorruption = useCallback(async () => {
        if (!originalImageSrc) {
            console.warn("No original image to detect corruption.");
            return;
        }
        setIsImageProcessingLoading(true);
        console.log("Calling backend for 'AI DetectFull Image'...");

        try {
            const newMaskDataUrl = await processImage(
                'detect', // The new process_type for full image detection
                originalImageSrc,
                null // No mask is sent for full image detection
            );

            if (newMaskDataUrl) {
                setShapes(prevShapes => {
                    const safePrevShapes = Array.isArray(prevShapes) ? prevShapes : [];
                    // Filter out any existing 'color_mask' shapes before adding the new one
                    const filteredShapes = safePrevShapes.filter(s => s && s.type !== 'color_mask');
                    return [...filteredShapes, {
                        id: `auto_mask_${Date.now()}`,
                        type: 'color_mask',
                        dataUrl: newMaskDataUrl,
                        x: 0, y: 0,
                        width: originalCanvasRef.current.width,
                        height: originalCanvasRef.current.height
                    }];
                });
                shouldSaveHistoryRef.current = true;
                setShowMaskOnSecondCanvas(true);
                console.log("Full image detection successful, mask received.");
            } else {
                console.warn("Backend did not return a processed image for full image detection.");
            }
        } catch (error) {
            console.error("Error during 'AI DetectFull Image' API call:", error);
        } finally {
            setIsImageProcessingLoading(false);
        }
    }, [originalImageSrc, setShapes, originalCanvasRef, shouldSaveHistoryRef, setIsImageProcessingLoading]);


    /**
     * Calls the backend to detect and expand an existing mask based on image content.
     */
    const detectCorruptionFromMask = useCallback(async () => {
        if (!activeDisplayMaskSrc || !originalImageSrc) {
            console.warn("Cannot expand mask: Original image or active mask is missing.");
            return;
        }

        setIsImageProcessingLoading(true);
        console.log("Calling backend for 'AI Adjust Mask'...");

        try {
            const newMaskDataUrl = await processImage(
                'detect_from_mask',
                originalImageSrc,
                activeDisplayMaskSrc
            );

            if (newMaskDataUrl) {
                // Clear any existing color-picked masks, as this is a new, full mask
                setColorPickedMaskDataUrl(null);

                // This is an expanded mask, so replace the old one
                setShapes(prevShapes => {
                    const safePrevShapes = Array.isArray(prevShapes) ? prevShapes : [];
                    return [...safePrevShapes.filter(s => s != null && s.type !== 'color_mask'), {
                        id: `expanded_mask_${Date.now()}`,
                        type: 'color_mask',
                        dataUrl: newMaskDataUrl,
                        x: 0, y: 0,
                        width: originalCanvasRef.current.width,
                        height: originalCanvasRef.current.height
                    }];
                });
                shouldSaveHistoryRef.current = true;
                setShowMaskOnSecondCanvas(true);
                console.log("Mask expansion successful, new mask received.");
            } else {
                console.warn("Backend did not return a processed image for mask expansion.");
            }
        } catch (error) {
            console.error("Error during 'AI Adjust Mask' API call:", error);
        } finally {
            setIsImageProcessingLoading(false);
        }
    }, [activeDisplayMaskSrc, originalImageSrc, setColorPickedMaskDataUrl, setShapes, originalCanvasRef, shouldSaveHistoryRef, setIsImageProcessingLoading]);


    const repairPixels = useCallback(async (method, description) => {
        if (!activeDisplayMaskSrc || !originalImageSrc) {
            console.warn("Cannot repair pixels: Original image or active mask is missing.");
            return;
        }

        setIsImageProcessingLoading(true);
        console.log("Calling backend for pixel repair with method:", method);

        try {
            const repairedImage = await processImage(
                method,
                originalImageSrc,
                activeDisplayMaskSrc,
                description
            );

            if (repairedImage) {
                setRepairedImageSrc(repairedImage);
                setShowMaskOnSecondCanvas(false);
                console.log("Pixel repair successful, repaired image received.");
            } else {
                console.warn("Backend did not return a repaired image.");
            }
        } catch (error) {
            console.error("Error during pixel repair API call:", error);
            // Optionally, clear the repaired image on error
            setRepairedImageSrc(null);
        } finally {
            setIsImageProcessingLoading(false);
        }
    }, [activeDisplayMaskSrc, originalImageSrc, setRepairedImageSrc, setIsImageProcessingLoading]);

    const onGenerateKeywords = useCallback(async (description) => {
        setIsImageProcessingLoading(true);
        try {
            // Await the fetch call with exponential backoff for the LLM API
            const apiKey = "";
            let chatHistory = [];
            chatHistory.push({ role: "user", parts: [{ text: `Generate a comma-separated list of keywords for the following description, without any introductory phrases or explanations: ${description}` }] });
            const payload = {
                contents: chatHistory
            };
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API call failed: ${response.status} ${response.statusText} - ${errorText}`);
            }

            const result = await response.json();
            const generatedText = result?.candidates?.[0]?.content?.parts?.[0]?.text;

            if (generatedText) {
                console.log("Generated Keywords:", generatedText);
                // For now, we'll just log them.
            } else {
                console.warn("LLM did not return any keywords.");
            }
        } catch (error) {
            console.error("Error generating keywords with LLM:", error);
        } finally {
            setIsImageProcessingLoading(false);
        }
    }, [setIsImageProcessingLoading]);


    return {
        repairedImageSrc,
        setRepairedImageSrc,
        showMaskOnSecondCanvas,
        setShowMaskOnSecondCanvas,
        detectCorruption,
        detectCorruptionFromMask,
        repairPixels,
        onGenerateKeywords,
    };
};

export default useImageProcessing;
