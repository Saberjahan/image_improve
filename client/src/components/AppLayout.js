// src/components/AppLayout.js

import React, { useCallback, useMemo } from "react"; // Import useMemo
import {
    Brush,
    Upload,
    Sparkles,
    PaintBucket, // Used for Pick Color
    SlidersHorizontal, // Used for Auto Detection
    Pencil, // For Free Shape
    Square, Circle, Triangle, Edit, // For Manual Drawing tools
    Wand2, // For Expand Mask & Global Match
    Palette, Aperture, // For Color Selection & Contiguous Match
    Wand, // For AI Repair
    Download, // For Download tools
    Tags // For Generated Keywords display
} from "lucide-react";

/**
 * AppLayout component defines the main structural layout of the Image Repair AI application.
 * It provides containers for the original image canvas, the mask/repaired image canvas,
 * and the control panel, ensuring a consistent and responsive design.
 * It also centralizes all icon imports and renders all UI elements for the control panel.
 *
 * @param {object} props - Component props.
 * @param {React.RefObject<HTMLCanvasElement>} originalCanvasRef - Ref for the original image canvas.
 * @param {React.RefObject<HTMLCanvasElement>} drawingOverlayCanvasRef - Ref for the drawing overlay canvas.
 * @param {React.RefObject<HTMLCanvasElement>} maskCanvasRef - Ref for the mask/repaired image canvas.
 * @param {React.RefObject<HTMLCanvasElement>} outputCanvasRef - Ref for the hidden output canvas.
 * @param {number} canvasWidth - Calculated width for canvases.
 * @param {number} canvasHeight - Calculated height for canvases.
 * @param {string|null} originalImageSrc - Data URL of the original image.
 * @param {string|null} activeDisplayMaskSrc - Data URL of the currently active mask.
 * @param {string|null} repairedImageSrc - Data URL of the repaired image.
 * @param {boolean} showMaskOnSecondCanvas - True if mask should be shown on the second canvas, false for repaired image.
 * @param {function(boolean): void} props.setShowMaskOnSecondCanvas - Setter for showMaskOnSecondCanvas.
 * @param {function(): void} undo - Function to undo the last action.
 * @param {function(): void} redo - Function to redo the last action.
 * @param {boolean} canUndo - True if undo is possible.
 * @param {boolean} canRedo - True if redo is possible.
 * @param {function(string): void} handleDownload - Function to download the image.
 * @param {function(React.MouseEvent): void} handleMouseDown - Mouse down event handler for drawing.
 * @param {function(React.MouseEvent): void} handleMouseMove - Mouse move event handler for drawing.
 * @param {function(React.MouseEvent): void} handleMouseUp - Mouse up event handler for drawing.
 * @param {function(React.MouseEvent): void} handleMouseLeave - Mouse leave event handler for drawing.
 * @param {function(React.MouseEvent): void} handleDoubleClick - Double click event handler for drawing.
 * @param {boolean} totalLoading - Indicates if any processing is active.
 * @param {object} controlPanelProps - All props needed by ControlPanelLogic, passed directly from App.js.
 */
const AppLayout = ({
    originalCanvasRef,
    drawingOverlayCanvasRef,
    maskCanvasRef,
    outputCanvasRef,
    canvasWidth,
    canvasHeight,
    originalImageSrc,
    activeDisplayMaskSrc,
    repairedImageSrc,
    showMaskOnSecondCanvas,
    setShowMaskOnSecondCanvas,
    undo,
    redo,
    reset,
    canUndo,
    canRedo,
    handleDownload,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleMouseLeave,
    handleDoubleClick,
    totalLoading,
    controlPanelProps
}) => {
    const {
        isLoading,
        isOriginalCanvasReadyForPixels,
        handleImageUpload,
        drawingMode,
        setDrawingMode,
        penSize,
        setPenSize,
        penColor,
        setPenColor,
        freeShapePoints,
        closeFreeShape,
        selectedShapeId,
        // deleteSelectedShape, // Removed: No longer destructured here
        shapes, // Ensure shapes is destructured here
        selectedColor,
        setColorTolerance,
        selectedColorRgbString,
        handleContiguousColorMatch,
        handleGlobalColorMatch,
        detectCorruption,
        detectCorruptionFromMask,
        repairMethod,
        setRepairMethod,
        imageTypeDescription,
        setImageTypeDescription,
        onGenerateKeywords,
        generatedKeywords,
        colorTolerance,
        repairPixels,
    } = controlPanelProps;

    // Define common Tailwind CSS classes for buttons
    const toolButtonBaseClasses = "p-2 rounded-lg flex items-center justify-center transition-all duration-300 text-sm font-medium transform hover:scale-105 shadow-md";
    const toolButtonActiveClasses = "bg-gradient-to-br from-blue-500 to-blue-700 text-white shadow-lg";
    const toolButtonInactiveClasses = "bg-gray-200 text-gray-700 hover:bg-gray-300";
    const toolButtonDisabledClasses = "bg-gray-100 text-gray-400 cursor-not-allowed opacity-60";

    // Determine disabled states based on current application state
    const isDrawingToolsDisabled = (!originalImageSrc || isLoading);
    const isColorSelectionDisabled = (!isOriginalCanvasReadyForPixels || isLoading);
    const isAIDetectionDisabled = (!originalImageSrc || isLoading);
    const isAIRepairDisabled = (!originalImageSrc || !activeDisplayMaskSrc || isLoading);

    const getToolButtonClasses = useCallback((mode, isDisabled) => {
        if (isDisabled) return `${toolButtonBaseClasses} ${toolButtonDisabledClasses}`;
        return `${toolButtonBaseClasses} ${drawingMode === mode ? toolButtonActiveClasses : toolButtonInactiveClasses}`;
    }, [drawingMode, toolButtonBaseClasses, toolButtonActiveClasses, toolButtonInactiveClasses, toolButtonDisabledClasses]);

    // Determine if the currently selected repair method benefits from keyword input
    const isKeywordRelevantMethod = useMemo(() => {
        const keywordMethods = [
            "contextual_attention",
            "gated_convolution",
            "partial_convolution",
            "generative_adversarial_network",
            "transformer_based",
            "stable_diffusion_inpaint",
            "latent_diffusion_inpaint",
            "controlnet_inpaint",
            "llm_guided_inpaint",
            "semantic_inpaint"
        ];
        return keywordMethods.includes(repairMethod);
    }, [repairMethod]);

    // Determine if "Expand Mask (AI)" button should be disabled
    // It should be disabled if:
    // 1. Image is not loaded or currently loading (totalLoading)
    // 2. No shapes have been drawn yet (shapes.length === 0)
    // 3. No active mask is displayed (activeDisplayMaskSrc is null)
    const isExpandMaskAIDisabled = totalLoading || shapes.length === 0 || !activeDisplayMaskSrc;


    return (
        <div className="w-full min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col items-center p-3 sm:p-4 font-sans text-gray-800">
            {/* Title and Description */}
            <header className="w-full max-w-7xl text-center mb-4 bg-gradient-to-r from-blue-600 to-purple-700 p-4 rounded-3xl shadow-2xl text-white mx-auto">
                <h1 className="text-4xl font-extrabold mb-1 tracking-tight drop-shadow-lg">
                    Image Repair AI
                </h1>
                <p className="text-base opacity-90">
                    Intelligently fix imperfections in your images with advanced AI tools.
                </p>
            </header>

            {/* Canvas Section with Buttons and Canvases */}
            <section className="w-full max-w-7xl bg-white rounded-3xl shadow-2xl p-4 border border-blue-300 mb-4 mx-auto">
                {/* This div uses a grid to manage the two columns for buttons and canvases */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full">

                    {/* Left Column: Upload Button + Original Canvas + Undo/Redo */}
                    <div className="flex flex-col items-center">
                        {/* Upload Original Image Button - Positioned above the canvas in its column */}
                        <label htmlFor="file-upload" className="block w-full sm:w-auto text-sm text-gray-700 mb-1 mx-auto">
                            <input
                                id="file-upload"
                                type="file"
                                accept="image/*"
                                onChange={handleImageUpload}
                                className="hidden"
                                disabled={totalLoading}
                            />
                            <span className="py-2 px-5 rounded-full border-0 text-base font-semibold bg-gradient-to-r from-blue-600 to-cyan-600 text-white hover:from-blue-700 hover:to-cyan-700 cursor-pointer shadow-lg flex items-center justify-center transition-all duration-300 transform hover:scale-105 h-9">
                                <Upload size={20} className="mr-2" /> Upload Original Image
                            </span>
                        </label>

                        {/* Original Canvas */}
                        <div className="relative border-4 border-blue-500 rounded-xl overflow-hidden bg-gray-100 flex items-center justify-center"
                             style={{ width: canvasWidth > 0 ? canvasWidth : '100%', height: canvasHeight > 0 ? canvasHeight : 'auto', minHeight: '200px' }}>
                            <canvas
                                ref={originalCanvasRef}
                                width={canvasWidth}
                                height={canvasHeight}
                                className="absolute top-0 left-0 w-full h-full object-contain"
                                style={{ display: originalImageSrc ? 'block' : 'none' }}
                            ></canvas>
                            <canvas
                                ref={drawingOverlayCanvasRef}
                                width={canvasWidth}
                                height={canvasHeight}
                                className="absolute top-0 left-0 w-full h-full z-10 cursor-crosshair"
                                style={{ display: originalImageSrc ? 'block' : 'none' }}
                                onMouseDown={handleMouseDown}
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                                onMouseLeave={handleMouseLeave}
                                onDoubleClick={handleDoubleClick}
                                onTouchStart={handleMouseDown}
                                onTouchMove={handleMouseMove}
                                onTouchEnd={handleMouseUp}
                                onTouchCancel={handleMouseUp}
                            ></canvas>
                            {!originalImageSrc && (
                                <div className="w-full h-full flex items-center justify-center text-gray-500 text-base p-3">
                                    Upload an image to start drawing and masking.
                                </div>
                            )}
                        </div>
                        {/* Undo/Redo/Reset buttons */}
                        <div className="flex flex-wrap justify-center gap-2 mt-2">
                            <button
                                onClick={undo}
                                disabled={!canUndo || totalLoading}
                                className={`px-4 py-1.5 rounded-full font-semibold text-white transition-all duration-300 shadow-md transform hover:scale-105 text-sm
                                            ${(!canUndo || totalLoading) ? 'bg-gray-400 cursor-not-allowed' : 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700'}`}
                            >
                                Undo
                            </button>
                            <button
                                onClick={redo}
                                disabled={!canRedo || totalLoading}
                                className={`px-4 py-1.5 rounded-full font-semibold text-white transition-all duration-300 shadow-md transform hover:scale-105 text-sm
                                            ${(!canRedo || totalLoading) ? 'bg-gray-400 cursor-not-allowed' : 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700'}`}
                            >
                                Redo
                            </button>
                            <button
                                onClick={reset}
                                disabled={!originalImageSrc || totalLoading}
                                className={`px-4 py-1.5 rounded-full font-semibold text-white transition-all duration-300 shadow-md transform hover:scale-105 text-sm
                                            ${(!originalImageSrc || totalLoading) ? 'bg-gray-400 cursor-not-allowed' : 'bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700'}`}
                            >
                                Reset All
                            </button>
                        </div>
                    </div>

                    {/* Right Column: Show Mask/Repaired Buttons + Mask/Repaired Canvas + Download Buttons */}
                    <div className="flex flex-col items-center">
                        {/* Show Mask / Show Repaired Image buttons - Positioned above the canvas in its column */}
                        <div className="flex flex-wrap justify-center gap-2 mb-1 w-full sm:w-auto">
                            <button
                                onClick={() => setShowMaskOnSecondCanvas(true)}
                                disabled={(totalLoading || !activeDisplayMaskSrc)}
                                className={`px-3 py-1.5 rounded-full font-semibold text-white transition-all duration-300 shadow-md transform hover:scale-105 h-9 text-sm
                                            ${showMaskOnSecondCanvas ? 'bg-gradient-to-r from-blue-600 to-blue-800' : 'bg-gray-500 hover:bg-gray-600'}
                                            ${(totalLoading || !activeDisplayMaskSrc) ? 'cursor-not-allowed opacity-60' : ''}`}
                            >
                                Show Mask
                            </button>
                            <button
                                onClick={() => setShowMaskOnSecondCanvas(false)}
                                disabled={(totalLoading || !repairedImageSrc)}
                                className={`px-3 py-1.5 rounded-full font-semibold text-white transition-all duration-300 shadow-md transform hover:scale-105 h-9 text-sm
                                            ${!showMaskOnSecondCanvas ? 'bg-gradient-to-r from-blue-600 to-blue-800' : 'bg-gray-500 hover:bg-gray-600'}
                                            ${(totalLoading || !repairedImageSrc) ? 'cursor-not-allowed opacity-60' : ''}`}
                            >
                                Show Repaired
                            </button>
                        </div>
                        <div className="relative border-4 border-blue-500 rounded-xl overflow-hidden bg-gray-100 flex items-center justify-center"
                             style={{ width: canvasWidth > 0 ? canvasWidth : '100%', height: canvasHeight > 0 ? canvasHeight : 'auto', minHeight: '200px' }}>
                            <canvas
                                ref={maskCanvasRef}
                                width={canvasWidth}
                                height={canvasHeight}
                                className="absolute top-0 left-0 w-full h-full object-contain"
                                style={{ display: (activeDisplayMaskSrc || repairedImageSrc) ? 'block' : 'none' }}
                            ></canvas>
                            {!originalImageSrc && (
                                <div className="w-full h-full flex items-center justify-center text-gray-500 text-base p-3">
                                    Mask or repaired image will appear here after processing.
                                </div>
                            )}
                        </div>
                        {/* Hidden canvas for off-screen processing */}
                        <canvas
                            ref={outputCanvasRef}
                            width={canvasWidth}
                            height={canvasHeight}
                            style={{ display: 'none' }}
                        ></canvas>
                        {/* Download buttons */}
                        <div className="flex flex-wrap justify-center gap-2 mt-2">
                            <button
                                onClick={() => handleDownload('png')}
                                disabled={(!repairedImageSrc && !activeDisplayMaskSrc) || totalLoading}
                                className={`px-5 py-1.5 rounded-full font-semibold text-white transition-all duration-300 shadow-md flex items-center justify-center transform hover:scale-105 text-sm
                                            ${((!repairedImageSrc && !activeDisplayMaskSrc) || totalLoading) ? 'bg-gray-400 cursor-not-allowed' : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700'}`}
                            >
                                <Download size={18} className="mr-2" /> Download PNG
                            </button>
                            <button
                                onClick={() => handleDownload('jpeg')}
                                disabled={(!repairedImageSrc && !activeDisplayMaskSrc) || totalLoading}
                                className={`px-5 py-1.5 rounded-full font-semibold text-white transition-all duration-300 shadow-md flex items-center justify-center transform hover:scale-105 text-sm
                                            ${((!repairedImageSrc && !activeDisplayMaskSrc) || totalLoading) ? 'bg-gray-400 cursor-not-allowed' : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700'}`}
                            >
                                <Download size={18} className="mr-2" /> Download JPG
                            </button>
                        </div>
                    </div>
                </div>
            </section>


            {/* Three-Column Tool Section (3:2:1 ratio) */}
            <section className="w-full max-w-7xl mb-4 mx-auto">
                <div className="p-4 bg-white rounded-3xl shadow-2xl border border-blue-300">
                    <h2 className="text-xl font-bold text-gray-800 mb-4 text-center">Select the Corrupted Area and Pixels</h2>
                    <div className="flex flex-col lg:flex-row gap-2"> {/* Reduced gap */}
                        {/* Manual Selection (3 parts) */}
                        <div className="w-full lg:flex-grow-[3] lg:basis-0 p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl shadow-inner border border-purple-200 flex flex-col justify-between">
                            <div> {/* Wrapper div for top content */}
                                <h2 className="text-xl font-bold text-purple-700 mb-3 text-center">Manual Selection</h2>
                                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-3"> {/* Reduced gap and mb */}
                                    <button
                                        onClick={() => setDrawingMode('brush')}
                                        className={`${getToolButtonClasses('brush', isDrawingToolsDisabled)}`}
                                        disabled={isDrawingToolsDisabled}
                                        title="Brush Tool"
                                    >
                                        <Brush size={18} className="mr-1" /> Brush
                                    </button>
                                    <button
                                        onClick={() => setDrawingMode('free_shape')}
                                        className={`${getToolButtonClasses('free_shape', isDrawingToolsDisabled)}`}
                                        disabled={isDrawingToolsDisabled}
                                        title="Free Shape (Polygon) Tool"
                                    >
                                        <Pencil size={18} className="mr-1" /> Free Shape
                                    </button>
                                    <button
                                        onClick={() => setDrawingMode('rectangle')}
                                        className={`${getToolButtonClasses('rectangle', isDrawingToolsDisabled)}`}
                                        disabled={isDrawingToolsDisabled}
                                        title="Rectangle Tool"
                                    >
                                        <Square size={18} className="mr-1" /> Square
                                    </button>
                                    <button
                                        onClick={() => setDrawingMode('circle')}
                                        className={`${getToolButtonClasses('circle', isDrawingToolsDisabled)}`}
                                        disabled={isDrawingToolsDisabled}
                                        title="Circle Tool"
                                    >
                                        <Circle size={18} className="mr-1" /> Circle
                                    </button>
                                    <button
                                        onClick={() => setDrawingMode('triangle')}
                                        className={`${getToolButtonClasses('triangle', isDrawingToolsDisabled)}`}
                                        disabled={isDrawingToolsDisabled}
                                        title="Triangle Tool"
                                    >
                                        <Triangle size={18} className="mr-1" /> Triangle
                                    </button>
                                    <button
                                        onClick={() => setDrawingMode('edit_shape')}
                                        className={`${getToolButtonClasses('edit_shape', (isDrawingToolsDisabled || shapes.length === 0))}`}
                                        disabled={(isDrawingToolsDisabled || shapes.length === 0)}
                                        title="Edit/Move/Resize Shapes"
                                    >
                                        <Edit size={18} className="mr-1" /> Edit Shape
                                    </button>
                                </div>
                                {/* Manual Controls - Pen Size/Color, Close/Delete Shape */}
                                <div className="flex items-center gap-1 p-2 bg-blue-50 rounded-lg shadow-inner border border-blue-100">
                                    <div className="flex-grow-[5] flex items-center gap-2"> {/* Added flex and items-center */}
                                        <label htmlFor="pen-size" className="text-sm font-medium text-gray-700 whitespace-nowrap">Pen Size: {penSize}px</label>
                                        <input
                                            type="range"
                                            id="pen-size"
                                            min="1"
                                            max="50"
                                            value={penSize}
                                            onChange={(e) => setPenSize(Number(e.target.value))}
                                            className="flex-grow h-2 bg-gradient-to-r from-blue-400 to-blue-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                            disabled={isDrawingToolsDisabled}
                                        />
                                    </div>
                                    <div className="flex-grow-[1] flex items-center gap-2"> {/* Added flex and items-center */}
                                        <label htmlFor="pen-color" className="text-sm font-medium text-gray-700 whitespace-nowrap">Color:</label>
                                        <input
                                            type="color"
                                            id="pen-color"
                                            value={penColor}
                                            onChange={(e) => setPenColor(e.target.value)}
                                            className="w-full h-7 rounded-lg cursor-pointer border-none shadow-inner"
                                            disabled={isDrawingToolsDisabled}
                                        />
                                    </div>
                                </div>
                                {drawingMode === 'free_shape' && freeShapePoints.length > 0 && (
                                    <button
                                        onClick={closeFreeShape}
                                        className={`${toolButtonBaseClasses} bg-gradient-to-r from-green-500 to-lime-600 text-white hover:from-green-600 hover:to-lime-700 w-full mt-0.5`}
                                        disabled={freeShapePoints.length < 3 || isDrawingToolsDisabled}
                                        title="Close Current Free Shape"
                                    >
                                        Close Shape ({freeShapePoints.length} Points)
                                    </button>
                                )}
                                {/* The "Delete Selected Shape" button was removed from here. */}
                            </div> {/* Closes wrapper div for top content */}
                            {/* The "Expand Mask (AI)" button has been moved to the new section below */}
                        </div>

                        {/* Color-Based Selection (2 parts) */}
                        <div className="w-full lg:flex-grow-[2] lg:basis-0 p-4 bg-gradient-to-br from-yellow-50 to-orange-50 rounded-xl shadow-inner border border-yellow-200">
                            <h2 className="text-xl font-bold text-orange-700 mb-3 text-center">Color-Based Selection</h2>
                            <div className="grid grid-cols-1 gap-3 mb-4"> {/* Reduced gap and mb */}
                                <button
                                    onClick={() => setDrawingMode('color_select')}
                                    className={getToolButtonClasses('color_select', isColorSelectionDisabled)}
                                    disabled={isColorSelectionDisabled}
                                    title="Pick a color from the image"
                                >
                                    <PaintBucket size={18} className="mr-1" /> Pick Color
                                </button>

                                {/* Moved Tolerance and Last Picked Color here */}
                                <div className="flex flex-col text-center p-2 bg-yellow-100 rounded-lg shadow-inner border border-yellow-200">
                                    <label htmlFor="color-tolerance" className="block text-sm font-medium text-gray-700 mb-1">
                                        Tolerance: {colorTolerance}
                                    </label>
                                    <input
                                        type="range"
                                        id="color-tolerance"
                                        min="0"
                                        max="255"
                                        value={colorTolerance}
                                        onChange={(e) => setColorTolerance(Number(e.target.value))}
                                        className="w-full h-2 bg-gradient-to-r from-indigo-400 to-purple-600 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                                        disabled={(isColorSelectionDisabled || !selectedColor)}
                                    />
                                    {selectedColor && (
                                        <div className="w-full text-sm text-gray-700 flex items-center justify-center text-center mt-2">
                                            Last Picked Color:
                                            <span
                                                className="inline-block w-5 h-5 rounded-full ml-2 border border-gray-300 shadow-sm"
                                                style={{ backgroundColor: selectedColorRgbString }}
                                            ></span>
                                            <span className="ml-2 font-mono">{selectedColorRgbString}</span>
                                        </div>
                                    )}
                                </div>

                                <button
                                    onClick={handleContiguousColorMatch}
                                    className={`${toolButtonBaseClasses} py-2 ${(!isColorSelectionDisabled && selectedColor) ? 'bg-gradient-to-r from-indigo-600 to-purple-700 text-white hover:from-indigo-700 hover:to-purple-800 shadow-lg' : toolButtonDisabledClasses}`}
                                    disabled={(isColorSelectionDisabled || !selectedColor)}
                                    title="Match contiguous pixels of picked color"
                                >
                                    <Aperture size={18} className="mr-1" /> Contiguous Match
                                </button>
                                <button
                                    onClick={handleGlobalColorMatch}
                                    className={`${toolButtonBaseClasses} py-2 ${(!isColorSelectionDisabled && selectedColor) ? 'bg-gradient-to-r from-indigo-600 to-purple-700 text-white hover:from-indigo-700 hover:to-purple-800 shadow-lg' : toolButtonDisabledClasses}`}
                                    disabled={(isColorSelectionDisabled || !selectedColor)}
                                    title="Match all pixels of picked color globally"
                                >
                                    <Wand2 size={18} className="mr-1" /> Global Match
                                </button>
                            </div>
                        </div>

                        {/* Auto Selection (1 part) */}
                        <div className="w-full lg:flex-grow-[1] lg:basis-0 p-4 bg-gradient-to-br from-green-50 to-teal-50 rounded-xl shadow-inner border border-green-200">
                            <h2 className="text-xl font-bold text-teal-700 mb-3 text-center">Auto Selection</h2>
                            <div className="grid grid-cols-1 gap-3 mb-4"> {/* Reduced gap and mb */}
                                <button
                                    onClick={detectCorruption}
                                    className={`${toolButtonBaseClasses} h-32 py-3 ${!isAIDetectionDisabled ? 'bg-gradient-to-r from-green-600 to-teal-600 text-white hover:from-green-700 hover:to-teal-700 shadow-lg' : toolButtonDisabledClasses} flex-col`}
                                    disabled={isAIDetectionDisabled}
                                    title="AI detects corruption across the entire image"
                                >
                                    <Sparkles size={32} className="mb-1" />
                                    <span className="text-sm">Auto-Mask (AI)</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* New Section with One Wide Cell */}
            <section className="w-full max-w-7xl mb-4 mx-auto">
                <div className="p-4 bg-white rounded-3xl shadow-2xl border border-blue-300">
                    {/* Single Wide Cell */}
                    <div className="w-full p-4 bg-gradient-to-br from-slate-50 to-gray-50 rounded-xl shadow-inner border border-slate-200">
                        {/* Expand Mask (AI) button moved here */}
                        <button
                            onClick={detectCorruptionFromMask}
                            className={`${toolButtonBaseClasses} w-full py-2 ${isExpandMaskAIDisabled ? toolButtonDisabledClasses : 'bg-gradient-to-r from-purple-600 to-pink-700 text-white hover:from-purple-700 hover:to-pink-800 shadow-lg'}`}
                            disabled={isExpandMaskAIDisabled}
                            title="AI refines detection within your current mask"
                        >
                            <Wand2 size={18} className="mr-1" /> (Optional) Adjust Mask (AI)
                        </button>
                    </div>
                </div>
            </section>

            {/* AI Repair Section (Wide Cell) */}
            <section className="w-full max-w-7xl bg-gradient-to-br from-red-50 to-orange-50 rounded-xl shadow-inner border border-red-200 p-4 mb-4 mx-auto">
                <h2 className="text-xl font-bold text-red-700 mb-3 text-center">AI Repair Controls</h2>
                <div className="flex flex-col sm:flex-row gap-3 items-center w-full mb-4"> {/* Reduced gap and mb */}
                    <label htmlFor="repair-method" className="block text-sm font-medium text-gray-700 whitespace-nowrap">Repair Method:</label>
                    <select
                        id="repair-method"
                        value={repairMethod}
                        onChange={(e) => setRepairMethod(e.target.value)}
                        className="flex-1 block w-full p-2 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm bg-gray-50 appearance-none pr-8 cursor-pointer"
                        disabled={isAIRepairDisabled}
                        title="Select AI repair algorithm"
                    >
                        <optgroup label="Classical Inpainting Methods">
                            <option value="telea">Telea (Fast Marching)</option>
                            <option value="navier_stokes">Navier-Stokes</option>
                            <option value="exemplar_based">Exemplar-Based (Criminisi)</option>
                            <option value="non_local_means">Non-Local Means</option>
                        </optgroup>
                        <optgroup label="Deep Learning (CNN/GAN) Inpainting">
                            <option value="contextual_attention">Contextual Attention</option>
                            <option value="gated_convolution">Gated Convolution (Heavy/Slow)</option>
                            <option value="partial_convolution">Partial Convolution</option>
                            <option value="generative_adversarial_network">Generative Adversarial Network (GAN)</option>
                            <option value="transformer_based">Transformer-Based</option>
                        </optgroup>
                        <optgroup label="Generative Diffusion Models">
                            <option value="stable_diffusion_inpaint">Stable Diffusion Inpaint</option>
                            <option value="latent_diffusion_inpaint">Latent Diffusion Inpaint</option>
                            <option value="controlnet_inpaint">ControlNet Inpaint</option>
                        </optgroup>
                        <optgroup label="Semantic & Advanced Inpainting">
                            <option value="llm_guided_inpaint">LLM-Guided Inpainting</option>
                            <option value="semantic_inpaint">Semantic Inpainting</option>
                        </optgroup>
                    </select>
                </div>
                <div className={`flex flex-col sm:flex-row gap-3 items-center w-full mb-4 transition-opacity duration-300 ${(!isKeywordRelevantMethod || isLoading) ? 'opacity-50 pointer-events-none' : ''}`}>
                    <label htmlFor="imageTypeDescription" className="block text-sm font-medium text-gray-700 whitespace-nowrap">
                        Image Type:
                    </label>
                    <input
                        type="text"
                        id="imageTypeDescription"
                        value={imageTypeDescription}
                        onChange={(e) => setImageTypeDescription(e.target.value)}
                        placeholder="a white-black dog playing in a park full of people and maple trees."
                        disabled={isLoading || !isKeywordRelevantMethod} // Disabled if not relevant or loading
                        className={`flex-1 block w-full p-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-sm
                                            ${(isLoading || !isKeywordRelevantMethod) ? 'opacity-50 cursor-not-allowed' : ''}`}
                    />
                    <button
                        onClick={() => onGenerateKeywords(imageTypeDescription)}
                        disabled={(isLoading || !imageTypeDescription.trim() || !isKeywordRelevantMethod)} // Disabled if not relevant, loading, or empty
                        className={`${toolButtonBaseClasses} px-3 py-2 text-sm ${(!isLoading && imageTypeDescription.trim() && isKeywordRelevantMethod) ? 'bg-gradient-to-r from-gray-700 to-gray-800 text-white hover:from-gray-800 hover:to-gray-900' : toolButtonDisabledClasses}`}
                        title="Generate keywords from the description for AI context"
                    >
                        Generate Keywords
                    </button>
                </div>

                {/* New: Display Generated Keywords */}
                {generatedKeywords && generatedKeywords.trim() && (
                    <div className={`flex items-start p-3 mt-2 bg-blue-50 rounded-lg shadow-inner border border-blue-200 text-sm text-gray-700 transition-opacity duration-300 ${(!isKeywordRelevantMethod || isLoading) ? 'opacity-50' : ''}`}>
                        <Tags size={18} className="text-blue-600 mr-2 flex-shrink-0 mt-0.5" />
                        <span className="font-medium text-blue-800">Generated Keywords: </span>
                        <span className="ml-1 flex-grow break-words">{generatedKeywords}</span>
                    </div>
                )}

                <button
                    onClick={() => repairPixels(repairMethod, imageTypeDescription)}
                    disabled={isAIRepairDisabled}
                    className={`${toolButtonBaseClasses} w-full py-2 mt-4 ${!isAIRepairDisabled ? 'bg-gradient-to-r from-teal-500 to-emerald-600 text-white hover:from-teal-600 hover:to-emerald-700 shadow-lg' : toolButtonDisabledClasses}`}
                    title="Apply selected AI repair method to the masked areas"
                >
                    <Wand size={20} className="mr-2" /> Repair Pixels
                </button>
            </section>

            {/* Loading Overlay */}
            {totalLoading && (
                <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 backdrop-blur-sm">
                    <div className="bg-white p-8 rounded-2xl shadow-3xl flex flex-col items-center">
                        <div className="animate-spin rounded-full h-14 w-14 border-t-4 border-b-4 border-blue-600 mb-4"></div>
                        <span className="text-gray-800 text-xl font-semibold">Processing... Please wait.</span>
                        <p className="text-gray-600 text-xs mt-1">This might take a moment.</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AppLayout;