// src/features/help/HelpContent.js
import React from 'react';

/**
 * Help component provides an overview of the Image Repair AI application
 * and explains the functionality of its various tools and processes.
 */
const HelpContent = () => { // Renamed from Help to HelpContent
  return (
    <div className="p-6 bg-white rounded-lg shadow-xl max-w-3xl mx-auto my-8 font-sans text-gray-800">
      <h2 className="text-3xl font-bold text-gray-900 mb-6 border-b-2 border-blue-300 pb-2">
        Image Repair AI: How It Works
      </h2>

      <section className="mb-6">
        <h3 className="text-2xl font-semibold text-blue-700 mb-3">Project Overview</h3>
        <p className="text-lg leading-relaxed">
          The Image Repair AI application is a powerful tool designed to help you identify and repair imperfections in images using both manual drawing tools and advanced AI capabilities. It provides a visual interface where you can precisely define areas for repair and then apply various AI algorithms to intelligently fill or correct those regions.
        </p>
      </section>

      <section className="mb-6">
        <h3 className="text-2xl font-semibold text-blue-700 mb-3">Application Layout</h3>
        <p className="mb-4 leading-relaxed">
          The application is divided into three main sections:
        </p>
        <ul className="list-disc list-inside ml-4 space-y-2">
          <li>
            <strong className="font-medium text-gray-900">Original Image Canvas (Left Panel):</strong> This is where you upload your image and interact with it using the drawing and selection tools. Your drawn shapes will appear as outlines on this canvas.
          </li>
          <li>
            <strong className="font-medium text-gray-900">Mask/Repaired Image Canvas (Right Panel):</strong> This canvas dynamically displays either the generated mask (showing the areas selected for repair) or the final AI-repaired image. You can toggle between these views.
          </li>
          <li>
            <strong className="font-medium text-gray-900">Control Panel (Bottom Section):</strong> This panel houses all the tools, settings, and AI processing options, organized into logical groups.
          </li>
        </ul>
      </section>

      <section className="mb-6">
        <h3 className="text-2xl font-semibold text-blue-700 mb-3">Core Processes</h3>
        <ul className="list-decimal list-inside ml-4 space-y-3">
          <li>
            <strong className="font-medium text-gray-900">Image Upload:</strong> Start by clicking "Choose Image" to load an image into the left canvas.
          </li>
          <li>
            <strong className="font-medium text-gray-900">Mask Creation:</strong> Use the drawing tools or AI detection features to create a "mask" on the image. The mask defines the areas you want the AI to repair. On the mask canvas, these areas will appear as filled white regions.
          </li>
          <li>
            <strong className="font-medium text-gray-900">AI Repair:</strong> Select an AI repair method and click "Repair Pixels." The AI will process the original image using the generated mask to fill in the masked areas.
          </li>
          <li>
            <strong className="font-medium text-gray-900">Download:</strong> Save your repaired image or the generated mask.
          </li>
        </ul>
      </section>

      <section className="mb-6">
        <h3 className="text-2xl font-semibold text-blue-700 mb-3">Control Panel Buttons & Features</h3>

        <div className="mb-4">
          <h4 className="text-xl font-semibold text-gray-800 mb-2">Image Handling</h4>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>
              <strong className="font-medium text-gray-900">Choose Image:</strong> Upload an image file (PNG, JPG, etc.) from your device to begin.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Download PNG / Download JPG:</strong> Save the image currently displayed on the right canvas (either the mask or the repaired image) to your computer.
            </li>
          </ul>
        </div>

        <div className="mb-4">
          <h4 className="text-xl font-semibold text-gray-800 mb-2">Drawing Tools</h4>
          <p className="mb-2 leading-relaxed">These tools allow you to manually draw shapes on the original image to define mask areas. Drawn shapes appear as outlines on the left canvas and as filled areas on the mask canvas.</p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>
              <strong className="font-medium text-gray-900">Brush (<span className="font-icon">🖌️</span>):</strong> Draw freehand lines. Click and drag.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Rectangle (<span className="font-icon">⬛</span>):</strong> Draw rectangular shapes. Click, drag, and release.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Circle (<span className="font-icon">⚪</span>):</strong> Draw circular shapes. Click, drag (sets radius), and release.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Triangle (<span className="font-icon">🔺</span>):</strong> Draw triangular shapes. Click for the first point, drag for the second, and release for the third.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Free Shape (<span className="font-icon">✍️</span>):</strong> Draw custom polygons. Click for each vertex, then double-click or click the "Close Free Shape" button to finalize.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Edit Shape (<span className="font-icon">✏️</span>):</strong> Select and manipulate existing shapes. Click on a shape to select it. Drag it to move, or drag its handles to resize.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Pen Size:</strong> Adjust the thickness of the drawing tools.
            </li>
          </ul>
        </div>

        <div className="mb-4">
          <h4 className="text-xl font-semibold text-gray-800 mb-2">Color Selection & Masking</h4>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>
              <strong className="font-medium text-gray-900">Eyedropper (<span className="font-icon">💧</span>):</strong> Select a color directly from the image. Click on any pixel on the original image canvas to pick its color.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Color Tolerance:</strong> Adjust how strictly the color matching should be for contiguous and global selections. A higher tolerance selects a wider range of similar colors.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Contiguous Color Match (<span className="font-icon">✨</span>):</strong> After picking a color, click this button to automatically select all connected pixels of similar color (within the set tolerance) to create a mask.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Global Color Match (<span className="font-icon">🌟</span>):</strong> After picking a color, click this button to automatically select *all* pixels of similar color (within the set tolerance) across the entire image to create a mask, regardless of connectivity.
            </li>
          </ul>
        </div>

        <div className="mb-4">
          <h4 className="text-xl font-semibold text-gray-800 mb-2">AI Processing</h4>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>
              <strong className="font-medium text-gray-900">AI Detect (<span className="font-icon">🔍</span>):</strong> Automatically detect "corrupted" or damaged areas in the image and generate a mask. (Simulated functionality)
            </li>
            <li>
              <strong className="font-medium text-gray-900">AI Expand Mask (<span className="font-icon">↔️</span>):</strong> Expand the current mask to include surrounding areas that might also need repair, useful for refining automatically detected masks or existing drawn masks. (Simulated functionality)
            </li>
            <li>
              <strong className="font-medium text-gray-900">Repair Method:</strong> Select the AI algorithm to use for repairing the masked areas. Options include:
              <ul className="list-circle list-inside ml-4 mt-1 space-y-0.5 text-sm">
                <li>Inpaint (Telea)</li>
                <li>Inpaint (Navier-Stokes)</li>
                <li>Contextual Attention</li>
                <li>Gated Convolution</li>
                <li>Partial Convolution</li>
                <li>Generative Adversarial Network</li>
                <li>Transformer-Based Inpaint</li>
                <li>Stable Diffusion Inpaint</li>
                <li>Latent Diffusion Inpaint</li>
                <li>LLM-Guided Inpaint</li>
                <li>Semantic Inpaint</li>
              </ul>
            </li>
            <li>
              <strong className="font-medium text-gray-900">Image Type:</strong> Provide a brief description (e.g., "dog," "building," "subsurface acoustic impedance") to help the AI understand the image content for more accurate repairs, especially with LLM-Guided methods. Click "Generate Keywords" to get AI-suggested keywords based on your description.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Repair Pixels (<span className="font-icon">🪄</span>):</strong> Apply the selected AI repair method to the areas defined by the current mask. The repaired image will appear on the right canvas.
            </li>
          </ul>
        </div>

        <div className="mb-4">
          <h4 className="text-xl font-semibold text-gray-800 mb-2">History Controls</h4>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>
              <strong className="font-medium text-gray-900">Undo:</strong> Revert the last drawing or masking action.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Redo:</strong> Reapply the last undone action.
            </li>
            <li>
              <strong className="font-medium text-gray-900">Reset:</strong> Clear all drawn shapes, masks, and history, returning the application to its initial state.
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h3 className="text-2xl font-semibold text-blue-700 mb-3">Tips for Usage</h3>
        <ul className="list-disc list-inside ml-4 space-y-2">
          <li>For precise repairs, use the drawing tools to carefully outline the areas you want to fix.</li>
          <li>Experiment with different AI Repair Methods to see which one yields the best results for your specific image and type of corruption.</li>
          <li>Use the "Image Type" description to provide context to the AI, which can significantly improve repair quality for advanced models.</li>
          <li>Remember that the left canvas shows outlines for drawing, while the right canvas shows the filled mask for processing.</li>
        </ul>
      </section>
    </div>
  );
};

export default HelpContent;
