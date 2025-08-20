// src/features/imageProcessing/imageApiUtils.js

const API_BASE_URL = 'http://localhost:3001/api'; // Hardcoded for offline simplicity

/**
 * Makes a POST request to the backend image processing API.
 * @param {string} processType - The type of processing to perform (e.g., 'detect', 'detect_from_mask', 'inpaint').
 * @param {string} imageDataB64 - Base64 encoded original image data.
 * @param {string|null} [maskDataB64=null] - Optional Base64 encoded mask data.
 * @param {string} [imageTypeDescription=''] - Optional description for AI context.
 * @returns {Promise<string>} A promise that resolves with the base64 encoded processed image data URL.
 * @throws {Error} If the API call fails or returns an error.
 */
export const processImage = async (processType, imageDataB64, maskDataB64 = null, imageTypeDescription = '') => {
    console.log(`imageApiUtils: Sending '${processType}' request to backend.`);

    const payload = {
        image: imageDataB64,
        mask: maskDataB64,
        process_type: processType,
        image_type_description: imageTypeDescription
    };

    try {
        const response = await fetch(`${API_BASE_URL}/image/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error(`imageApiUtils: API Error during ${processType}:`, errorData);
            throw new Error(errorData.error || `Failed to ${processType} via API.`);
        }

        const result = await response.json();
        console.log(`imageApiUtils: '${processType}' API response result:`, result);
        console.log(`imageApiUtils: Processed image data URL received:`, result.processed_image ? "present" : "null/empty");

        return result.processed_image;

    } catch (error) {
        console.error(`imageApiUtils: Error during API call for ${processType}:`, error);
        throw error; // Re-throw to be handled by the calling hook/component
    }
};
