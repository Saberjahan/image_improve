# server/services/ai_service.py

import numpy as np
import time
import random
import cv2 # Import OpenCV
import tensorflow as tf
import warnings
from flask import request  
import os
import torch
from PIL import Image, ImageDraw, ImageFilter
from flask import jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Concatenate
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from skimage.restoration import inpaint as sk_inpaint
from skimage import img_as_float, img_as_ubyte
from skimage.transform import resize

import csv # Import the csv library
warnings.filterwarnings('ignore')

# Global model cache
def process_image_with_ai(image: Image.Image, mask: Image.Image = None, process_type: str = 'inpaint', image_type_description: str = '') -> Image.Image:
    """
    Performs AI-based image processing (expansion,detection, inpainting).

    Args:
        image (PIL.Image.Image): The original input image.
        mask (PIL.Image.Image, optional): A mask image (white for areas to process, black otherwise).
                                          Defaults to None.
        process_type (str): The type of AI process to simulate.
                            Expected values: 'detect', 'detect_from_mask', 'telea', 'navier_stokes',
                            'contextual_attention', 'gated_convolution', 'partial_convolution',
                            'generative_adversarial_network', 'transformer_based',
                            'stable_diffusion_inpaint', 'latent_diffusion_inpaint',
                            'llm_guided_inpaint', 'semantic_inpaint'.
        image_type_description (str): A description of the image content, for AI context.

    Returns:
        PIL.Image.Image: The processed image (either a generated mask or a repaired image).
    """
    print(f"AI Service: Processing '{process_type}' for image with description: '{image_type_description}'")
    time.sleep(1.5) # Simulate processing time

    width, height = image.size # Get dimensions from the original image
    _model_cache = {}
    _multiscale_model_cache = {}

    if process_type == 'detect_from_mask':
        
        # --- START OF USER-DEFINED PARAMETERS ---
        # Adjust these values to fine-tune the AI's behavior.
        
        # Multiplier that controls the sensitivity of the global threshold. 
        # Higher values will lower the threshold, allowing for a larger, more
        # inclusive final mask that can capture distant but weakly similar areas.
        expansion_multiplier = 3.0
        
        # The size of the kernel used in morphological operations to close gaps
        # and connect nearby components after the global classification. A larger
        # size will bridge larger gaps.
        morphological_kernel_size = 9 # Must be an odd number
        
        # Minimum size in pixels for a mask component to be considered valid.
        # Smaller components will be removed during post-processing.
        min_size = 300
        
        # Number of training epochs for the multi-scale feature extractor.
        # Increasing this can improve accuracy at the cost of speed.
        num_epochs = 1
        
        # The input size for the deep learning model.
        # Larger sizes can capture more detail but require more memory and are slower.
        dl_input_size = (512, 512)
        
        # --- END OF USER-DEFINED PARAMETERS ---

        if mask is None:
            print("AI Service: No mask provided. Falling back to full detection.")
            return process_image_with_ai(image, None, 'detect', image_type_description)

        print("AI Service: Global feature-based classification and mask generation...")

        def build_multiscale_feature_extractor(input_shape):
            """
            Build a simplified, faster multi-scale feature extraction model.
            Returns both the autoencoder and multiple scale encoders.
            """
            input_img = Input(shape=input_shape)
            
            # Scale 1: Fine details (1/2 resolution)
            x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
            x1 = BatchNormalization()(x1)
            scale1_features = MaxPooling2D((2, 2), padding='same')(x1)
            
            # Scale 2: Medium details (1/4 resolution)
            x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(scale1_features)
            x2 = BatchNormalization()(x2)
            scale2_features = MaxPooling2D((2, 2), padding='same')(x2)
            
            # Scale 3: Coarse semantics (1/8 resolution)
            x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(scale2_features)
            x3 = BatchNormalization()(x3)
            encoded = MaxPooling2D((2, 2), padding='same')(x3)
            
            # Simple decoder without skip connections
            x = UpSampling2D((2, 2))(encoded)
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
            
            # Create models
            autoencoder = Model(input_img, decoded)
            
            # Individual scale encoders
            scale1_encoder = Model(input_img, scale1_features)
            scale2_encoder = Model(input_img, scale2_features)
            scale3_encoder = Model(input_img, encoded)
            
            return autoencoder, scale1_encoder, scale2_encoder, scale3_encoder

        def extract_multiscale_properties(feature_maps, mask, image_rgb):
            """
            Extract properties from multiple scales with robust statistics.
            """
            properties = {}
            
            for scale_name, feature_map in feature_maps.items():
                try:
                    if feature_map is None or feature_map.size == 0:
                        continue
                        
                    fh, fw, fc = feature_map.shape
                    ih, iw = mask.shape
                    
                    # Resize mask to feature dimensions
                    mask_resized = cv2.resize(mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)
                    
                    if np.sum(mask_resized) < 5:  # Need minimum pixels
                        continue
                    
                    # Extract features from masked region
                    masked_features = feature_map[mask_resized == 1]
                    
                    if len(masked_features) == 0:
                        continue
                    
                    # Robust statistics
                    feature_mean = np.mean(masked_features, axis=0)
                    feature_std = np.std(masked_features, axis=0)
                    
                    properties[scale_name] = {
                        'mean': feature_mean,
                        'std': feature_std + 1e-6,  # Prevent division by zero
                        'shape': (fh, fw, fc)
                    }
                    
                except Exception as e:
                    print(f"Warning: Failed to extract properties for {scale_name}: {e}")
                    continue
            
            # Extract color properties from original image
            try:
                masked_colors = image_rgb[mask == 1]
                if len(masked_colors) > 0:
                    properties['color'] = {
                        'mean': np.mean(masked_colors, axis=0),
                        'std': np.std(masked_colors, axis=0) + 1e-6,
                    }
            except Exception as e:
                print(f"Warning: Failed to extract color properties: {e}")
            
            return properties

        def compute_multiscale_similarity(feature_maps, image_rgb, properties):
            """
            Compute similarity map using multi-scale features for the entire image.
            This method is now global, not iterative.
            """
            if not properties:
                return None
            
            h, w = image_rgb.shape[:2]
            similarity_maps = []
            scale_weights = {'scale1': 0.3, 'scale2': 0.4, 'scale3': 0.3}  # Emphasize medium scale
            
            # Compute similarity for each scale
            for scale_name, feature_map in feature_maps.items():
                if scale_name not in properties or feature_map is None:
                    continue
                    
                try:
                    props = properties[scale_name]
                    fh, fw, fc = props['shape']
                    
                    # Calculate similarity using robust statistics
                    features_2d = feature_map.reshape(-1, fc)
                    mean_feat = props['mean']
                    std_feat = props['std']
                    
                    # Mahalanobis-like distance
                    normalized_diff = (features_2d - mean_feat) / std_feat
                    distances = np.sqrt(np.sum(normalized_diff ** 2, axis=1))
                    
                    # Convert to similarity (higher is more similar)
                    similarity = np.exp(-distances / 2.0)  # Gaussian-like similarity
                    
                    # Reshape and resize to original image size
                    similarity_small = similarity.reshape(fh, fw)
                    similarity_map = cv2.resize(similarity_small, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    # Apply scale weight
                    weight = scale_weights.get(scale_name, 1.0 / len(feature_maps))
                    similarity_maps.append(similarity_map * weight)
                    
                except Exception as e:
                    print(f"Warning: Failed to compute similarity for {scale_name}: {e}")
                    continue
            
            if not similarity_maps:
                return np.zeros((h, w))
            
            # Combine similarity maps
            combined_similarity = np.sum(similarity_maps, axis=0)
            
            # Add color similarity if available
            if 'color' in properties:
                try:
                    color_props = properties['color']
                    color_mean = color_props['mean']
                    color_std = color_props['std']
                    
                    # Compute color similarity
                    color_diff = (image_rgb - color_mean) / color_std
                    color_distances = np.sqrt(np.sum(color_diff ** 2, axis=2))
                    color_similarity = np.exp(-color_distances / 2.0)
                    
                    # Combine with feature similarity (weighted)
                    combined_similarity = 0.7 * combined_similarity + 0.3 * color_similarity
                    
                except Exception as e:
                    print(f"Warning: Failed to compute color similarity: {e}")
            
            return combined_similarity

        # Main processing logic starts here
        try:
            # Get original image dimensions
            orig_w, orig_h = image.size
            
            # Prepare input image for deep learning model
            img_resized = image.resize(dl_input_size, Image.LANCZOS)
            img_np = np.array(img_resized.convert('RGB')).astype(np.float32) / 255.0
            
            # Conditional denoising
            if np.std(img_np) > 0.15:
                img_np = cv2.bilateralFilter(img_np, 3, 50, 50)
            
            h, w = img_np.shape[:2]
            input_shape = (h, w, 3)

            # Build or retrieve cached models
            cache_key = f"{input_shape[0]}x{input_shape[1]}"
            if cache_key not in _multiscale_model_cache:
                print("AI Service: Building multi-scale feature extraction models...")
                
                autoencoder, scale1_encoder, scale2_encoder, scale3_encoder = build_multiscale_feature_extractor(input_shape)
                autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                
                _multiscale_model_cache[cache_key] = {
                    'autoencoder': autoencoder,
                    'scale1_encoder': scale1_encoder,
                    'scale2_encoder': scale2_encoder,
                    'scale3_encoder': scale3_encoder
                }
            else:
                print("AI Service: Using cached multi-scale models.")
            
            models = _multiscale_model_cache[cache_key]
            autoencoder = models['autoencoder']
            
            # Fast training
            print(f"AI Service: Training multi-scale feature extractor for {num_epochs} epochs...")
            input_img_batch = np.expand_dims(img_np, axis=0)
            
            # Train with error handling
            try:
                autoencoder.fit(input_img_batch, input_img_batch, epochs=num_epochs, verbose=0)
            except Exception as e:
                print(f"Warning: Training failed: {e}. Using pre-trained features.")

            # Prepare initial mask
            mask_resized = mask.resize((w, h), Image.NEAREST)
            initial_mask = (np.array(mask_resized.convert('L')) > 128).astype(np.uint8)

            if np.sum(initial_mask) == 0:
                print("AI Service: Empty mask provided. Returning original mask.")
                return mask

            # Basic morphological cleaning to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel)

            # Extract multi-scale features
            print("AI Service: Extracting multi-scale features...")
            feature_maps = {}
            
            try:
                feature_maps['scale1'] = models['scale1_encoder'].predict(input_img_batch, verbose=0)[0]
                feature_maps['scale2'] = models['scale2_encoder'].predict(input_img_batch, verbose=0)[0]
                feature_maps['scale3'] = models['scale3_encoder'].predict(input_img_batch, verbose=0)[0]
                
                print(f"AI Service: Extracted features - Scale1: {feature_maps['scale1'].shape}, Scale2: {feature_maps['scale2'].shape}, Scale3: {feature_maps['scale3'].shape}")
                
            except Exception as e:
                print(f"Error extracting features: {e}")
                # Fallback to simple processing without multi-scale
                return mask

            # STEP 1: Extract multi-scale properties from the initial, un-cleaned mask
            print("AI Service: Step 1 - Extracting multi-scale properties...")
            initial_properties = extract_multiscale_properties(feature_maps, initial_mask, img_np)
            
            if not initial_properties:
                print("AI Service: Failed to extract properties.")
                return mask
            
            print(f"AI Service: Extracted properties for {len(initial_properties)} scales/modalities")
            
            # STEP 2: Compute multi-scale similarity map for the ENTIRE image
            print("AI Service: Step 2 - Computing multi-scale similarity map for the whole image...")
            similarity_map = compute_multiscale_similarity(feature_maps, img_np, initial_properties)
            
            if similarity_map is None:
                print("AI Service: Failed to compute similarity map.")
                return mask
            
            print(f"AI Service: Similarity map range: {np.min(similarity_map):.3f} to {np.max(similarity_map):.3f}")

            # STEP 3: Apply a global threshold to the similarity map
            print("AI Service: Step 3 - Applying global threshold...")
            
            # Use the similarity map to find the most probable pixels of the initial mask
            core_similarities = similarity_map[initial_mask == 1]
            if len(core_similarities) > 0:
                threshold_base = np.median(core_similarities)
                threshold_spread = np.std(core_similarities)
            else:
                threshold_base = 0.5
                threshold_spread = 0.1
                
            # Use a more aggressive threshold based on the expansion multiplier
            final_threshold = max(0.05, threshold_base - threshold_spread * expansion_multiplier)
            
            # Generate the new mask by thresholding the entire similarity map
            final_mask_small = (similarity_map >= final_threshold).astype(np.uint8)
            
            print(f"AI Service: Generated mask based on global threshold of {final_threshold:.3f}")
            print(f"AI Service: New mask has {np.sum(final_mask_small)} pixels")

            # STEP 4: Post-processing and upscaling
            print("AI Service: Step 4 - Post-processing...")
            
            # Use a configurable kernel size for morphological closing to connect
            # any remaining small gaps.
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphological_kernel_size, morphological_kernel_size))
            final_mask_small = cv2.morphologyEx(final_mask_small, cv2.MORPH_CLOSE, kernel_close)
            
            # Remove small components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask_small, connectivity=8)
            cleaned_final = np.zeros_like(final_mask_small)
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    cleaned_final[labels == i] = 1

            # Upscale the final mask to the original image dimensions
            result_mask_upscaled = cv2.resize(cleaned_final, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            result_mask = (result_mask_upscaled * 255).astype(np.uint8)
            
            print(f"AI Service: Final upscaled mask: {np.sum(result_mask > 0)} pixels")
            print("AI Service: Global feature-based classification complete.")
            
            return Image.fromarray(result_mask, mode='L')

        except Exception as e:
            print(f"AI Service: Error in global classification: {e}")
            print("AI Service: Falling back to original mask.")
            return mask

    elif process_type == 'detect':
        # --- START OF USER-DEFINED PARAMETERS ---
        # Adjust these values to fine-tune the AI's behavior.

        # A lower value means a higher sensitivity to anomalies, resulting in a larger mask.
        # This is a percentage of the total distribution's variance.
        anomaly_sensitivity = 2.0 
        
        # The input size for the deep learning model.
        # Larger sizes can capture more detail but require more memory and are slower.
        dl_input_size = (512, 512)
        
        # Minimum size in pixels for a mask component to be considered valid.
        # Smaller components will be removed during post-processing.
        min_size = 300
        
        # The size of the kernel used in morphological operations.
        # A larger size will bridge larger gaps between masked components.
        morphological_kernel_size = 9 # Must be an odd number

        # Number of training epochs for the multi-scale feature extractor.
        num_epochs = 1
        
        # --- END OF USER-DEFINED PARAMETERS ---

        def build_multiscale_feature_extractor(input_shape):
            """
            Build a simplified, faster multi-scale feature extraction model.
            Returns both the autoencoder and multiple scale encoders.
            """
            input_img = Input(shape=input_shape)
            
            # Scale 1: Fine details (1/2 resolution)
            x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
            x1 = BatchNormalization()(x1)
            scale1_features = MaxPooling2D((2, 2), padding='same')(x1)
            
            # Scale 2: Medium details (1/4 resolution)
            x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(scale1_features)
            x2 = BatchNormalization()(x2)
            scale2_features = MaxPooling2D((2, 2), padding='same')(x2)
            
            # Scale 3: Coarse semantics (1/8 resolution)
            x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(scale2_features)
            x3 = BatchNormalization()(x3)
            encoded = MaxPooling2D((2, 2), padding='same')(x3)
            
            # Simple decoder without skip connections
            x = UpSampling2D((2, 2))(encoded)
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
            
            # Create models
            autoencoder = Model(input_img, decoded)
            
            # Individual scale encoders
            scale1_encoder = Model(input_img, scale1_features)
            scale2_encoder = Model(input_img, scale2_features)
            scale3_encoder = Model(input_img, encoded)
            
            return autoencoder, scale1_encoder, scale2_encoder, scale3_encoder

        def extract_multiscale_properties(feature_maps, mask, image_rgb):
            """
            Extract properties from multiple scales with robust statistics.
            If mask is None, properties are extracted from the entire image.
            """
            properties = {}
            
            for scale_name, feature_map in feature_maps.items():
                try:
                    if feature_map is None or feature_map.size == 0:
                        continue
                        
                    fh, fw, fc = feature_map.shape
                    
                    # Use mask if available, otherwise use the whole feature map
                    if mask is not None:
                        ih, iw = mask.shape
                        mask_resized = cv2.resize(mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)
                        if np.sum(mask_resized) < 5:
                            continue
                        masked_features = feature_map[mask_resized == 1]
                    else:
                        masked_features = feature_map.reshape(-1, fc)
                    
                    if len(masked_features) == 0:
                        continue
                    
                    feature_mean = np.mean(masked_features, axis=0)
                    feature_std = np.std(masked_features, axis=0)
                    
                    properties[scale_name] = {
                        'mean': feature_mean,
                        'std': feature_std + 1e-6,
                        'shape': (fh, fw, fc)
                    }
                    
                except Exception as e:
                    print(f"Warning: Failed to extract properties for {scale_name}: {e}")
                    continue
            
            # Extract color properties from original image
            try:
                if mask is not None:
                    masked_colors = image_rgb[mask == 1]
                else:
                    masked_colors = image_rgb.reshape(-1, 3)

                if len(masked_colors) > 0:
                    properties['color'] = {
                        'mean': np.mean(masked_colors, axis=0),
                        'std': np.std(masked_colors, axis=0) + 1e-6,
                    }
            except Exception as e:
                print(f"Warning: Failed to extract color properties: {e}")
            
            return properties

        def compute_multiscale_similarity(feature_maps, image_rgb, properties):
            """
            Compute similarity map using multi-scale features for the entire image.
            This method is now global, not iterative.
            """
            if not properties:
                return None
            
            h, w = image_rgb.shape[:2]
            similarity_maps = []
            scale_weights = {'scale1': 0.3, 'scale2': 0.4, 'scale3': 0.3}  # Emphasize medium scale
            
            # Compute similarity for each scale
            for scale_name, feature_map in feature_maps.items():
                if scale_name not in properties or feature_map is None:
                    continue
                    
                try:
                    props = properties[scale_name]
                    fh, fw, fc = props['shape']
                    
                    features_2d = feature_map.reshape(-1, fc)
                    mean_feat = props['mean']
                    std_feat = props['std']
                    
                    # Mahalanobis-like distance
                    normalized_diff = (features_2d - mean_feat) / std_feat
                    distances = np.sqrt(np.sum(normalized_diff ** 2, axis=1))
                    
                    # Convert to similarity (higher is more similar)
                    similarity = np.exp(-distances / 2.0)
                    
                    # Reshape and resize to original image size
                    similarity_small = similarity.reshape(fh, fw)
                    similarity_map = cv2.resize(similarity_small, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    # Apply scale weight
                    weight = scale_weights.get(scale_name, 1.0 / len(feature_maps))
                    similarity_maps.append(similarity_map * weight)
                    
                except Exception as e:
                    print(f"Warning: Failed to compute similarity for {scale_name}: {e}")
                    continue
            
            if not similarity_maps:
                return np.zeros((h, w))
            
            # Combine similarity maps
            combined_similarity = np.sum(similarity_maps, axis=0)
            
            # Add color similarity if available
            if 'color' in properties:
                try:
                    color_props = properties['color']
                    color_mean = color_props['mean']
                    color_std = color_props['std']
                    
                    # Compute color similarity
                    color_diff = (image_rgb - color_mean) / color_std
                    color_distances = np.sqrt(np.sum(color_diff ** 2, axis=2))
                    color_similarity = np.exp(-color_distances / 2.0)
                    
                    # Combine with feature similarity (weighted)
                    combined_similarity = 0.7 * combined_similarity + 0.3 * color_similarity
                    
                except Exception as e:
                    print(f"Warning: Failed to compute color similarity: {e}")
            
            return combined_similarity


        print("AI Service: Starting full detection using global anomaly classification.")

        try:
            # Get original image dimensions
            orig_w, orig_h = image.size
            
            # Prepare input image for deep learning model
            dl_input_size = (512, 512)
            img_resized = image.resize(dl_input_size, Image.LANCZOS)
            img_np = np.array(img_resized.convert('RGB')).astype(np.float32) / 255.0

            if np.std(img_np) > 0.15:
                img_np = cv2.bilateralFilter(img_np, 3, 50, 50)
            
            h, w = img_np.shape[:2]
            input_shape = (h, w, 3)

            print("AI Service: Building multi-scale feature extraction models...")
            autoencoder, scale1_encoder, scale2_encoder, scale3_encoder = build_multiscale_feature_extractor(input_shape)
            autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Fast training
            print(f"AI Service: Training multi-scale feature extractor for {num_epochs} epochs...")
            input_img_batch = np.expand_dims(img_np, axis=0)
            try:
                autoencoder.fit(input_img_batch, input_img_batch, epochs=num_epochs, verbose=0)
            except Exception as e:
                print(f"Warning: Training failed: {e}. Using pre-trained features.")

            # Extract multi-scale features from the entire image
            print("AI Service: Extracting multi-scale features from the entire image...")
            feature_maps = {
                'scale1': scale1_encoder.predict(input_img_batch, verbose=0)[0],
                'scale2': scale2_encoder.predict(input_img_batch, verbose=0)[0],
                'scale3': scale3_encoder.predict(input_img_batch, verbose=0)[0]
            }

            # Calculate global features from the entire image (no mask)
            print("AI Service: Calculating global features to define 'normal' patterns...")
            global_properties = extract_multiscale_properties(feature_maps, None, img_np)

            if not global_properties:
                print("AI Service: Failed to calculate global properties. Returning empty mask.")
                return Image.new('L', image.size, 0)

            # Compute similarity map for the entire image based on global features
            print("AI Service: Computing similarity map against global features...")
            similarity_map = compute_multiscale_similarity(feature_maps, img_np, global_properties)

            if similarity_map is None:
                print("AI Service: Failed to compute similarity map. Returning empty mask.")
                return Image.new('L', image.size, 0)
            
            # Find anomalous areas by thresholding the similarity map
            print("AI Service: Applying threshold to find anomalies...")
            
            # A low similarity score indicates an anomaly. We'll use a threshold based on
            # the overall similarity distribution.
            mean_similarity = np.mean(similarity_map)
            std_similarity = np.std(similarity_map)

            # Calculate a dynamic threshold. Lower `anomaly_sensitivity` results in a higher
            # threshold (less sensitive), a higher `anomaly_sensitivity` results in a lower
            # threshold (more sensitive).
            final_threshold = max(0.01, mean_similarity - std_similarity * anomaly_sensitivity)

            # Mask areas that have low similarity to the overall image pattern
            anomaly_mask_small = (similarity_map <= final_threshold).astype(np.uint8)
            
            print(f"AI Service: Detected {np.sum(anomaly_mask_small)} anomalous pixels based on a threshold of {final_threshold:.3f}.")
            
            # Post-processing and upscaling
            print("AI Service: Post-processing and upscaling the mask...")
            
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphological_kernel_size, morphological_kernel_size))
            final_mask_small = cv2.morphologyEx(anomaly_mask_small, cv2.MORPH_CLOSE, kernel_close)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask_small, connectivity=8)
            cleaned_final = np.zeros_like(final_mask_small)
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    cleaned_final[labels == i] = 1

            result_mask_upscaled = cv2.resize(cleaned_final, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            result_mask = (result_mask_upscaled * 255).astype(np.uint8)

            print(f"AI Service: Final upscaled mask: {np.sum(result_mask > 0)} pixels.")
            print("AI Service: Global anomaly detection complete.")
            
            return Image.fromarray(result_mask, mode='L')

        except Exception as e:
            print(f"AI Service: Error in global anomaly detection: {e}.")
            print("AI Service: Returning empty mask.")
            return Image.new('L', image.size, 0)

    elif process_type in ['telea', 'navier_stokes','exemplar_based', 'non_local_means',
                          'contextual_attention','gated_convolution','partial_convolution',
                          'generative_adversarial_network','transformer_based',
                          'stable_diffusion_inpaint','latent_diffusion_inpaint','controlnet_inpaint',
                          'llm_guided_inpaint','semantic_inpaint']:
        

        print(f"AI Service: Performing inpainting using method: {process_type}")

        # Common preprocessing
        img_np_rgb = np.array(image.convert('RGB'))
        mask_resized_pil = mask.resize(image.size, Image.NEAREST)
        mask_np = np.array(mask_resized_pil)
        
        #import csv
        #with open(r'O:\Data\LLM\Image_Improve\server\services\mask.csv', 'w', newline='') as file:
        #    writer = csv.writer(file)
        #    for row in mask_np:
        #        writer.writerow(row)
                
        # Method-specific implementations
        if process_type == 'telea':
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)
            mask_clean = cv2.medianBlur(mask_for_inpaint, 3)
            repaired_rgb = cv2.inpaint(img_np_rgb[..., ::-1], mask_clean, 5, cv2.INPAINT_TELEA)[..., ::-1]
            repaired_pil = Image.fromarray(repaired_rgb)

        elif process_type == 'navier_stokes':
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)
            mask_clean = cv2.medianBlur(mask_for_inpaint, 3)
            repaired_rgb = cv2.inpaint(img_np_rgb[..., ::-1], mask_clean, 5, cv2.INPAINT_NS)[..., ::-1]
            repaired_pil = Image.fromarray(repaired_rgb)

        elif process_type in ('exemplar_based', 'criminisi'):
            mask_bool = mask_np == 0  # assuming 0 = corrupted
            image_float = img_as_float(img_np_rgb)
            repaired_float = sk_inpaint.inpaint_biharmonic(image_float, mask_bool, channel_axis=-1)
            repaired_rgb = img_as_ubyte(repaired_float)
            repaired_pil = Image.fromarray(repaired_rgb)

        elif process_type == 'non_local_means':
            # Prepare mask: 0 = valid, 1 = damaged
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)
            # Clean mask with median filter
            mask_clean = cv2.medianBlur(mask_for_inpaint, 3)
            # Convert image to BGR
            img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
            # Denoise first for better inpainting quality
            denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)
            # Downscale for coarse inpainting and upscale back
            small_img = cv2.resize(denoised, (0, 0), fx=0.5, fy=0.5)
            small_mask = cv2.resize(mask_clean, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            coarse = cv2.inpaint(small_img, small_mask * 255, 6, cv2.INPAINT_NS)  # multiply by 255 because OpenCV expects 0/255
            coarse_up = cv2.resize(coarse, (denoised.shape[1], denoised.shape[0]))
            # Use dilated mask to blend coarse inpainted regions with original denoised
            mask_dil = cv2.dilate(mask_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            refined = denoised.copy()
            refined[mask_dil > 0] = coarse_up[mask_dil > 0]
            # Final fine inpainting step
            result = cv2.inpaint(refined, mask_clean * 255, 4, cv2.INPAINT_NS)
            # Convert back to PIL
            repaired_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        elif process_type == 'contextual_attention':
            # Prepare mask: 0 = valid, 1 = damaged
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)

            img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
            mask_clean = cv2.medianBlur(mask_for_inpaint, 3)
            
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            
            valid = mask_clean == 0
            avg_grad = np.mean(grad_mag[valid]) if np.any(valid) else 50
            radius = 3 if avg_grad > 80 else 6 if avg_grad > 40 else 10
            
            result = cv2.inpaint(img_bgr, mask_clean, radius, cv2.INPAINT_NS)
            result = cv2.inpaint(result, mask_clean, max(2, radius // 2), cv2.INPAINT_TELEA)
            
            repaired_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
   
        elif process_type == 'gated_convolution':
            print("AI Service: Processing 'gated_convolution' for image with description: ''")
            print("AI Service: Performing inpainting using method: gated_convolution")

            model_path = "gated_conv_model.keras"

            # Define the Gated Convolution Layer
            class GatedConv2D(tf.keras.layers.Layer):
                def __init__(self, filters, kernel_size, **kwargs):
                    super(GatedConv2D, self).__init__(**kwargs)
                    self.conv = Conv2D(filters, kernel_size, padding='same', activation=None)
                    self.gate = Conv2D(filters, kernel_size, padding='same', activation='sigmoid')

                def call(self, x):
                    activation = self.conv(x)
                    gate_output = self.gate(x)
                    return activation * gate_output

            # Build and cache the model if not already done
            global gated_conv_model
            if 'gated_conv_model' not in globals():
                if os.path.exists(model_path):
                    print("Loading cached Gated Convolution model...")
                    gated_conv_model = load_model(model_path, custom_objects={'GatedConv2D': GatedConv2D})
                else:
                    print("Building and saving Gated Convolution model...")
                    inputs = Input(shape=(None, None, 4))  # RGB + mask

                    # Encoder
                    conv1 = GatedConv2D(32, 5)(inputs)
                    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

                    conv2 = GatedConv2D(64, 3)(pool1)
                    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

                    # Bottleneck
                    conv3 = GatedConv2D(128, 3)(pool2)

                    # Decoder
                    up1 = UpSampling2D(size=(2, 2))(conv3)
                    up1 = Concatenate()([up1, conv2])
                    conv4 = GatedConv2D(64, 3)(up1)

                    up2 = UpSampling2D(size=(2, 2))(conv4)
                    up2 = Concatenate()([up2, conv1])
                    conv5 = GatedConv2D(32, 5)(up2)

                    outputs = Conv2D(3, 3, padding='same', activation='sigmoid')(conv5)

                    gated_conv_model = Model(inputs=inputs, outputs=outputs)
                    gated_conv_model.compile(optimizer='adam', loss='mean_squared_error')

                    gated_conv_model.save(model_path)

            if mask is None:
                repaired_pil = image
            else:
                img_np_rgb = np.array(image, dtype=np.float32) / 255.0
                mask_np = np.array(mask.convert('L'), dtype=np.float32) / 255.0

                h, w = img_np_rgb.shape[:2]
                target_h = (h // 4) * 4
                target_w = (w // 4) * 4

                img_resized = cv2.resize(img_np_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
                mask_resized = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_AREA)

                inverted_mask = 1.0 - np.expand_dims(mask_resized, axis=-1)
                model_input = np.concatenate([img_resized, inverted_mask], axis=-1)
                model_input = np.expand_dims(model_input, axis=0)

                repaired_output = gated_conv_model.predict(model_input)[0]

                blended_image = img_resized * inverted_mask + repaired_output * (1.0 - inverted_mask)
                blended_image = np.clip(blended_image, 0, 1)

                repaired_pil = Image.fromarray((blended_image * 255).astype(np.uint8))
                repaired_pil = repaired_pil.resize(image.size, Image.Resampling.LANCZOS)

            return repaired_pil

        elif process_type == 'partial_convolution':
            import argparse
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torchvision.transforms.functional as TF

            class PartialConv2d(nn.Conv2d):
                """
                A custom 2D convolutional layer that only operates on valid pixels, updating
                the mask for subsequent layers. It handles cases where a kernel may fall
                on a masked area by normalizing the output based on the number of valid
                input pixels.
                """
                def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, groups=1, bias=True,
                            padding_mode='zeros'):
                    # Inherit from the parent class (Conv2d)
                    super(PartialConv2d, self).__init__(in_channels, out_channels,
                                                        kernel_size, stride=stride,
                                                        padding=padding, dilation=dilation,
                                                        groups=groups, bias=bias,
                                                        padding_mode=padding_mode)
                    # Define a kernel filled with ones for updating the mask
                    self.mask_kernel = torch.ones(self.out_channels, self.in_channels,
                                                self.kernel_size[0], self.kernel_size[1])
                    # Define a constant for normalization
                    self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] \
                                                        * self.mask_kernel.shape[3]
                    self.update_mask = None
                    self.mask_ratio = None
                    # Initialize the weights for image convolution
                    torch.nn.init.xavier_uniform_(self.weight)

                def forward(self, img, mask):
                    with torch.no_grad():
                        # Ensure the mask kernel is on the same device as the input image
                        if self.mask_kernel.type() != img.type():
                            self.mask_kernel = self.mask_kernel.to(img)
                        
                        # Use a standard convolution to calculate the sum of the mask values
                        update_mask = F.conv2d(mask, self.mask_kernel, bias=None,
                                            stride=self.stride, padding=self.padding,
                                            dilation=self.dilation, groups=1)
                        
                        # Calculate the mask ratio (sum of valid pixels in the kernel area)
                        self.mask_ratio = self.sum1 / (update_mask + 1e-8)
                        self.mask_ratio.masked_fill_(update_mask == 0, 0.0)

                        # Update the mask: any pixel with at least one valid input pixel becomes valid
                        new_mask = (update_mask > 0).float()
                    
                    # Convolve the image with the mask applied
                    img_conv = super(PartialConv2d, self).forward(img * mask)
                    # Convolve with the bias
                    if self.bias is not None:
                        img_conv = img_conv - self.bias.view(1, -1, 1, 1)

                    # Apply the normalization and new mask
                    output = img_conv * self.mask_ratio + (self.bias.view(1, -1, 1, 1) if self.bias is not None else 0) * new_mask

                    return output, new_mask

            class PConvUNet(nn.Module):
                """
                Partial Convolutional UNet with 8 encoder-decoder layers.
                """
                def __init__(self, finetune=False, layer_size=8):
                    super(PConvUNet, self).__init__()
                    self.freeze_enc_bn = finetune
                    self.layer_size = layer_size

                    # Encoder layers
                    self.enc_1 = PartialConv2d(3, 64, 7, 2, 3, bias=False)
                    self.enc_2 = PartialConv2d(64, 128, 5, 2, 2, bias=False)
                    self.enc_3 = PartialConv2d(128, 256, 5, 2, 2, bias=False)
                    self.enc_4 = PartialConv2d(256, 512, 3, 2, 1, bias=False)
                    self.enc_5 = PartialConv2d(512, 512, 3, 2, 1, bias=False)
                    self.enc_6 = PartialConv2d(512, 512, 3, 2, 1, bias=False)
                    self.enc_7 = PartialConv2d(512, 512, 3, 2, 1, bias=False)
                    self.enc_8 = PartialConv2d(512, 512, 3, 2, 1, bias=False)  # New layer

                    # Decoder layers
                    self.dec_8 = PartialConv2d(512 + 512, 512, 3, 1, 1, bias=False)  # New layer
                    self.dec_7 = PartialConv2d(512 + 512, 512, 3, 1, 1, bias=False)
                    self.dec_6 = PartialConv2d(512 + 512, 512, 3, 1, 1, bias=False)
                    self.dec_5 = PartialConv2d(512 + 512, 512, 3, 1, 1, bias=False)
                    self.dec_4 = PartialConv2d(512 + 256, 256, 3, 1, 1, bias=False)
                    self.dec_3 = PartialConv2d(256 + 128, 128, 3, 1, 1, bias=False)
                    self.dec_2 = PartialConv2d(128 + 64, 64, 3, 1, 1, bias=False)
                    self.dec_1 = PartialConv2d(64 + 3, 3, 3, 1, 1, bias=True)

                    self.relu = nn.ReLU(True)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, img, mask):
                    # Encoder
                    enc_f, enc_m = [], []

                    feature, update_mask = self.enc_1(img, mask)
                    feature = self.relu(feature)
                    enc_f.append(feature)
                    enc_m.append(update_mask)

                    for i in range(2, self.layer_size + 1):  # 2..8
                        feature, update_mask = getattr(self, f"enc_{i}")(feature, update_mask)
                        feature = self.relu(feature)
                        enc_f.append(feature)
                        enc_m.append(update_mask)

                    # Decoder
                    dec_in = enc_f.pop()  # enc_8
                    dec_mask = enc_m.pop()

                    for layer_num in reversed(range(1, self.layer_size)):  # 7 down to 1
                        dec_in = F.interpolate(dec_in, scale_factor=2, mode='nearest')
                        dec_mask = F.interpolate(dec_mask, scale_factor=2, mode='nearest')

                        enc_feat = enc_f.pop()
                        enc_mask = enc_m.pop()

                        dec_in = torch.cat([dec_in, enc_feat], dim=1)
                        dec_mask = torch.cat([dec_mask, enc_mask], dim=1)

                        feature, update_mask = getattr(self, f'dec_{layer_num + 1}')(dec_in, dec_mask)

                        if layer_num > 1:
                            feature = self.relu(feature)

                        dec_in = feature
                        dec_mask = update_mask

                    feature = F.interpolate(dec_in, scale_factor=2, mode='nearest')
                    update_mask = F.interpolate(dec_mask, scale_factor=2, mode='nearest')
                    feature = torch.cat([feature, img], dim=1)
                    update_mask = torch.cat([update_mask, mask], dim=1)

                    output, output_mask = self.dec_1(feature, update_mask)
                    output = self.sigmoid(output)

                    return output, output_mask

            def inpaint_and_return_pil(img_np_rgb: np.ndarray, mask_np: np.ndarray, model_path: str):
                """
                Performs image inpainting on a NumPy array and returns the result as a PIL Image.
                """
                # 1. Device Setup
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # 2. Model Loading
                print("Loading the Model...")
                model = PConvUNet(finetune=False, layer_size=7)

                # Load the pre-trained weights and adjust the keys to match the current model
                state_dict = torch.load(model_path, map_location=device)['model']
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Check for the unexpected '.conv' and remove it
                    if '.conv.weight' in k:
                        new_k = k.replace('.conv.weight', '.weight')
                        new_state_dict[new_k] = v
                    elif '.conv.bias' in k:
                        new_k = k.replace('.conv.bias', '.bias')
                        new_state_dict[new_k] = v
                    else:
                        new_state_dict[k] = v

                model.load_state_dict(new_state_dict)
                model.to(device)
                model.eval()

                # 3. Data Preparation: Convert NumPy arrays to PyTorch tensors
                print("Converting inputs from NumPy arrays to PyTorch tensors...")
                img_tensor = TF.to_tensor(Image.fromarray(img_np_rgb)).to(device)
                # Ensure mask has 3 channels for broadcasting with image
                mask_tensor = torch.from_numpy(mask_np).float().to(device).unsqueeze(0).expand(3, -1, -1)

                # Store the original dimensions for resizing the final output
                original_size = (img_tensor.shape[1], img_tensor.shape[2])
                
                # Resize input tensors to 256x256 for optimal model performance
                print(f"Resizing inputs to 256x256...")
                inp_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
                mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)

                # 4. Create the input with holes (inp)
                inp_with_holes = inp_tensor * mask_tensor

                # 5. Model Prediction
                print("Performing Model Prediction...")
                with torch.no_grad():
                    raw_out, _ = model(inp_with_holes, mask_tensor)

                # 6. Post-processing
                # The output is still at 256x256, we need to resize it back to the original size
                print(f"Resizing output back to {original_size}...")
                resized_output = F.interpolate(raw_out, size=original_size, mode='bilinear', align_corners=False)
                
                # Move the output tensor back to the CPU and remove the batch dimension.
                resized_output = resized_output.to(torch.device('cpu')).squeeze()
                # Clamp the values to the valid range [0.0, 1.0].
                resized_output = resized_output.clamp(0.0, 1.0)
                
                # 7. Final Image Generation (Repaired PIL image)
                # Get the original image and mask resized to original dimensions for final combination
                original_img_tensor = TF.to_tensor(Image.fromarray(img_np_rgb)).to('cpu')
                original_mask_tensor = torch.from_numpy(mask_np).float().to('cpu')
                
                output_tensor = (original_mask_tensor * original_img_tensor) + ((1 - original_mask_tensor) * resized_output)
                
                # Convert the final tensor to a PIL Image object.
                repaired_pil = TF.to_pil_image(output_tensor)
                
                print("Inpainting complete.")
                return repaired_pil
                                  
            model_path = "O:\\Data\\LLM\\Image_Improve\\model\\pretrained_pconv.pth"
            repaired_pil = inpaint_and_return_pil(img_np_rgb, mask_np, model_path)
            repaired_pil.save("repaired_image.png")




        elif process_type == 'generative_adversarial_network':
            # Placeholder for GAN-based inpainting (EdgeConnect, LAMA, etc.)
            # Would require loading pre-trained GAN model
            print("AI Service: GAN inpainting not yet implemented, falling back to Navier-Stokes")
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)         
            img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
            repaired_np_bgr = cv2.inpaint(img_np_bgr, mask_for_inpaint, inpaintRadius=5, flags=cv2.INPAINT_NS)
            repaired_pil = Image.fromarray(cv2.cvtColor(repaired_np_bgr, cv2.COLOR_BGR2RGB))
            
        elif process_type == 'transformer_based':
            # Placeholder for transformer-based inpainting
            # Would require loading Vision Transformer or hybrid model
            print("AI Service: Transformer inpainting not yet implemented, falling back to Telea")
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)
            img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
            repaired_np_bgr = cv2.inpaint(img_np_bgr, mask_for_inpaint, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
            repaired_pil = Image.fromarray(cv2.cvtColor(repaired_np_bgr, cv2.COLOR_BGR2RGB))
            
        elif process_type == 'stable_diffusion_inpaint':
            # Placeholder for Stable Diffusion inpainting
            # Would require loading SD model with inpainting weights
            print("AI Service: Stable Diffusion inpainting not yet implemented, falling back to Navier-Stokes")
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)
            img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
            repaired_np_bgr = cv2.inpaint(img_np_bgr, mask_for_inpaint, inpaintRadius=5, flags=cv2.INPAINT_NS)
            repaired_pil = Image.fromarray(cv2.cvtColor(repaired_np_bgr, cv2.COLOR_BGR2RGB))
            
        elif process_type == 'latent_diffusion_inpaint':
            # Placeholder for Latent Diffusion inpainting
            # Would require loading latent diffusion model
            print("AI Service: Latent diffusion inpainting not yet implemented, falling back to Navier-Stokes")
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)
            img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
            repaired_np_bgr = cv2.inpaint(img_np_bgr, mask_for_inpaint, inpaintRadius=5, flags=cv2.INPAINT_NS)
            repaired_pil = Image.fromarray(cv2.cvtColor(repaired_np_bgr, cv2.COLOR_BGR2RGB))
            
        elif process_type == 'llm_guided_inpaint':
            # Placeholder for LLM-guided inpainting
            # Would require integration with LLM for semantic understanding
            print("AI Service: LLM-guided inpainting not yet implemented, falling back to Telea")
            print(f"AI Service: Image context - {image_type_description}")
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)
            img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
            repaired_np_bgr = cv2.inpaint(img_np_bgr, mask_for_inpaint, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
            repaired_pil = Image.fromarray(cv2.cvtColor(repaired_np_bgr, cv2.COLOR_BGR2RGB))
            
        elif process_type == 'semantic_inpaint':
            # Placeholder for semantic-aware inpainting
            # Would require model with semantic understanding capabilities
            print("AI Service: Semantic inpainting not yet implemented, falling back to Navier-Stokes")
            print(f"AI Service: Image context - {image_type_description}")
            mask_for_inpaint = np.where(mask_np != 0, 0, 1).astype(np.uint8)
            img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
            repaired_np_bgr = cv2.inpaint(img_np_bgr, mask_for_inpaint, inpaintRadius=5, flags=cv2.INPAINT_NS)
            repaired_pil = Image.fromarray(cv2.cvtColor(repaired_np_bgr, cv2.COLOR_BGR2RGB))
        
        return repaired_pil

    else:
        print(f"AI Service: Unknown process_type '{process_type}'. Returning original image.")
        return image


def generate_keywords_from_description(description: str) -> list[str]:
    """
    Generates keywords from a user-provided description using an LLM.

    Args:
        description (str): The user's description of the image.

    Returns:
        list[str]: A list of generated keywords.
    """
    print(f"AI Service: Generating keywords for description: '{description}'")
    time.sleep(0.5) # Simulate a small delay

    keywords = [word.strip() for word in description.lower().split(',') if word.strip()]
    
    if 'dog' in keywords or 'puppy' in keywords:
        keywords.extend(['canine', 'pet', 'animal'])
    if 'building' in keywords or 'house' in keywords:
        keywords.extend(['architecture', 'structure', 'urban'])
    if 'nature' in keywords or 'tree' in keywords:
        keywords.extend(['landscape', 'green', 'outdoor'])
    if 'water' in keywords or 'ocean' in keywords:
        keywords.extend(['sea', 'blue', 'liquid'])

    return list(set(keywords))