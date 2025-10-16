/**
 * YOLOv8 Object Detection Frontend
 * Handles all UI interactions and API communication
 */

// API Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';

// Global state
let currentImage = null;
let currentResults = null;
let outputImagePath = null;

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeBtn = document.getElementById('removeBtn');
const detectBtn = document.getElementById('detectBtn');
const btnText = document.getElementById('btnText');
const loading = document.getElementById('loading');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');
const errorMessage = document.getElementById('errorMessage');

// Results Elements
const noResults = document.getElementById('noResults');
const resultsContainer = document.getElementById('resultsContainer');
const totalDetections = document.getElementById('totalDetections');
const processingTime = document.getElementById('processingTime');
const imageSize = document.getElementById('imageSize');
const detectionsList = document.getElementById('detectionsList');
const jsonViewer = document.getElementById('jsonViewer');
const downloadJson = document.getElementById('downloadJson');
const downloadImage = document.getElementById('downloadImage');

/**
 * Initialize event listeners
 */
function init() {
    // Upload zone click
    uploadZone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);

    // Remove image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Detect button
    detectBtn.addEventListener('click', runDetection);

    // Confidence slider
    confidenceSlider.addEventListener('input', (e) => {
        confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Download buttons
    downloadJson.addEventListener('click', downloadJsonFile);
    downloadImage.addEventListener('click', downloadAnnotatedImage);

    // Check API health on load
    checkApiHealth();
}

/**
 * Check if API is available
 */
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✅ API is healthy');
        }
    } catch (error) {
        showError('⚠️ Warning: Cannot connect to API. Make sure the backend is running on http://127.0.0.1:8000');
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        loadImage(file);
    }
}

/**
 * Handle drag over
 */
function handleDragOver(event) {
    event.preventDefault();
    uploadZone.classList.add('dragover');
}

/**
 * Handle drag leave
 */
function handleDragLeave(event) {
    event.preventDefault();
    uploadZone.classList.remove('dragover');
}

/**
 * Handle drop
 */
function handleDrop(event) {
    event.preventDefault();
    uploadZone.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        loadImage(file);
    } else {
        showError('Please upload a valid image file (JPG, PNG, JPEG)');
    }
}

/**
 * Load and preview image
 */
function loadImage(file) {
    // Validate file type
    if (!file.type.match('image/(jpeg|jpg|png)')) {
        showError('Invalid file type. Please upload JPG, PNG, or JPEG');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Maximum size is 10MB');
        return;
    }

    currentImage = file;
    const reader = new FileReader();

    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadZone.style.display = 'none';
        imagePreview.style.display = 'block';
        detectBtn.disabled = false;
        hideError();
        
        // Reset results
        hideResults();
    };

    reader.onerror = () => {
        showError('Failed to load image. Please try again.');
    };

    reader.readAsDataURL(file);
}

/**
 * Reset upload state
 */
function resetUpload() {
    currentImage = null;
    currentResults = null;
    outputImagePath = null;
    fileInput.value = '';
    previewImg.src = '';
    uploadZone.style.display = 'block';
    imagePreview.style.display = 'none';
    detectBtn.disabled = true;
    hideError();
    hideResults();
}

/**
 * Run object detection
 */
async function runDetection() {
    if (!currentImage) return;

    // Prepare UI
    detectBtn.disabled = true;
    loading.style.display = 'block';
    btnText.textContent = '⏳ Processing...';
    hideError();
    hideResults();

    // Prepare form data
    const formData = new FormData();
    formData.append('file', currentImage);
    formData.append('conf_threshold', confidenceSlider.value);
    formData.append('save_output', 'true');

    try {
        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Detection failed');
        }

        const data = await response.json();
        currentResults = data;
        outputImagePath = data.output_image_path;
        
        displayResults(data);
        
    } catch (error) {
        console.error('Detection error:', error);
        showError(`Detection failed: ${error.message}`);
    } finally {
        loading.style.display = 'none';
        detectBtn.disabled = false;
        btnText.textContent = '🚀 Detect Objects';
    }
}

/**
 * Display detection results
 */
function displayResults(data) {
    noResults.style.display = 'none';
    resultsContainer.style.display = 'block';

    // Update stats
    totalDetections.textContent = data.total_detections;
    processingTime.textContent = `${data.processing_time.toFixed(2)}s`;
    imageSize.textContent = `${data.image_size.width}x${data.image_size.height}`;

    // Display detections list
    detectionsList.innerHTML = '';
    
    if (data.detections.length === 0) {
        detectionsList.innerHTML = '<div style="text-align: center; padding: 40px; color: #707070;">No objects detected. Try lowering the confidence threshold.</div>';
    } else {
        data.detections.forEach((detection, index) => {
            const item = createDetectionItem(detection, index);
            detectionsList.appendChild(item);
        });
    }

    // Display JSON
    jsonViewer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Create detection item element
 */
function createDetectionItem(detection, index) {
    const div = document.createElement('div');
    div.className = 'detection-item';
    
    // Set confidence class
    if (detection.confidence > 0.85) {
        div.classList.add('high-confidence');
    } else if (detection.confidence > 0.7) {
        div.classList.add('medium-confidence');
    } else {
        div.classList.add('low-confidence');
    }

    // Get confidence color
    const confidenceColor = detection.confidence > 0.85 ? '#00ff88' : 
                           detection.confidence > 0.7 ? '#ffff00' : '#ffa500';

    div.innerHTML = `
        <div class="detection-header">
            <div class="detection-class">${detection.class_name}</div>
            <div class="detection-confidence" style="background: ${confidenceColor}20; color: ${confidenceColor};">
                ${(detection.confidence * 100).toFixed(1)}%
            </div>
        </div>
        <div class="detection-details">
            <div><strong>Class ID:</strong> ${detection.class_id}</div>
            <div><strong>Position:</strong> (${detection.bbox.x1.toFixed(0)}, ${detection.bbox.y1.toFixed(0)}) to (${detection.bbox.x2.toFixed(0)}, ${detection.bbox.y2.toFixed(0)})</div>
            <div><strong>Size:</strong> ${detection.bbox.width.toFixed(0)}x${detection.bbox.height.toFixed(0)} px</div>
        </div>
    `;

    // Add hover animation
    div.addEventListener('mouseenter', () => {
        div.style.boxShadow = `0 4px 20px ${confidenceColor}40`;
    });

    div.addEventListener('mouseleave', () => {
        div.style.boxShadow = 'none';
    });

    return div;
}

/**
 * Hide results
 */
function hideResults() {
    noResults.style.display = 'block';
    resultsContainer.style.display = 'none';
}

/**
 * Download JSON file
 */
function downloadJsonFile() {
    if (!currentResults) return;

    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `detection_results_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

/**
 * Download annotated image
 */
async function downloadAnnotatedImage() {
    if (!outputImagePath) {
        showError('No annotated image available. The detection may not have produced results.');
        return;
    }

    try {
        const imageUrl = `${API_BASE_URL}${outputImagePath}`;
        const response = await fetch(imageUrl);
        
        if (!response.ok) {
            throw new Error('Failed to fetch annotated image');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `detected_objects_${Date.now()}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
    } catch (error) {
        console.error('Download error:', error);
        showError('Failed to download annotated image. Please try again.');
    }
}

/**
 * Show error message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.style.display = 'none';
}

/**
 * Format confidence score with color
 */
function getConfidenceColor(confidence) {
    if (confidence > 0.85) return '#00ff88';
    if (confidence > 0.7) return '#ffff00';
    return '#ffa500';
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

/**
 * Add keyboard shortcuts
 */
document.addEventListener('keydown', (e) => {
    // Press 'D' to detect (when image is loaded)
    if (e.key === 'd' || e.key === 'D') {
        if (!detectBtn.disabled) {
            runDetection();
        }
    }
    
    // Press 'R' to reset
    if (e.key === 'r' || e.key === 'R') {
        if (currentImage) {
            resetUpload();
        }
    }
    
    // Press 'Escape' to clear error
    if (e.key === 'Escape') {
        hideError();
    }
});

/**
 * Initialize app on page load
 */
document.addEventListener('DOMContentLoaded', init);

/**
 * Log app info to console
 */
console.log('%c🔮 YOLOv8 AI Vision Frontend', 'font-size: 20px; color: #8a2be2; font-weight: bold;');
console.log('%cConnected to API:', 'color: #00d4ff;', API_BASE_URL);
console.log('%cKeyboard Shortcuts:', 'color: #00d4ff;');
console.log('  D - Run Detection');
console.log('  R - Reset Upload');
console.log('  Esc - Close Error');
console.log('%c✨ Ready to detect objects!', 'color: #00ff88; font-weight: bold;');