// DOM elements
const video = document.getElementById('video');
const emojiElement = document.getElementById('emoji');
const classNameElement = document.getElementById('class-name');

// Model configuration
const MODEL_PATH = './rps_web_model/model.json';
const CLASS_NAMES = ['Rock', 'Paper', 'Scissors'];
const EMOJIS = ['✊', '✋', '✌️'];

let model;

/**
 * Sets up the camera stream and configures the video element
 */
async function setupCamera() {
    try {
        // Try to get environment (rear) camera first, fall back to default
        const constraints = {
            video: { 
                facingMode: 'environment',
                width: { ideal: 224 },
                height: { ideal: 224 }
            }
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } catch (error) {
        console.error('Error accessing camera:', error);
        classNameElement.textContent = 'Camera access denied';
        throw error;
    }
}

/**
 * Loads the pre-trained TensorFlow.js model
 */
async function loadModel() {
    try {
        console.log('Loading model...');
        
        // Use `loadLayersModel` for models converted from Keras.
        // This is the correct function that matches our model format.
        model = await tf.loadLayersModel(MODEL_PATH);
        
        console.log('Model loaded successfully');
        
        // Warm up the model (this is good practice!)
        const warmUpTensor = tf.zeros([1, 224, 224, 3]);
        const prediction = model.predict(warmUpTensor);
        prediction.dispose(); // Also dispose the warmup prediction
        warmUpTensor.dispose();
        
        console.log('Model warmed up.');
        return model;

    } catch (error) {
        console.error('Error loading model:', error);
        // Display the error to the user if an element exists
        if (classNameElement) {
            classNameElement.textContent = 'Error loading model';
        }
        throw error;
    }
}

/**
 * Main prediction loop - processes video frames and classifies gestures
 */
async function predict() {
    if (!model) return;
    
    // Capture frame from video and convert to tensor
    const frame = tf.browser.fromPixels(video);
    
    // Preprocess the tensor for the model
    const resized = tf.image.resizeBilinear(frame, [224, 224]);
    const normalized = resized.toFloat();
    const batched = normalized.expandDims(0);
    
    // Run prediction
    const prediction = model.predict(batched);
    const classIndex = prediction.argMax(-1).dataSync()[0];
    
    // Update UI with prediction results
    emojiElement.textContent = EMOJIS[classIndex];
    classNameElement.textContent = CLASS_NAMES[classIndex];
    classNameElement.classList.remove('loading');
    
    // Clean up tensors to prevent memory leaks
    tf.dispose([frame, resized, normalized, batched, prediction]);
    
    // Continue the prediction loop
    requestAnimationFrame(predict);
}

/**
 * Main application initialization function
 */
async function run() {
    try {
        await setupCamera();
        await video.play();
        
        await loadModel();
        
        // Start the prediction loop
        predict();
        
    } catch (error) {
        console.error('Error initializing application:', error);
    }
}

// Start the application when the page loads
document.addEventListener('DOMContentLoaded', run);
