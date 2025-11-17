**Role:** You are an expert Frontend Developer specializing in creating performant, mobile-first web applications that integrate with machine learning models using TensorFlow.js.

**Task:** Develop a simple and clean mobile-first web application that uses a device's camera to classify hand gestures for the game "Rock, Paper, Scissors" in real-time. The application will load a pre-trained TensorFlow.js model and display the prediction continuously.

**Instruction:**
You must generate two files: `index.html` and `app.js`.

**For `index.html`:**
1.  Create a modern, responsive HTML5 structure.
2.  Include the TensorFlow.js library from a CDN (`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js`).
3.  Add a `<meta name="viewport" ...>` tag to ensure proper scaling on mobile devices.
4.  The body should contain:
    *   A simple heading like "Rock, Paper, Scissors AI".
    *   A `<video>` element, styled to be a centered square (e.g., 300px by 300px) that fills its container. It must have `autoplay`, `muted`, and `playsinline` attributes.
    *   A `<div>` element below the video to act as a container for the prediction result. This container should have a placeholder text like "Loading Model...".
    *   Inside the result container, include a `<span>` for the predicted emoji (e.g., '✊', '✋', '✌️') and another `<span>` for the class name text (e.g., "Rock").
5.  Apply basic CSS within a `<style>` tag to center the content, make it look clean, and style the prediction text to be large and clear.

**For `app.js`:**
1.  Define constants for the `video` element, the emoji/text result spans, the model path (`'./rps_web_model/model.json'`), and the class names (`['Rock', 'Paper', 'Scissors']` with corresponding emojis).
2.  Create an `async` function `setupCamera()` to request access to the user's camera. It should try to get the 'environment' (rear) camera first, but fall back to the default camera if not available.
3.  Create an `async` function `loadModel()` that loads the TensorFlow.js Layers Model from the specified path.
4.  Create an `async` function `predict()` which will be the main application loop:
    *   Use `tf.browser.fromPixels()` to capture the current frame from the video.
    *   **Crucially, preprocess the tensor:** Resize it to `(224, 224)`, ensure it's float32, and `expandDims()` to create a batch of 1.
    *   Run `model.predict()` on the processed tensor.
    *   Use `prediction.argMax(-1).dataSync()[0]` to get the index of the highest probability class.
    *   Update the `innerText` of the result spans with the correct emoji and class name.
    *   Dispose of all created tensors (`frame`, `resized`, etc.) using `tf.dispose()` to prevent memory leaks.
    *   Use `requestAnimationFrame(predict)` to create a smooth, efficient loop.
5.  Create a main `async` function `run()` that calls `setupCamera()`, then `loadModel()`, and finally starts the `predict()` loop. Call `run()` to start the application.
6.  Add comments to explain the key parts: model loading, tensor preprocessing, prediction, and memory management (`tf.dispose`).

**Format:**
*   Provide the content for `index.html` inside a single code block.
*   Provide the content for `app.js` inside a separate, single code block.
*   The code should be complete, well-commented, and immediately usable.
