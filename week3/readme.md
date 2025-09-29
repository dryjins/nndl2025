
Role: You are a senior front‑end engineer and ML instructor building a browser‑only TensorFlow.js MNIST demo for students.

Context: Build a GitHub Pages–deployable web app that trains a CNN classifier on MNIST CSV files that ARE HOSTED IN THE SAME REPO (same origin). The CSVs have no header; each row is label (0–9) followed by 784 pixel values. Load via fetch from relative paths in the repo (e.g., ./data/mnist_train.csv, ./data/mnist_test.csv), normalize pixels to , run entirely client‑side with TF.js and tfjs‑vis for charts, and implement FILE‑BASED Save/Load for the model (download and re‑upload). Do not use IndexedDB or external origins.

Instruction: Output exactly three fenced code blocks in this order, labeled “index.html”, “data-loader.js”, and “app.js”, with no extra prose.

Instruction — index.html:
- Include CDNs: <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script> and <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>; add minimal CSS for a responsive layout and a preview strip.
- UI controls: Load Data, Train, Evaluate, Test 5 Random, Save Model (Download), Load Model (From Files), Reset. Provide two file inputs for model loading (model.json and weights.bin).
- Dataset path inputs default to ./data/mnist_train.csv and ./data/mnist_test.csv (relative paths in this repo); show a small note: “Host CSVs inside this GitHub Pages repo to avoid CORS; use raw fetch on relative URLs only.”
- Sections: Data Status, Training Logs with “Toggle Visor”, Metrics (overall accuracy + charts), Random 5 Preview (row of canvases + predicted labels), Model Info (layers/params).
- Defer‑load data-loader.js then app.js; add a one‑line Pages note (Settings → Pages → main, root).

Instruction — data-loader.js:
- Implement loadTrain(), loadTest(), getRandomTestBatch(k=5). Fetch CSVs from relative paths (e.g., './data/mnist_train.csv'); parse with tf.data.csv() (hasHeader=false) or manual parsing fallback; map first column to label, remaining 784 to pixels.
- Normalize /255, reshape to [N,28,28,1], one‑hot labels depth 10; provide configurable train/val split (e.g., 90/10). Cache tensors in memory and expose counts and a helper to draw 28×28 to canvas.

Instruction — app.js:
- CNN: Conv2D(32, 3, relu, same) → Conv2D(64, 3, relu, same) → MaxPool2D(2) → Dropout(0.25) → Flatten → Dense(128, relu) → Dropout(0.5) → Dense(10, softmax); compile with adam, categoricalCrossentropy, metrics=['accuracy'].
- Training: epochs input (default 5–10), batchSize 64–128; use tfjs‑vis fitCallbacks to plot loss/val_loss and acc/val_acc live. Report duration and best val accuracy.
- Evaluation: compute test accuracy; render confusion matrix heatmap and per‑class accuracy bar chart in Visor; print overall accuracy.
- Random 5 preview: sample 5 test images; render horizontally as canvases (scaled up) with predicted labels under each, green/red for correct/incorrect.
- Save (file‑based only): await model.save('downloads://mnist-cnn') to download model.json + weights.bin.
- Load (file‑based only): read two files from file inputs and load via tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile])); replace the active model, call model.summary(), rebind buttons.
- Memory safety: tf.tidy around temporary ops; dispose old models/tensors when replaced; try/catch with user alerts.

Formatting:
- Produce only three fenced code blocks labeled exactly “index.html”, “data-loader.js”, and “app.js”; use browser‑side JS only; keep comments in English; do not include any text outside the code blocks.
