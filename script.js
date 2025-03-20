// import { FilesetResolver, FaceDetector } from "./package/vision_bundle.mjs";
import { FilesetResolver, FaceDetector } from "https://dc9911a6.h5-resource-test.pages.dev/vision_bundle.mjs";



async function preprocessFaceImage(faceImageData, targetWidth = 112, targetHeight = 112) {
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = targetWidth;
    offscreenCanvas.height = targetHeight;
    const offscreenCtx = offscreenCanvas.getContext('2d');

  
    // Create an ImageBitmap from the faceImageData (to handle scaling)
    const bitmap = await createImageBitmap(faceImageData);
    // Draw the bitmap onto the offscreen canvas (this scales the image)
    offscreenCtx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
  
    // Extract the resized image data (in RGBA order)
    const resizedImageData = offscreenCtx.getImageData(0, 0, targetWidth, targetHeight);
    const { data } = resizedImageData; // Uint8ClampedArray
    const numPixels = targetWidth * targetHeight;
  
    // Create a Float32Array to hold normalized pixel values.
    // We want NCHW layout: shape = [1, 3, targetHeight, targetWidth]
    // That means first all red values, then green, then blue.
    const float32Data = new Float32Array(3 * numPixels);
  
    // Loop over each pixel and extract R, G, and B, normalize by dividing by 255.
    for (let i = 0; i < numPixels; i++) {
      const baseIndex = i * 4;
      float32Data[i] = data[baseIndex] / 255.0;                    // Red channel
      float32Data[i + numPixels] = data[baseIndex + 1] / 255.0;      // Green channel
      float32Data[i + 2 * numPixels] = data[baseIndex + 2] / 255.0;  // Blue channel
      // We ignore the alpha channel (data[baseIndex + 3])
    }
  
    // Define the tensor shape: [batch, channels, height, width]
    const dims = [1, 3, targetHeight, targetWidth];
    const inputTensor = new ort.Tensor("float32", float32Data, dims);
  
    return inputTensor;
  }

let facedetector;

async function initializeMediaPipe() {
    const filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const startime = performance.now();
    facedetector = await FaceDetector.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: "./blaze_face_short_range.tflite",
        },
        runningMode: "VIDEO"
    });
    const endtime = performance.now();
    const loadtime = Math.round(endtime -startime);
    const fdloadelement = document.getElementById("facedetector_load")
    fdloadelement.textContent = loadtime;
    console.log("Face detector model loaded");
    return facedetector;
}

async function initializeONNXLandmarkModel() {
    // Load your ONNX model for landmarks
    const sessionOption = { 
        executionProviders: ['wasm'], 
        executionMode: "parallel", 
        enableCpuMemArena: true,
        enableGraphCapture: false,
        enableMemPattern: true,
        enableProfiling: false,
        interOpNumThreads: 2,
        intraOpNumThreads: 2,
        graphOptimizationLevel: "all"
     };
    const startime = performance.now();
    const session = await ort.InferenceSession.create('./model/onnx/v2/own_pfld_lite_sim_0.0242.onnx', sessionOption);
    const endtime = performance.now();
    const loadtime = Math.round(endtime -startime);
    const onnxloadelement = document.getElementById("onnx106_load")
    onnxloadelement.textContent = loadtime;
    console.log("ONNX landmark model loaded");
    console.log("ONNX session", session);
    console.log("inputname", session.inputNames);
    console.log("outputname", session.outputNames);
    return session;
  }

async function setupCamera() {
    try {
        const video = document.getElementById('video');
        const constraints = {
            video: {
                width: { ideal: 720 },
                height: { ideal: 640 },
                frameRate: { ideal: 60 }
            }
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                console.log("Actual Video Width:", video.videoWidth);
                console.log("Actual Video Height:", video.videoHeight);
                console.log("Actual Frame Rate:", video.srcObject.getVideoTracks()[0].getSettings().frameRate);
                resolve(video);
            };
        });
    } catch (error) {
        console.error("Error accessing the webcam:", error);
    }
}

async function loadModel() {
    console.log('Model load start');
    const startime = performance.now();
    const model = await tf.loadGraphModel('./model/tensorflowjs/model.json');
    const endtime = performance.now();
    const loadtime = Math.round(endtime -startime);
    const tfjsloadelement = document.getElementById("tfjs106_load")
    tfjsloadelement.textContent = loadtime;
    console.log('Model loaded');

    // Inspect the model input details
    model.inputs.forEach(input => {
        console.log(`Input name: ${input.name}`);
        console.log(`Input shape: ${input.shape}`);
        console.log(`Input dtype: ${input.dtype}`);
    });

    // Inspect the model output details
    model.outputs.forEach(output => {
        console.log(`Output name: ${output.name}`);
        console.log(`Output shape: ${output.shape}`);
        console.log(`Output dtype: ${output.dtype}`);
    });

    return model;
}

async function detectAndProcessFaces(video, face_detector, model, onnxmodel) {
    console.log("Start processing");
    const videoCanvas = document.getElementById('videoCanvas');
    const videoCtx = videoCanvas.getContext('2d');
    const landmarksCanvas = document.getElementById('landmarksCanvas');
    const landmarksCanvasCtx = landmarksCanvas.getContext('2d');
    const leftEarValueElement = document.getElementById('leftEarValue');
    const rightEarValueElement = document.getElementById('rightEarValue');
    const marelement = document.getElementById('MAR');
    const yawelement = document.getElementById('YAW');
    const fdInference = document.getElementById('facedetector_inference');
    const tfjs106Inference = document.getElementById('tfjs106_inference');
    const onnx106Inference = document.getElementById('onnx106_inference');
    const tfjs106tensor = document.getElementById('tfjs106_tensor');
    const onnx106tensor = document.getElementById('onnx106_tensor');
    const tfjs_onnx_different = document.getElementById('onnx_tfjs_different');

    videoCanvas.width = video.videoWidth;
    videoCanvas.height = video.videoHeight;
    landmarksCanvas.width = video.videoWidth;
    landmarksCanvas.height = video.videoHeight;
    const scaleFactor = 112;
    const onnx_data_options = {
        dataType : "float32", 
        ImageFormat: "RGB", 
        resizedHeight: 112,
        resizedWidth: 112, 
        norm: {
            mean: 255                                
          },
        ImageTensorLayout: "NCHW"
    };

    const processFrame = async () => {
        try {
            // console.time("processing")
            videoCtx.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);
            const imageData = videoCtx.getImageData(0, 0, videoCanvas.width, videoCanvas.height);
            const startTimeFD = performance.now();
            const detections = await face_detector.detectForVideo(imageData, performance.now());
            const endTimeFD = performance.now();
            const fdinferenceTime = Math.round(endTimeFD - startTimeFD);
            const faces = detections.detections;
            // console.log(faces);
            // console.log("FD Inference time:", fdinferenceTime, "ms");
            fdInference.textContent = fdinferenceTime;

            if (faces.length > 0) {
                for (const detection of faces) {
                    const boundingBox = detection.boundingBox;

                    // Extract the face region from the bounding box
                    const faceImageData = videoCtx.getImageData(
                        boundingBox.originX, 
                        boundingBox.originY, 
                        boundingBox.width, 
                        boundingBox.height
                    );
                    // console.log(faceImageData)
                    const scaleFactorX = boundingBox.width / 112;
                    const scaleFactorY = boundingBox.height / 112;
                    
                    const starttf_time = performance.now();
                    // Convert the image data to a tensor and preprocess it to match the model's input shape
                    let input = tf.tidy(() => {
                        let tempInput = tf.browser.fromPixels(faceImageData).toFloat();  // [360, 270, 3]
                        tempInput = tf.image.resizeBilinear(tempInput, [112, 112]);  // Resize to [112, 112, 3]
                        tempInput = tempInput.div(tf.scalar(255));  // Normalize the pixel values to [0, 1]
                        tempInput = tempInput.transpose([2, 0, 1]);  // Rearrange dimensions to [3, 112, 112]
                        return tempInput.expandDims(0);  // Add batch dimension to get [1, 3, 112, 112]
                    });
                    const endtf_time = performance.now();
                    
                    const input_onnx_Tensor = await preprocessFaceImage(faceImageData, 112, 112);
                    const endonnx_time = performance.now();


                    const onnx_tensor_time = Math.round(endonnx_time - endtf_time);
                    const tfjs_tensor_time = Math.round(endtf_time - starttf_time);
                    tfjs106tensor.textContent = tfjs_tensor_time;
                    onnx106tensor.textContent = onnx_tensor_time;

                    const feeds = {"input": input_onnx_Tensor};
                    const startTimeonnx = performance.now();
                    const onnx_pred = await onnxmodel.run(feeds);
                    const endTimeonnx = performance.now();
                    const onnxinferencetime = Math.round(endTimeonnx - startTimeonnx);
                    onnx106Inference.textContent = onnxinferencetime;
                    const outputTensor = onnx_pred.output;

                    // To view its underlying data, access its cpuData property
                    const outputData = outputTensor.cpuData;
                    console.log("Output data onnx:", outputData);

                    const startTimetfjs = performance.now();
                    const predictions = model.predict(input);
                    const endTimetfjs = performance.now();
                    const tfjsinferenceTime = Math.round(endTimetfjs - startTimetfjs);
                    tfjs106Inference.textContent = tfjsinferenceTime;

                    tfjs_onnx_different.textContent = Math.round(((tfjsinferenceTime - onnxinferencetime)/tfjsinferenceTime)*100);
                    

                    const normalized_landmarks = await predictions[0].data();
                    console.log("Output data tfjs:", normalized_landmarks);


                    const landmarks = normalizeLandmarksArray(outputData);
                    // console.timeEnd("processing")
                    
                    const { valueleft, valueright } = EyeAspectRatio(landmarks);
                    const mar = MouthAspectRatio(landmarks)
                    const yaw = calculateYaw(landmarks)

                    leftEarValueElement.textContent = valueleft.toFixed(2);
                    rightEarValueElement.textContent = valueright.toFixed(2);
                    marelement.textContent = mar.toFixed(2);
                    yawelement.textContent = yaw.toFixed(2);
                    // const rescaledLandmarks = outputData.map(value => value * scaleFactor);
                    // const finalLandmarks = rescaledLandmarks.map((value, index) => {
                    //     if (index % 2 === 0) {
                    //         // X coordinate
                    //         return value * scaleFactorX + boundingBox.originX;
                    //     } else {
                    //         // Y coordinate
                    //         return value * scaleFactorY+ boundingBox.originY;
                    //     }
                    // });


                    // const specificIndices = [0, 93, 94, 95, 96];
                    // drawLandmarksindex(landmarksCanvasCtx, finalLandmarks, specificIndices);



                }
            }
        } catch (error) {
            console.error("Error during frame processing:", error);
        }
        
        requestAnimationFrame(processFrame);
    };

    processFrame();
}


function drawLandmarksindex(ctx, landmarks, indices = []) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = "red";
    ctx.strokeStyle = "red";  // Set the color for the landmarks
    ctx.lineWidth = 1;  // Set the line width for the landmarks

    indices.forEach(index => {
        const x = landmarks[index * 2];
        const y = landmarks[index * 2 + 1];
        ctx.beginPath();
        ctx.arc(x, y, 1, 0, 2 * Math.PI);  // Draw a circle for each landmark
        ctx.fill();
    });
}

function drawLandmarks(ctx, landmarks) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = "red";
    ctx.strokeStyle = "red";  // Set the color for the landmarks
    ctx.lineWidth = 1;  // Set the line width for the landmarks

    for (let i = 0; i < landmarks.length; i += 2) {
        const x = landmarks[i];
        const y = landmarks[i + 1];
        ctx.beginPath();
        ctx.arc(x, y, 1, 0, 2 * Math.PI);  // Draw a circle for each landmark
        ctx.fill();
    }
}

function distance(point1, point2) {
    return Math.sqrt(Math.pow(point1[0] - point2[0], 2) + Math.pow(point1[1] - point2[1], 2));
}

function normalizeLandmarksArray(normalized_landmarks) {
    const landmarks = [];
    for (let i = 0; i < normalized_landmarks.length; i += 2) {
        landmarks.push({
            x: normalized_landmarks[i],
            y: normalized_landmarks[i + 1]
        });
    }
    return landmarks;
}


function EyeAspectRatio(landmarks) {
    // Left eye landmarks
    let p1 = [landmarks[53].x, landmarks[53].y];
    let p2 = [landmarks[54].x, landmarks[54].y];
    let p3 = [landmarks[57].x, landmarks[57].y];
    let p4 = [landmarks[58].x, landmarks[58].y];
    let p5 = [landmarks[59].x, landmarks[59].y];
    let p6 = [landmarks[62].x, landmarks[62].y];

    // Right eye landmarks
    let m1 = [landmarks[64].x, landmarks[64].y];
    let m2 = [landmarks[65].x, landmarks[65].y];
    let m3 = [landmarks[67].x, landmarks[67].y];
    let m4 = [landmarks[69].x, landmarks[69].y];
    let m5 = [landmarks[70].x, landmarks[70].y];
    let m6 = [landmarks[73].x, landmarks[73].y];

    // Calculate distances for the right eye
    let part1 = distance(m2, m6);
    let part2 = distance(m3, m5);
    let part3 = distance(m1, m4);
    let valueright = (part1 + part2) / (2 * part3);

    // Calculate distances for the left eye
    let value1 = distance(p2, p6);
    let value2 = distance(p3, p5);
    let value3 = distance(p1, p4);
    let valueleft = (value1 + value2) / (2 * value3);

    return { valueleft, valueright };
}

function MouthAspectRatio(landmarks) {
    // Left eye landmarks
    let p1 = [landmarks[33].x, landmarks[33].y];
    let p2 = [landmarks[47].x, landmarks[47].y];
    let p3 = [landmarks[49].x, landmarks[49].y];
    let p4 = [landmarks[40].x, landmarks[40].y];
    let p5 = [landmarks[50].x, landmarks[50].y];
    let p6 = [landmarks[52].x, landmarks[52].y];



    // Calculate distances for the left eye
    let value1 = distance(p2, p6);
    let value2 = distance(p3, p5);
    let value3 = distance(p1, p4);
    let mar = (value1 + value2) / (2 * value3);


    return mar ;
}

function calculateYaw(landmarks) {
    // Extract the required landmarks
    let p1 = [landmarks[95].x, landmarks[95].y];
    let p2 = [landmarks[25].x, landmarks[25].y];
    let p3 = [landmarks[9].x, landmarks[9].y];

    // Calculate distances
    let value1 = Math.abs(p1[0] - p2[0]);
    let value2 = Math.abs(p2[0] - p3[0]);

    // Calculate the final value
    let value = value1 / value2;
    return value ;
}

async function main() {
    console.log("Backend", tf.getBackend());
    const video = await setupCamera();
    const model = await loadModel();
    const onnxmodel = await initializeONNXLandmarkModel();
    const face_detector = await initializeMediaPipe();
    detectAndProcessFaces(video, face_detector, model, onnxmodel);
}
// main()
tf.setBackend('wasm').then(() => main());