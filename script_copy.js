// import { FilesetResolver, FaceDetector } from "./package/vision_bundle.mjs";
import { FilesetResolver, FaceDetector } from "./vision_package/vision_bundle.mjs";

let accelerator = "wasm"

function average(arr) {
    if (arr.length === 0) return 0;  // Prevent division by zero
    return arr.reduce((sum, value) => sum + value, 0) / arr.length;
  }


let facedetector;

async function initializeMediaPipe() {
    // "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    const startime = performance.now();
    const filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
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

async function setupCamera() {
    try {
        const video = document.getElementById('video');
        const constraints = {
            video: {
                width: { ideal: 360 },
                height: { ideal: 270 },
                frameRate: { ideal: 30 }
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

async function loadLiteModel() {
    // console.log('Model load start');
    try {
        // console.log('Start load lite');
        const startime = performance.now();
        const tfliteModel = await tflite.loadTFLiteModel("./model/lite/model_lite.tflite");
        const endtime = performance.now();
        const loadtime = Math.round(endtime -startime);
        const liteloadelement = document.getElementById("lite106_load")
        liteloadelement.textContent = loadtime;
        console.log(tfliteModel.inputs[0].shape); // e.g., [1, 3, 112, 112]
        console.log(tfliteModel.inputs[0].dtype);
        console.log(tfliteModel.inputs); // e.g., [1, 3, 112, 112]
        console.log(tfliteModel.outputs);
        console.log('Done load lite');
        return tfliteModel;
    } catch (error) {
        console.error("Error on load lite model", error);
    }
    
}


async function detectAndProcessFaces(video, face_detector, model_lite) {
    console.log("Start processing");
    let flag = true;
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
    const lite106Inference = document.getElementById('lite106_inference');
    // const onnx106Inference = document.getElementById('onnx106_inference');
    const tfjs106tensor = document.getElementById('tfjs106_tensor');
    // const onnx106tensor = document.getElementById('onnx106_tensor');
    const tfjs_onnx_different = document.getElementById('onnx_tfjs_different');
    const lite_tfjs_different = document.getElementById('lite_tfjs_different');
    let inference_avg = [];

    videoCanvas.width = video.videoWidth;
    videoCanvas.height = video.videoHeight;
    landmarksCanvas.width = video.videoWidth;
    landmarksCanvas.height = video.videoHeight;
    const scaleFactor = 112;
    const processingStartTime = performance.now();
    

    const processFrame = async () => {
        if (performance.now() - processingStartTime > 8 * 1000) {
            console.log("Stopping processing after 10 seconds");
            const average_inference = Math.round(average(inference_avg))
            tfjs_onnx_different.textContent = average_inference
            return;
        }
        
        try {
            // console.time("processing")
            videoCtx.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);
            const imageData = videoCtx.getImageData(0, 0, videoCanvas.width, videoCanvas.height);
            const startTimeFD = performance.now();
            const detections = await face_detector.detectForVideo(imageData, performance.now());
            const endTimeFD = performance.now();
            const fdinferenceTime = Math.round(endTimeFD - startTimeFD);
            const faces = detections.detections;
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


                    // const onnx_tensor_time = Math.round(endonnx_time - endtf_time);
                    const tfjs_tensor_time = Math.round(endtf_time - starttf_time);
                    tfjs106tensor.textContent = tfjs_tensor_time;
                    
                    // onnx106tensor.textContent = onnx_tensor_time;

                    // console.log('tjfs tensor', input.dataSync())
                    // console.log("onnx tensor", input_onnx_Tensor.cpuData)

                    // const feeds = {"input": input_onnx_Tensor};
                    // const startTimeonnx = performance.now();
                    // const onnx_pred = await onnxmodel.run(feeds);
                    // const endTimeonnx = performance.now();
                    // const onnxinferencetime = Math.round(endTimeonnx - startTimeonnx);
                    // onnx106Inference.textContent = onnxinferencetime;
                    // const outputTensor = onnx_pred.output;
                    // const normalized_landmarks_onnx = outputTensor.cpuData;
    
                    const startTimelite = performance.now();
                    const config = { batchSize: 1, verbose: true };
                    let inputs = {
                        'serving_default_input:0': tf.zeros([1, 3, 112, 112], 'float32')
                      };
                    // const outputs = model_lite.predict(inputs, config);

                    // const firstOutputData = outputs[0].dataSync().slice();
                    // console.log('First output:', firstOutputData);
                    
                    // const predictionsData = predictions_lite.dataSync().slice(); // Creates a new copy of the data
                    // predictions_lite.dispose(); // Dispose the tensor if it's no longer needed
                    // console.log(predictionsData);
                    // model_lite.cleanUp();

                                        
                    // const endTimelite = performance.now(); 
                    // const liteinferenceTime = Math.round(endTimelite - startTimelite);
                    // lite106Inference.textContent = liteinferenceTime;
                    
                    
                    // const normalized_landmarks_lite = await predictions_lite[0].data();

                    // const startTimetfjs = performance.now();
                    // const predictions = model.predict(input);
                    // const endTimetfjs = performance.now();
                    // const tfjsinferenceTime = Math.round(endTimetfjs - startTimetfjs);
                    // tfjs106Inference.textContent = tfjsinferenceTime;
                    // const normalized_landmarks_tfjs = await predictions[0].data();
                    

                    // inference_avg.push(Math.round(((tfjsinferenceTime - liteinferenceTime)/tfjsinferenceTime)*100))

                    // inference_avg.push(Math.round(((tfjsinferenceTime - onnxinferencetime)/tfjsinferenceTime)*100))
                    
                    // console.log("Output data tfjs:", normalized_landmarks_tfjs);
                    // console.log("Output data onnx:", normalized_landmarks_onnx);

                    const landmarks_tfjs = normalizeLandmarksArray(normalized_landmarks_tfjs);
                    // const landmarks_onnx = normalizeLandmarksArray(normalized_landmarks_onnx);
                    // console.log('tfjs', landmarks_tfjs)
                    // console.log('onnx', landmarks_onnx)
                    
                    const { valueleft, valueright } = EyeAspectRatio(landmarks_tfjs);
                    const mar = MouthAspectRatio(landmarks_tfjs)
                    const yaw = calculateYaw(landmarks_tfjs)
            

                    leftEarValueElement.textContent = valueleft.toFixed(2);
                    rightEarValueElement.textContent = valueright.toFixed(2);
                    marelement.textContent = mar.toFixed(2);
                    yawelement.textContent = yaw.toFixed(2);
                    const rescaledLandmarks = normalized_landmarks_tfjs.map(value => value * scaleFactor);
                    const finalLandmarks = rescaledLandmarks.map((value, index) => {
                        if (index % 2 === 0) {
                            // X coordinate
                            return value * scaleFactorX + boundingBox.originX;
                        } else {
                            // Y coordinate
                            return value * scaleFactorY+ boundingBox.originY;
                        }
                    });


                    const specificIndices = [0, 93, 94, 95, 96, 69, 53, 35, 36, 37 ,38 , 39, 42, 43, 44, 45, 46];
                    drawLandmarksindex(landmarksCanvasCtx, finalLandmarks, specificIndices);

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
    console.log("TFJS Backend", tf.getBackend());
    const video = await setupCamera();
    const model_lite = await loadLiteModel();
    const face_detector = await initializeMediaPipe();
    detectAndProcessFaces(video, face_detector, model_lite);
}
// main()
tf.setBackend(accelerator).then(() => main());