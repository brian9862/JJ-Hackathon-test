import { createRequire } from 'module';
const require = createRequire(import.meta.url);

//Tensorflow
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as mobilenet from '@tensorflow-models/mobilenet';

const classifier = knnClassifier.create()
const webcamElement = document.getElementById("webcam")

//Node.js 
const fs = require('fs');
const path = require('path');

//?
let net

async function app() {
  
  net = await mobilenet.load()

  const webcam = await tf.data.webcam(webcamElement)
    //function adds examples to ML classifier that will be trained using the MobileNet model.

  const addExample = async (classId) => { 
    //function captures an image from the webcam. Object is assigned as "img"

    const img = await webcam.capture()

    const activation = net.infer(img, true) //create a directory and iterate over lots of images

    classifier.addExample(activation, classId)

    img.dispose()
  }

  //test code
  const addExampleFromDirectory = async (className) => {
    const directoryPath = `C:\Users\brian\wkspaces\JJ Hackathon\ML Images`; // Replace with the actual path to the directory
  
    // Assign class ID based on className
    let classId;
    switch (className) {
      case 'Recycle':
        classId = 0;
        break;
      case 'Garbage':
        classId = 1;
        break;
      case 'Composte':
        classId = 2;
        break;
      default:
        throw new Error(`Invalid className: ${className}`);
    }
  
    // Get a list of all files in the directory
    const fileNames = await fs.promises.readdir(directoryPath);
  
    // Load each image file as a tensor and pass it to net.infer()
    for (const fileName of fileNames) {
      const imagePath = path.join(directoryPath, fileName);
      const buffer = await fs.promises.readFile(imagePath);
      const image = tf.node.decodeImage(buffer, 3);
      const activation = net.infer(image, true);
      classifier.addExample(activation, classId);
      image.dispose();
    }
  }
  //test code end
  
  
  document.getElementById("Recycle").addEventListener("click", () => addExample(0))
  document.getElementById("Garbage").addEventListener("click", () => addExample(1))
  document.getElementById("Composte").addEventListener("click", () => addExample(2))

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture()

      const activation = net.infer(img, "conv_preds")

      const result = await classifier.predictClass(activation)

      const classes = ["Recycle", "Garbage", "Composte"]

      document.getElementById("console").innerText = `
                prediction: ${classes[result.label]}\n
                probabilty: ${result.confidences[result.label]}
            `

      img.dispose()
    }

      await tf.nextFrame()
    }
  }

app()