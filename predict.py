# USAGE:
# python predict.py

# Enable/disable debugging logs (0,1,2,3)
# 0 -> all, 3 -> none
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the necessary packages
from cnn.preprocessing import ImageToArrayPreprocessor
from cnn.preprocessing import SimplePreprocessor
from cnn.datasets import SimpleDatasetLoader
from keras.models import load_model
from pathlib import Path
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2

# Add compatibility with TF2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number", required=True, help="# of test")
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-id", "--imagedim", required=True, help="Image dimensions (pre-processing")
ap.add_argument("-ts", "--testsize", required=True, help="Size of the test partition")
ap.add_argument("-td", "--testdata", required=True, help="Dataset for testing")
args = vars(ap.parse_args())

# Modify parameters / assign arguments
TEST = int(args["number"])
DATASET = os.path.basename(os.path.normpath(args["dataset"]))
WIDTH = int(args["imagedim"])
HEIGHT = int(args["imagedim"])
TEST_SIZE = float(args["testsize"])

# Create output strings
model_path = str(TEST) + "_model_" + DATASET + "_" + str(WIDTH) + "_" + str(int(TEST_SIZE*100)) +  ".h5"
mlbi_path = str(TEST) + "_mlbi_" + DATASET + "_" + str(WIDTH) + "_" + str(int(TEST_SIZE*100)) + ".pickle"

# Configure output dir
BASE_OUTPUT = "output/" + DATASET + "/"
Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)

# Configure input paths
TEST_DS = args["testdata"]
MODEL_PATH = BASE_OUTPUT + model_path
MLBI_PATH = BASE_OUTPUT + mlbi_path

# Load object detector and label binarizer from disk
print("[INFO] Loading network...")
model = load_model(MODEL_PATH)
mlb = pickle.loads(open(MLBI_PATH, "rb").read())

# Load images
print("[INFO] Loading images...")
imagePaths = np.array(list(paths.list_images(TEST_DS)))

# Initialize the image preprocessors
sp = SimplePreprocessor(WIDTH, HEIGHT)
iap = ImageToArrayPreprocessor()

# Load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# Loop over the sample images
for (k, imagePath) in enumerate(imagePaths):

    # Load the example image
    image = cv2.imread(imagePath)
    output = imutils.resize(image, width=400)
    
    print("\n")
    print("Image: ", imagePath)

    # Make predictions on the images
    print("[INFO] classifying image...")
    preds = model.predict(data)[k]
    idxs = np.argsort(preds)[::-1][:2]

    # Loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(idxs):
        # Build the label and draw the label on the image
        label = "{}: {:.2f}%".format(mlb.classes_[j], preds[j] * 100)
        cv2.putText(output, label, (10, (i * 30) + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, preds):
        print("{}: {:.2f}%".format(label, p * 100))

    #cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (3, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display to screen
    cv2.imshow("Output", output)
    cv2.waitKey(0)
