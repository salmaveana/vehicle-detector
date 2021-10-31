# USAGE
# python train.py --dataset path/to/dataset

# Enable/disable debugging logs (0,1,2,3)
# 0 -> all, 3 -> none
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import scikit-learn Libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Import local libraries
from cnn.smallervggnet import SmallerVGGNet
from cnn.preprocessing import ImageToArrayPreprocessor
from cnn.preprocessing import SimplePreprocessor
from cnn.datasets import SimpleDatasetLoader

# Import remaining libraries
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from imutils import paths
from pathlib import Path
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number", required=True, help="# of test")
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-id", "--imagedim", required=True, help="Image dimensions (pre-processing")
ap.add_argument("-ts", "--testsize", required=True, help="Size of the test partition")
ap.add_argument("-lr", "--learnrate", required=True, help="Initial learning rate")
ap.add_argument("-bs", "--batchsize", required=True, help="Batch size")
ap.add_argument("-e", "--epochs", required=True, help="Number of epochs")
args = vars(ap.parse_args())

# Modify parameters / assign arguments
TEST = int(args["number"])
DATASET = os.path.basename(os.path.normpath(args["dataset"]))
WIDTH = int(args["imagedim"])
HEIGHT = int(args["imagedim"])
TEST_SIZE = float(args["testsize"])
LR = float(args["learnrate"])
BS = int(args["batchsize"])
EPOCHS = int(args["epochs"])
VERBOSE = 10

# LR to exponential
lr = "{:e}".format(LR)

# Create output strings
model_path = str(TEST) + "_model_" + DATASET + "_" + str(WIDTH) + "_" + str(int(TEST_SIZE*100)) +  ".h5"
mlbi_path = str(TEST) + "_mlbi_" + DATASET + "_" + str(WIDTH) + "_" + str(int(TEST_SIZE*100)) + ".pickle"
cmat_path = str(TEST) + "_cmat_" + DATASET + "_" + str(WIDTH) + "_" + str(int(TEST_SIZE*100)) + ".png"
plot_path = str(TEST) + "_plot_" + DATASET + "_" + str(WIDTH) + "_" + str(int(TEST_SIZE*100)) + ".png"
pred_path = str(TEST) + "_pred_" + DATASET + "_" + str(WIDTH) + "_" + str(int(TEST_SIZE*100)) + ".csv"
conf_path = str(TEST) + "_conf_" + DATASET + "_" + str(WIDTH) + "_" + str(int(TEST_SIZE*100)) + ".txt"

# Configure output dir
BASE_OUTPUT = "output/" + DATASET + "/"
Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)

# Configure output paths
MODEL_PATH = BASE_OUTPUT + model_path
MLBI_PATH = BASE_OUTPUT + mlbi_path
CMAT_PATH = BASE_OUTPUT + cmat_path
PLOT_PATH = BASE_OUTPUT + plot_path
PRED_PATH = BASE_OUTPUT + pred_path
CONF_PATH = BASE_OUTPUT + conf_path

# Save configuration variables
config_dictionary = {
        "test" : TEST,
        "dataset_name" : DATASET,
        "image_dimensions" : WIDTH,
        "test_size" : TEST_SIZE,
        "initial_learning_rate" : LR,
        "batch_size" : BS,
        "number_of_epochs" : EPOCHS
        }
f = open(CONF_PATH, "w")
str = repr(config_dictionary)
f.write("initial_configuration = " + str + "\n")
f.close()

# Initialize the image preprocessors
sp = SimplePreprocessor(WIDTH, HEIGHT)
iap = ImageToArrayPreprocessor()

# Load images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Load dataset from disk
# Scale the raw pixel intensities (range [0, 1])
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# Loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# Partition data into training and testing 
# 75% of data for training, 25% of data for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=TEST_SIZE, random_state=42)

# Convert the labels from integers to vectors
#trainY = LabelBinarizer().fit_transform(trainY)
#testY = LabelBinarizer().fit_transform(testY)

# If there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
if len(mlb.classes_) == 2:
	labels = to_categorical(labels)

# Initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(learning_rate=LR)
model = SmallerVGGNet.build(width=WIDTH, height=HEIGHT, depth=3, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Print network summary
model.summary()

# train the network
print("[INFO] training network...")
H = model.fit(
        trainX, trainY, batch_size=BS, 
        validation_data=(testX, testY), 
        epochs=EPOCHS, verbose=VERBOSE)

# Save the model to disk
print("[INFO] serializing network...")
model.save(MODEL_PATH, save_format="h5")

# Save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(MLBI_PATH, "wb")
f.write(pickle.dumps(mlb))
f.close()

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["camioneta", "sedan"])
print(report)
report_dict = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["camioneta", "sedan"], output_dict=True)
df = pandas.DataFrame(report_dict).transpose()
df.to_csv(PRED_PATH)

# Create Confusion Matrix
print("\n[INFO] Confusion Matrix...")
cm = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
print(cm)

# Set labels, title and ticks for confusion matrix
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Greens')  # annot=True to annotate cells
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['camioneta', 'sedan'])
ax.yaxis.set_ticklabels(['camioneta', 'sedan'])
plt.savefig(CMAT_PATH)

# Plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(PLOT_PATH)
plt.show()

