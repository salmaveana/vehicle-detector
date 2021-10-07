# USAGE
# python train.py --dataset path/to/dataset

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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle

# Modify parameters
LR = 0.005
BS = 32
EPOCHS = 100
VERBOSE = 10
MODEL_PATH = "output/car.h5"
MLB_PATH = "output/car.pickle"
CM_PATH = "output/car_cm.png"
PLOT_PATH = "output/car_plot.png"

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# Initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
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
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

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
model = SmallerVGGNet.build(width=32, height=32, depth=3, classes=2)
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
f = open(MLB_PATH, "wb")
f.write(pickle.dumps(mlb))
f.close()

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["sedan", "suv"]))

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
ax.xaxis.set_ticklabels(['sedan', 'suv'])
ax.yaxis.set_ticklabels(['sedan', 'suv'])
plt.savefig(CM_PATH)

# Plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(PLOT_PATH)
plt.show()

