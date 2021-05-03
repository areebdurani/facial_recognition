# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import shutil

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            pass #shutil.copy2(s, d)
            
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
if os.path.exists(args["encodings"]):
    with open(args["encodings"], "rb") as f:
        dct = pickle.load(f)
    #f.close()
    for ens in dct["encodings"]:
        knownEncodings.append(ens)
    for name in dct["names"]:
        knownNames.append(name)
print("previously registered images: ", len(knownNames))
f = open(args["encodings"], "wb")

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)


# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

f.write(pickle.dumps(data))
f.close()

for fol in os.listdir(args["dataset"]):
    folpath = args["dataset"] + "/" + fol
    if not os.path.exists("dataset/" + fol):
        shutil.copytree(folpath, "dataset/" + fol)
    shutil.rmtree(folpath)

