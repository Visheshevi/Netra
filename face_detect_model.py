##### Face DetectionModel #####


import numpy as np
import cv2
import os, errno
#import pathlib

face_cascade = cv2.CascadeClassifier('/media/visheshchanana/New Volume/Projects/Netra/Classifiers/haarcascade_frontalface.xml')
# faceCascade = cv2.CascadeClassifier(cascPath)
BASE_DIR = '/media/visheshchanana/New Volume/Projects/datasets/faces/ff'
BASE_DIR_2 = '/media/visheshchanana/New Volume/Projects/datasets/faceCollection'

def import_image(path):
	return cv2.imread(path,3)

## Display the image
def display_image(image,title):
	cv2.imshow(title,image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

## Utilizes OpenCV's inbuilt haar classifier to find out the face in the image(frontal face). Return the cropped face that would be required for the training
def haar_cascade(img):
	temp_image = img
	final_img = []
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
	face = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
	for (x, y, w, h) in face:
		temp_image = crop_image(img,x,y,w,h)
		final_img.append(temp_image)
    	# cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	return temp_image
	# return final_img

def haar_cascade_2(img):
	temp_image = img
	final_img = []
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
	face = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(1, 1),flags = cv2.CASCADE_SCALE_IMAGE)
	for (x, y, w, h) in face:
		temp_image = crop_image(img,x,y,w,h)
		final_img.append(temp_image)
    	# cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	# return temp_image
	return final_img

# def haar_cascade(img):
# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	image = face_cascade.detectMultiScale(gray, 1.3, 5)
# 	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# 	for (x,y,w,h) in faces:
# 	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# 	    roi_gray = gray[y:y+h, x:x+w]
# 	    roi_color = img[y:y+h, x:x+w]
# 	img = crop_image(img,x,y,w,h)
# 	return img

## Crops the image to the required measurements
def crop_image(img,x,y,w,h):
	return img[y+1:y+h+1, x+1:x+w+1]



def load_images_from_folder(path):
	import_image  = []
	for temp_img in os.listdir(path):
		temp_path = os.path.join(path,temp_img)
		# print(temp_path)
		# img = import_image(temp_path)
		img = cv2.imread(temp_path,3)
		if img is not None:
			import_image.append(img)
	return import_image

def create_temp_dir(folder_name):
	try:
		# temp_folder = BASE_DIR + '/' + folder_name
		print(os.getcwd())
		os.mkdir(folder_name)
	except OSError as e:
		if e.errno != errno.EEXIST:
			print ("Creation of the directory %s failed" % folder_name)
		else:
			print("Directory %s already exists" % folder_name)
	else:
		print ("Successfully created the directory %s " % folder_name)


def add_face_temp_folder(original_path, temp_path, name):
	transfer_image = []
	num = 1
	transfer_image = load_images_from_folder(original_path)
	for img in transfer_image:
		img = haar_cascade(img)
		image_name = name + '_' + str(num) + '.jpeg'
		num += 1
		cv2.imwrite(os.path.join(temp_path,image_name),img)
		cv2.waitKey(0)


def add_face_temp_folder_2(original_path, temp_path, name):
	transfer_image = []
	num = 1
	transfer_image = load_images_from_folder(original_path)
	# display_image(transfer_image[0],name)
	final_img = []
	final_img = haar_cascade_2(transfer_image[0])
	for img in final_img:
	# 	img = haar_cascade(img)
		image_name = name + '_' + str(num) + '.jpeg'
		num += 1
		cv2.imwrite(os.path.join(temp_path,image_name),img)
		cv2.waitKey(0)

def main():
	PANDYA_ORI = BASE_DIR + '/pandya'
	KOHLI_ORI = BASE_DIR + '/kohli'
	RAINA_ORI = BASE_DIR + '/raina'
	PANDYA_PATH = BASE_DIR + '/pandya_temp'
	KOHLI_PATH = BASE_DIR + '/kohli_temp'
	RAINA_PATH = BASE_DIR + '/raina_temp'
	
	# ABHI_DIR = BASE_DIR_2 + '/Abhi'

	full_images = []

	# add_face_temp_folder_2(ABHI_DIR, ABHI_DIR,'abhishek')

	create_temp_dir(PANDYA_PATH)
	create_temp_dir(KOHLI_PATH)
	create_temp_dir(RAINA_PATH)

	add_face_temp_folder(PANDYA_ORI, PANDYA_PATH, 'pandya')
	add_face_temp_folder(KOHLI_ORI, KOHLI_PATH, 'kohli')
	add_face_temp_folder(RAINA_ORI, RAINA_PATH, 'raina')
	

	# Alternate way to create  folder
	# pathlib.Path(PANDYA_PATH).mkdir(parents=True, exist_ok=True)

	# full_images = load_images_from_folder(PANDYA_PATH)
	# display_image(full_images[2],'Pandya')
	

	# print(full_images[0])
	# IMG_PATH = '/media/visheshchanana/New Volume/Images/for testing/side_face_1.jpeg'
	# image = import_image(IMG_PATH)
	# display_image(image,'Pandya')
	# image = haar_cascade(image)
	# display_image(image, 'Hardik Face')


if __name__== "__main__":
	main()
