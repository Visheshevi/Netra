#### Face Detection ####


import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('/media/visheshchanana/New Volume/Projects/Netra/Classifiers/haarcascade_frontalface.xml')

def import_image(path):
	return cv2.imread(path,3)

## Display the image
def display_image(image,title):
	cv2.imshow(title,image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

## Utilizes OpenCV's inbuilt haar classifier to find out the face in the image(frontal face). Return the cropped face that would be required for the training
def haar_cascade(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	image = face_cascade.detectMultiScale(gray, 1.3, 5)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	img = crop_image(img,x,y,w,h)
	return img

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


def main():
	IMAGES_PATH = '/media/visheshchanana/New Volume/Images/for testing/pandya'
	full_images = []
	full_images = load_images_from_folder(IMAGES_PATH)
	display_image(full_images[2],'Pandya')
	# print(full_images[0])
	# IMG_PATH = '/media/visheshchanana/New Volume/Images/for testing/side_face_1.jpeg'
	# image = import_image(IMG_PATH)
	# display_image(image,'Pandya')
	# image = haar_cascade(image)
	# display_image(image, 'Hardik Face')


if __name__== "__main__":
	main()


# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename))
#         if img is not None:
#             images.append(img)
#     return images