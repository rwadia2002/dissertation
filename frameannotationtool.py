import os
import cv2
import random

def display_images(image_paths):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        print("Current image:", image_path)
        cv2.imshow('Image', image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        annotation = None
        if key == ord('1'):
            annotation = 'left'
        elif key == ord('2'):
            annotation = 'centre'
        elif key == ord('3'):
            annotation = 'right'
        elif key == ord('4'):
            annotation = 'unsure'

        if annotation:
            annotation_file = os.path.splitext(image_path)[0] + '.txt'
            if not os.path.exists(annotation_file):
                with open(annotation_file, 'w') as f:
                    f.write(annotation)

# Example usage:

root_directory = '/Users/rahulwadia/Downloads/imageframes'
image_paths = []
for subdir, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(subdir, file))
random.shuffle(image_paths)  # Shuffle the list of image paths
display_images(image_paths)


