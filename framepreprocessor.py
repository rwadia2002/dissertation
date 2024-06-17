import os
import dlib
import cv2
# Load the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/rahulwadia/Downloads/shape_predictor_68_face_landmarks_GTX.dat")  # Provide path to the pretrained landmark predictor
import random
rootdirectory = '/Users/rahulwadia/Downloads/'
frames_folder = os.path.join(rootdirectory, 'imageframes')
os.makedirs(frames_folder, exist_ok=True)
#'/Users/rahulwadia/Downloads/5518996/sleep dataset/25 zgt/rgb.avi'
subjectno=0
import os

def facedetect(filename, number):
    frameno = 0
    cap = cv2.VideoCapture(filename)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Failed to read the first frame")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    significant_movement_threshold = 30
    subjectdirec = os.path.join(frames_folder, str(number))  # Corrected subject directory path
    os.makedirs(subjectdirec, exist_ok=True)
    prev_frame_saved = False  # Flag to track if the previous frame was saved

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        threshold = 30
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum(cv2.contourArea(contour) for contour in contours)

        if total_area > significant_movement_threshold:
            cv2.putText(frame, 'Significant Movement Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.imshow('Frame', frame)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            elapsed_time_seconds = frameno / cap.get(cv2.CAP_PROP_FPS)
            hours = int(elapsed_time_seconds // 3600)
            minutes = int((elapsed_time_seconds % 3600) // 60)
            seconds = int(elapsed_time_seconds % 60)
            timestamp = f'{hours:02}:{minutes:02}:{seconds:02}'

            if len(faces) > 0:
                x, y, w, h = faces[0]
                cropped_face = frame[y:y + h, x:x + w]
                # Save the frame only if it's not identical to the previous one
                if not prev_frame_saved:
                    print("Cropped face saved successfully")
                    edges = cv2.Canny(gray, 50, 150)

                    # Find contours in the image
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Draw contours on the original image
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

                    # Display the image
                    cv2.imshow("Contours", frame)
                    cv2.imwrite(f'{subjectdirec}/frame{frameno}.jpg', cropped_face)

                    # Write the timestamp or point of change to a text file
                    with open(f'{subjectdirec}/timestamps.txt', 'a') as f:
                        f.write(f'Frame {frameno}: {timestamp}\n')

                    prev_frame_saved = True
                else:
                    print("Identical frame skipped")

        else:
            cv2.imshow('Frame', frame)
            prev_frame_saved = False  # Reset the flag if the current frame doesn't meet the threshold

        prev_gray = gray
        frameno += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def process_files_in_directory(directory, subjectno=0):
    for root, dirs, files in os.walk(directory):
        # Process files in the current directory
        for file in files:
            file_path = os.path.join(root, file)
            # Skip files that are not AVI files
            if not file.endswith('.avi'):
                continue
            print("Processing file:", file_path)
            facedetect(file_path, subjectno)
            subjectno += 1

        # Recursively process subdirectories
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            process_files_in_directory(subdir_path, subjectno)


# Example usage:

process_files_in_directory('/Users/rahulwadia/Downloads/5518996/sleep dataset')