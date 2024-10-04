import cv2
import numpy as np  # Changed numpy to np for consistency

# Import the prediction function
from prediction import predict  # Ensure the prediction function is correctly imported

# Define the ROI coordinates
y_start, x_start, roi_height, roi_width = 10, 10, 350, 350  # Corrected the roi_width to 240 to match the original code

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize the sentence variable
sentence = ""
threshold = 0.5  # Define a threshold for prediction confidence
window_name = "Sign Language Recognition"  # Window name for displaying the video

while True:
    ret, frame = cap.read()
    if not ret:  # Fixed the condition to check if frame is not captured
        print("No frame captured")
        continue

    cv2.rectangle(frame, (x_start, y_start), (x_start + roi_width, y_start + roi_height), (255, 0, 0), 3)

    # Crop the region of interest (ROI) from the frame
    img1 = frame[y_start:y_start + roi_height, x_start: x_start + roi_width]
    
    # Convert the ROI to YCrCb color space and apply Gaussian blur
    img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    blur = cv2.GaussianBlur(img_ycrcb, (11, 11), 0)

    # Define skin color range in YCrCb
    skin_ycrcb_min = np.array((0, 138, 67))
    skin_ycrcb_max = np.array((255, 173, 133))

    # Create a binary mask where white represents the skin region
    mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)

    # Define a kernel for dilation
    kernel = np.ones((2, 2), dtype=np.uint8)

    # Apply dilation to the mask
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Extract the hand region using the mask
    new = cv2.bitwise_and(img1, img1, mask=mask)
    
    # Display the mask and the extracted hand region
    cv2.imshow("mask", mask)
    cv2.imshow("new", new)

    hand_bg_rm = new
    # hand = img1

    # Capture keyboard input
    c = cv2.waitKey(1) & 0xFF

    if c == ord('c') or c == ord('C'):
        sentence = ""

    if c == ord('d') or c == ord('D'):
        sentence = sentence[:-1]

    if c == ord('m') or c == ord('M'):
        sentence += " "

    # If a valid hand area is cropped
    # if hand.shape[0] != 0 and hand.shape[1] != 0:
    #     conf, label = predict(hand_bg_rm)
    #     if conf >= threshold:
    #         cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255))

    #     if c == ord('n') or c == ord('N'):
    #         sentence += label

    if hand_bg_rm.shape[0] != 0 and hand_bg_rm.shape[1] != 0:
        conf, label = predict(hand_bg_rm)
        print(f"Prediction: {label}, Confidence: {conf}")
        if conf >= threshold:
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255))

        if c == ord('n') or c == ord('N'):
            sentence += label

    # Display the sentence on the frame
    cv2.putText(frame, sentence, (50, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255))
    
    # Display the frame
    cv2.imshow(window_name, frame)

    # Exit the loop if the ESC key is pressed
    if c == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy
# from prediction import *

# # Define the ROI coordinates
# y_start, x_start,roi_height, roi_width = 10, 350, 225, 590

# cap = cv2.VideoCapture(0)
# while True:
#     ret,frame = cap.read()
#     if ret is None:
#         print("No frame captured")
#         continue
#     cv2.rectangle(frame, (x_start, y_start), (x_start+roi_width, y_start+roi_height), (255, 0, 0), 3)

#     img1 = frame[y_start:y_start+roi_height, x_start: x_start+roi_width]
#     img_ycrcb = cv2.cvtColor(img1,cv2.COLOR_BGR2YCR_CB)
#     blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)

#     skin_ycrcb_min = np.array((0,138,67))
#     skin_ycrcb_max = np.array((255,173,133))

#     mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)

#     kernel = np.ones((2,2),dtype=np.unit8)

#     mask =cv2.dilate(mask,kernel,iterations=1)

#     new = cv2.bitwise_and(img1,img1,mask=mask)
#     cv2.imshow("mask",mask)
#     cv2.imshow("new",new)
#     hand_bg_rm = new
#     hand = img1

#     c = cv2.waitKey(1) & 0xff

#     if c==ord('c') or c==ord('C'):
#         sentence = ""
    
#     if c==ord('d') or c==ord('D'):
#         sentence = sentence[:-1]
    
#     if c==ord('m') or c==ord('M'):
#         sentence += " "

#     #if valid hand area is cropped
#     if hand.shape[0]!=0 and hand.shape[1]!=0:
#         conf, label = predict(hand_bg_rm)
#         if conf>=threshold:
#             cv2.putText(frame,label, (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (0,0,255))

#         if c == ord('n') or c == ord('N'):
#             sentence+=label

#     cv2.putText(frame, sentence,(50,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (0,0,255))
#     cv2.imshow(window_name, frame)

#     if c==27:
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()

# cap.release()
# cv2.destroyAllWindows()






















# import cv2
# import numpy as np
# import tensorflow as tf
# model = tf.keras.models.load_model('cnn_model.h5')

# # Load the trained model
# # model = load_model('cnn_model.h5')

# # Define the labels for sign language gestures
# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
#           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# # Coordinates of the Region of Interest (ROI) in the frame
# x, y, w, h = 420, 140, 200, 200

# # Function to preprocess the image for prediction
# def preprocess_image(img):
#     # Resize the image to match the model's expected input size
#     img_resized = cv2.resize(img, (100, 100))
#     # Convert RGB to grayscale
#     img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
#     # Normalize the pixel values
#     img_normalized = img_gray / 255.0
#     # Reshape the image to add batch dimension and single channel
#     img_reshaped = img_normalized.reshape(1, 100, 100, 1)
#     return img_reshaped

# # Function to display the predicted label on the image
# def display_prediction(img, prediction):
#     # Put the predicted label on the frame
#     cv2.putText(img, labels[prediction], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
#     # Draw a rectangle around the ROI
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
#     # Show the frame with the prediction
#     cv2.imshow('Sign Language Recognition', img)

# # Function to capture video from webcam and perform prediction
# def predict_from_webcam():
#     # Open the webcam
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()  # Read a frame from the webcam
        
#         if not ret:
#             break  # If frame is not captured, break the loop

#         # Preprocess the image
#         frame = cv2.flip(frame, 1)  # Flip the frame horizontally
#         img_cropped = frame[y:y+h, x:x+w]  # Crop the frame to the ROI
#         img_processed = preprocess_image(img_cropped)  # Preprocess the cropped image

#         # Perform prediction
#         prediction = np.argmax(model.predict(img_processed), axis=-1)
        
#         # Display the prediction on the frame
#         display_prediction(frame, prediction[0])

#         # Check for 'q' key to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Run the prediction function
# predict_from_webcam()





# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np


# cap = cv2.VideoCapture(0)
# model = load_model("trained_model.h5")

# variables = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
#              'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# # Define the ROI coordinates
# top, right, bottom, left = 10, 350, 225, 590

# # Open the video capture
# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()

# # Capture the initial frame
# ret, first_frame = cap.read()
# if not ret or first_frame is None:
#     print("Error: Could not read the initial frame.")
#     cap.release()
#     exit()

# # Flip the initial frame and extract the ROI
# first_gray = cv2.flip(first_frame, 1)
# roi = first_gray[top:bottom, right:left]

# # Initialize the string for concatenated output
# string = ""

# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         print("Error: Could not read a frame.")
#         break

#     gray_frame = cv2.flip(frame, 1)
#     roi2 = gray_frame[top:bottom, right:left]
#     cv2.rectangle(gray_frame, (right, top), (left, bottom), (0, 0, 255), 2)

#     # Image Processing Steps
#     diff = cv2.absdiff(roi, roi2)
#     diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#     diff = cv2.GaussianBlur(diff, (11, 11), 0)
#     _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

#     cv2.imshow("Threshold", diff)

#     key = cv2.waitKey(1)
#     if key == 27 or key == ord('q'):  # use the ESC key
#         break
#     if key == ord('r'):
#         roi = gray_frame[top:bottom, right:left]
#         string = ""

#     img = cv2.resize(diff, (100, 100))
#     img = tf.cast(img, tf.float32)  # Convert to data type

#     img_arr = np.array(img)
#     t2 = np.expand_dims(img_arr, axis=-1)
#     t2 = np.expand_dims(t2, axis=0)
#     p = model.predict(t2)
#     p2 = np.argmax(p)
#     out = variables[p2]

#     font = cv2.FONT_HERSHEY_COMPLEX
#     cv2.putText(gray_frame, out, (22, 34), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(gray_frame, string, (22, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(gray_frame, "r: Reset;  s: Append;  q: Quit", (22, 470), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

#     cv2.imshow("Frame", gray_frame)
#     if key == ord('s'):
#         string += out

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np

# cap = cv2.VideoCapture(0)
# model = load_model("cnn_model.h5")

# variables = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
#              'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# # Define the ROI coordinates
# top, right, bottom, left = 10, 350, 225, 590

# # Open the video capture
# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()

# # Capture the initial frame
# ret, first_frame = cap.read()
# if not ret or first_frame is None:
#     print("Error: Could not read the initial frame.")
#     cap.release()
#     exit()

# # Flip the initial frame and extract the ROI
# first_gray = cv2.flip(first_frame, 1)
# roi = first_gray[top:bottom, right:left]

# # Initialize the string for concatenated output
# string = ""

# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         print("Error: Could not read a frame.")
#         break

#     gray_frame = cv2.flip(frame, 1)
#     roi2 = gray_frame[top:bottom, right:left]
#     cv2.rectangle(gray_frame, (right, top), (left, bottom), (0, 0, 255), 2)

#     # Image Processing Steps
#     diff = cv2.absdiff(roi, roi2)
#     diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#     diff = cv2.GaussianBlur(diff, (11, 11), 0)
#     _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

#     cv2.imshow("Threshold", diff)

#     key = cv2.waitKey(1)
#     if key == 27 or key == ord('q'):  # use the ESC key
#         break
#     if key == ord('r'):
#         roi = gray_frame[top:bottom, right:left]
#         string = ""

#     img = cv2.resize(diff, (100, 100))
#     img = tf.cast(img, tf.float32)  # Convert to data type

#     img_arr = np.array(img)
#     t2 = np.expand_dims(img_arr, axis=-1)
#     t2 = np.expand_dims(t2, axis=0)
    
#     # Print shape for debugging
#     print(f"Input shape to model: {t2.shape}")

#     p = model.predict(t2)
#     print(f"Predictions: {p}")  # Print the prediction for debugging
#     p2 = np.argmax(p)
#     out = variables[p2]

#     font = cv2.FONT_HERSHEY_COMPLEX
#     cv2.putText(gray_frame, out, (22, 34), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(gray_frame, string, (22, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(gray_frame, "r: Reset;  s: Append;  q: Quit", (22, 470), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

#     cv2.imshow("Frame", gray_frame)
#     if key == ord('s'):
#         string += out

# cap.release()
# cv2.destroyAllWindows()
