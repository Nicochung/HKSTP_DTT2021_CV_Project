import numpy as np
from tkinter import filedialog
import tkinter as tk
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import time
import cv2

# Import Model
MODEL_PATH = 'best_model_combine.h5'
FRUIT_LABEL = ["Fresh Apple", "Fresh Banana", "Fresh Orange", "Rotten Apple", "Rotten Banana", "Rotten Orange"]
model = tf.keras.models.load_model(MODEL_PATH)
MODEL_IMAGE_SIZE = (128, 128)

# For GUI
root = tk.Tk()
root.title('Fruit Predict/Detector')
root.geometry('550x400')
mediaFrame = tk.Frame(root).pack()

media = tk.Label(mediaFrame)
media.pack()


# Function for upload file and predict
def oas():
    img_path = filedialog.askopenfilename(title='Choose',
                                          filetypes=[
                                              ("jpeg files", "*.jpg"),
                                              ("png files", "*.png"),])

    cv2image = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (400, 300)), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    media.imgtk = imgtk
    media.configure(image=imgtk)
    test_image = image.load_img(img_path, target_size=MODEL_IMAGE_SIZE)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255 # Make sure to resize and rescale the image
    result = model.predict(test_image)
    max_index = np.argmax(result[0], axis=0)
    result_label.configure(text=f'This is predicted as {FRUIT_LABEL[max_index]}')


# Function for using webcam for real time prediction
def rtp():
    start_predict = False
    cnt = 0
    CNTTHRESH = 50
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    # Time is used to get the Frames Per Second (FPS)
    last_time = time.time()
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    # Get image resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Location parameters for the box and text
    BOX_OFFSET = 150
    START_X = int(width // 2 - BOX_OFFSET)
    START_Y = int(height // 2 - BOX_OFFSET)
    END_X = int(width // 2 + BOX_OFFSET)
    END_Y = int(height // 2 + BOX_OFFSET)
    PREDICTION_X = int(width // 2 - BOX_OFFSET)
    PREDICTION_Y = int(height // 2 - BOX_OFFSET)
    while True:
        _, frame = cap.read()
        boxed_img = frame[START_Y:END_Y, START_X:END_X]
        # Blur the background
        frame = cv2.blur(frame, (30, 30))
        # Rectangle marker
        r = cv2.rectangle(frame, (START_X, START_Y), (END_X, END_Y), (100, 50, 200), 3)
        # Replacing the blur boxed with original image
        frame[START_Y:END_Y, START_X:END_X] = boxed_img

        # Add a FPS label to image
        text = f"FPS: {int(1 / (time.time() - last_time))}"
        last_time = time.time()
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        if start_predict:
            test_image = cv2.resize(boxed_img, MODEL_IMAGE_SIZE)
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image /= 255   # Make sure to resize and rescale the image
            result = model.predict(test_image)
            max_index = np.argmax(result[0], axis=0)
            cv2.putText(frame, FRUIT_LABEL[max_index], (PREDICTION_X, PREDICTION_Y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Warming Up", (PREDICTION_X, PREDICTION_Y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Fruit Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not start_predict:
            cnt += 1
        if not start_predict and cnt > CNTTHRESH:
            start_predict = True

    cv2.destroyAllWindows()


b1 = tk.Button(root, text="Open Image and Predict", command=oas).pack()
b2 = tk.Button(root, text="Real Time Prediction from WebCam", command=rtp).pack()
result_label = tk.Label(root)
result_label.pack()
root.mainloop()
