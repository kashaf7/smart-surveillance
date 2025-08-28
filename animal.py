from ultralytics import YOLO
import cv2
import math
import time
import pygame  # For sound playback
from gtts import gTTS  # Google Text to Speech
import pywhatkit as kit  # For sending WhatsApp messages
import os

# Constants
SIREN_SOUND_PATH = "C:\\Users\\syeda\\OneDrive\\Desktop\\smartsurvellance using yolo animal dectection\\alarm.mp3"
MODEL_PATH = "yolov8.pt"
WHATSAPP_NUMBER = "+918431328607"  # Replace with your WhatsApp number
NOTIFICATION_COOLDOWN = 30  # Seconds
FRAME_SIZE = (640, 480)
CONFIDENCE_THRESHOLD = 50  # Percentage
MIN_DETECTION_TIME = 5  # Seconds

# Initialize pygame mixer for sound
pygame.mixer.init()

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam.")

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

# Class names for detection
classnames = [
    "antelope", "bear", "cheetah", "human", "coyote", "crocodile", "deer", "elephant",
    "flamingo", "fox", "giraffe", "gorilla", "hedgehog", "hippopotamus", "hornbill",
    "horse", "hummingbird", "hyena", "kangaroo", "koala", "leopard", "lion", "meerkat",
    "mole", "monkey", "moose", "okapi", "orangutan", "ostrich", "otter", "panda",
    "pelecaniformes", "porcupine", "raccoon", "reindeer", "rhino", "rhinoceros",
    "snake", "squirrel", "swan", "tiger", "turkey", "wolf", "woodpecker", "zebra",
]

# Define a set of harmful animals
harmful_animals = {"lion", "bear", "crocodile"}
last_notification_time = 0

# Dictionary to track detection times for each animal
detection_times = {}


def play_voice_message(message):
    """Function to play a voice message using gTTS."""
    try:
        tts = gTTS(text=message, lang="en")
        audio_file = "voice_message.mp3"
        tts.save(audio_file)

        # Play the audio using pygame
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing voice message: {e}")


def send_whatsapp_message(animal_detected):
    """Function to send WhatsApp messages at the current time."""
    try:
        # Get the current time
        current_time = time.localtime()
        hour = current_time.tm_hour
        minute = current_time.tm_min + 1  # Send the message 1 minute from now

        # Ensure minute is within 0-59
        if minute >= 60:
            minute -= 60
            hour = (hour + 1) % 24

        # Prepare the message
        message = f"ALERT! Harmful animal detected: {animal_detected}. Stay vigilant and take immediate precautions!!"
        print(f"Sending WhatsApp alert for {animal_detected} at {hour}:{minute}...")

        # Send the WhatsApp message
        kit.sendwhatmsg(WHATSAPP_NUMBER, message, hour, minute)
        print("WhatsApp message sent successfully!")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")


def main():
    global last_notification_time

    # Main loop for video capture and detection
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        # Perform inference with YOLO
        results = model(frame)

        # Process bounding boxes and display results
        for result in results:
            boxes = result.boxes  # Get detection results
            for box in boxes:
                confidence = box.conf[0] * 100  # Confidence in percentage
                class_index = int(box.cls[0])  # Get class index

                if confidence > CONFIDENCE_THRESHOLD and 0 <= class_index < len(classnames):
                    animal_detected = classnames[class_index]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get bounding box coordinates

                    # Display bounding box and class label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw rectangle
                    cv2.putText(
                        frame,
                        f"{animal_detected} {math.ceil(confidence)}%",
                        (x1 + 8, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )  # Draw label

                    # Check if the detected animal is harmful
                    if animal_detected in harmful_animals:
                        current_time = time.time()

                        # Track detection time for the animal
                        if animal_detected not in detection_times:
                            detection_times[animal_detected] = current_time

                        # Check if the animal has been detected for at least 5 seconds
                        if current_time - detection_times[animal_detected] >= MIN_DETECTION_TIME:
                            # Trigger alerts only if the cooldown period has passed
                            if current_time - last_notification_time > NOTIFICATION_COOLDOWN:
                                # Play siren sound
                                try:
                                    pygame.mixer.music.load(SIREN_SOUND_PATH)
                                    pygame.mixer.music.play()
                                    time.sleep(5)  # Wait for the siren to finish
                                except Exception as e:
                                    print(f"Error playing siren sound: {e}")

                                # Capture the image of the detected animal
                                try:
                                    image_filename = f"{animal_detected}_{int(current_time)}.jpg"
                                    cv2.imwrite(image_filename, frame)  # Save the frame as an image
                                    print(f"Image saved: {image_filename}")
                                except Exception as e:
                                    print(f"Error saving image: {e}")

                                # Prepare and play voice message
                                voice_message = f"Alert! {animal_detected} detected. Sending alert to admin. Be careful and take immediate precautions."
                                play_voice_message(voice_message)

                                # Send WhatsApp message instantly
                                send_whatsapp_message(animal_detected)

                                last_notification_time = current_time  # Update the notification time

                                # Reset detection time for the animal
                                detection_times.pop(animal_detected, None)
                    else:
                        # Reset detection time if the animal is no longer detected
                        detection_times.pop(animal_detected, None)

        # Display the frame with detected animals
        cv2.imshow("Animal Detection", frame)

        # Press 'Esc' key to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:  # ASCII 27 is the Escape key
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()