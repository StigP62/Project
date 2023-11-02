import cv2
import numpy as np
import os
import json
import queue
import threading

def load_config():
    default_config = {"min_val": 0, "max_val": 255}
    config_filename = "config.json"

    # If config file exists, try to load it
    if os.path.isfile(config_filename):
        try:
            with open(config_filename, "r") as file:
                config = json.load(file)

                # Validate the loaded configuration
                if 0 <= config["min_val"] <= 255 and 0 <= config["max_val"] <= 255:
                    return config
                else:
                    print("Invalid config values. Reverting to defaults.")
        except Exception as e:
            print(f"An error occurred while reading the config file: {e}")

    # Return default config if load failed
    return default_config

def save_config(min_val, max_val):
    config = {"min_val": min_val, "max_val": max_val}
    with open("config.json", "w") as file:
        json.dump(config, file)

class FrameProcessor(threading.Thread):
    def __init__(self, frame_queue, config, config_lock):
        super().__init__()
        self.frame_queue = frame_queue
        self.config = config
        self.config_lock = config_lock
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                # Convert to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                with self.config_lock:
                    min_val, max_val = self.config["min_val"], self.config["max_val"]

                # Apply mask based on current config
                mask = cv2.inRange(gray_frame, min_val, max_val)

                # Apply the Hough Line Transform on the mask to detect lines
                edges = cv2.Canny(mask, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)  # Adjust parameters as needed

                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing directly on the original frame

                cv2.imshow('original with lines', frame)  # Show the original frame with lines
                cv2.imshow('mask', mask)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

def main():
    config = load_config()
    config_lock = threading.Lock()

    frame_queue = queue.Queue()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return

    # Create trackbars and window
    cv2.namedWindow('settings')
    cv2.createTrackbar('min_val', 'settings', config["min_val"], 255, lambda x: None)
    cv2.createTrackbar('max_val', 'settings', config["max_val"], 255, lambda x: None)

    frame_processor = FrameProcessor(frame_queue, config, config_lock)
    frame_processor.start()

    while frame_processor.running:
        ret, frame = camera.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        frame_queue.put(frame)

        # Update the shared configuration with the current trackbar values
        with config_lock:
            config["min_val"] = cv2.getTrackbarPos('min_val', 'settings')
            config["max_val"] = cv2.getTrackbarPos('max_val', 'settings')

        # Small sleep to keep the main thread responsive for trackbar interactions
        cv2.waitKey(1)

    frame_processor.join()  # Wait for the processing thread to finish

    camera.release()
    cv2.destroyAllWindows()

    # Save the configuration upon exit
    save_config(config["min_val"], config["max_val"])

if __name__ == '__main__':
    main()
