import cv2
import numpy as np
import os
import json
import queue
import threading

def load_config():
    default_config = {
        "min_val": 0, 
        "max_val": 255,
        "hough_threshold": 50,
        "min_line_length": 10,
        "max_line_gap": 5,
        "rho": 1.0  # default value is 1.0, within the allowed range
    }
    config_filename = "config.json"

    if os.path.isfile(config_filename):
        try:
            with open(config_filename, "r") as file:
                config = json.load(file)
                # Validate the loaded configuration values
                if all(key in config for key in default_config):
                    return config
                else:
                    print("Invalid config values. Reverting to defaults.")
        except Exception as e:
            print(f"An error occurred while reading the config file: {e}")

    return default_config

def save_config(config):
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
                    min_val = self.config["min_val"]
                    max_val = self.config["max_val"]
                    hough_threshold = self.config["hough_threshold"]
                    min_line_length = self.config["min_line_length"]
                    max_line_gap = self.config["max_line_gap"]
                    rho = self.config["rho"]

                mask = cv2.inRange(gray_frame, min_val, max_val)

                edges = cv2.Canny(mask, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, rho, np.pi/180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.imshow('original with lines', frame)
                cv2.imshow('mask', mask)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

def main():
    config = load_config()
    config_lock = threading.Lock()

    frame_queue = queue.Queue() 
    # Camera build in camera: [camera = cv2.VideoCapture(0)] external camera: [camera = cv2.VideoCapture(2)]
    camera = cv2.VideoCapture(2)

    if not camera.isOpened():
        print("Error: Cameracould not be opened.")
        return

    cv2.namedWindow('settings')
    cv2.createTrackbar('min_val', 'settings', config["min_val"], 255, lambda x: None)
    cv2.createTrackbar('max_val', 'settings', config["max_val"], 255, lambda x: None)
    cv2.createTrackbar('threshold', 'settings', config["hough_threshold"], 255, lambda x: None)
    cv2.createTrackbar('min_line_length', 'settings', config["min_line_length"], 100, lambda x: None)
    cv2.createTrackbar('max_line_gap', 'settings', config["max_line_gap"], 50, lambda x: None)
    
    # Adjusting for floating point 'rho' handling
    initial_rho_scaled = int(config["rho"] * 100)  # Scaling the float for trackbar compatibility
    cv2.createTrackbar('rho', 'settings', initial_rho_scaled, 1000, lambda x: None)  # Values from 1 to 1000 (0.01 to 10.0 in float)

    frame_processor = FrameProcessor(frame_queue, config, config_lock)
    frame_processor.start()

    while frame_processor.running:
        ret, frame = camera.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        frame_queue.put(frame)

        with config_lock:
            config["min_val"] = cv2.getTrackbarPos('min_val', 'settings')
            config["max_val"] = cv2.getTrackbarPos('max_val', 'settings')
            config["hough_threshold"] = cv2.getTrackbarPos('threshold', 'settings')
            config["min_line_length"] = cv2.getTrackbarPos('min_line_length', 'settings')
            config["max_line_gap"] = cv2.getTrackbarPos('max_line_gap', 'settings')
            
            # Getting the scaled integer value from trackbar and converting it back to float
            rho_scaled = cv2.getTrackbarPos('rho', 'settings')
            config["rho"] = max(0.01, rho_scaled / 100.0)  # Ensure it doesn't go below the absolute minimum

        cv2.waitKey(1)

    frame_processor.join()
    camera.release()
    cv2.destroyAllWindows()

    save_config(config)

if __name__ == '__main__':
    main()