import cv2
import numpy as np
import threading
from queue import Queue

mutex = threading.Lock()

class FrontCam(threading.Thread):

    def __init__(self):

        threading.Thread.__init__(self)

        self.capture = cv2.VideoCapture(0)
        self.frame_queue = Queue(maxsize=100)

    def run(self):
        self.frame_collector()

    def frame_collector(self):
        while True:
            _, frame = self.capture.read()

            # if cannot, we'll just get 'em next time, boys
            if mutex.acquire():
                if self.frame_queue.full() == False:
                    self.frame_queue.put(frame)                
                mutex.release()
    
    def get_frame(self):
        frame = None
        if mutex.acquire():
            if self.frame_queue.empty() == False:
                frame = self.frame_queue.get(0)
            mutex.release()

        return frame
    
    def stop_cam(self):
        self.capture.release()
        cv2.destroyAllWindows()

# this is just an example of how to import and setup the FrontCam class
# and how to get frames from it. It is meant to be imported as from vid_cam import FrontCam
def main():
    vc = FrontCam()
    vc.start()

    while True:
        f = vc.get_frame()
        if f is not None:
            cv2.imshow("Frame", f)
            cv2.waitKey(1)

main()