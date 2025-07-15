
import time

LastTime = time.time()
def calculateTime():
    global LastTime
    currentTime = time.time()
    elapsedTime = currentTime - LastTime
    LastTime = currentTime
    return elapsedTime
