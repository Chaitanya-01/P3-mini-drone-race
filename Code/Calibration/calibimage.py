from djitellopy import Tello
import time
import cv2

# Tello drone code template:
# Check the connection to the drone
drone = Tello()
drone.connect()

# CHECK SENSOR READINGS------------------------------------------------------------------
print('Battery, ', drone.get_battery())

drone.streamon()
# drone.takeoff()
i = 0
while True:
    frame = drone.get_frame_read().frame
    # frame = cv2.resize(frame, (360, 240)) # Windows
    # frame = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_LINEAR) # Ubuntu
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    H, W, _ = frame.shape
    # cv2.imread(frame)
    filename = f"frame{i}.png"
    cv2.imshow('video', frame)
    key = cv2.waitKey(2) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        drone.move_forward(30)
    elif key == ord('s'):
        drone.move_back(30)
    elif key == ord('a'):
        drone.move_left(30)
    elif key == ord('d'):
        drone.move_right(30)
    elif key == ord('p'):
        print(f"Taking picture....\n")
        filename = f"frame{i}.png"
        cv2.imwrite(filename, frame)
    i += 1
# print(f"Landing....")
# drone.land()
