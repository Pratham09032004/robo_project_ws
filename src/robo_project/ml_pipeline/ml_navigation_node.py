import os
import time

from infer import predict_pose


image_folder = os.path.expanduser("~/robo_project_ws/src/dataset/front")

image_files = sorted(os.listdir(image_folder))


for image_name in image_files:

    image_path = os.path.join(image_folder, image_name)

    x, z = predict_pose(image_path)

    print(f"\nImage: {image_name}")

    print(f"Predicted Position -> X: {x:.2f}, Z: {z:.2f}")

    if x > 0.5:

        command = "TURN RIGHT"

    elif x < -0.5:

        command = "TURN LEFT"

    else:

        command = "MOVE FORWARD"

    print(f"Navigation Command: {command}")

    time.sleep(0.5)
