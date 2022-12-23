import mmap
import face_recognition
import os
import glob
import numpy as np
import time

def encode_new_faces(image_path, output_folder):
    """
    Detects new images in the specified image path, encodes the faces in the images,
    and stores the encoded faces in the specified output folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use a set to store the names of already processed image files
    processed_images = set()

    while True:
        # Get a list of all image files in the image path
        image_paths = glob.glob(os.path.join(image_path, "*"))

        # Encode and store the new faces
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            output_path = os.path.join(output_folder, base_name + ".npy")
            # Skip already encoded faces
            if not os.path.exists(output_path):
                with open(image_path, "rb") as f:
                    # Map the file into memory using mmap
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    image = np.frombuffer(mm, dtype=np.uint8)
                encoding = face_recognition.face_encodings(image)[0]
                np.save(output_path, encoding)
                processed_images.add(base_name)

        # Sleep for 5 minutes before checking for new images again
        time.sleep(300)

# Create a thread for encoding new faces
thread = threading.Thread(target=encode_new_faces, args=(image_path, output_folder))

# Start the thread
thread.start()
