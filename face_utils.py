import face_recognition
import numpy as np


def get_face_encoding(image_path: str):
    image = face_recognition.load_image_file(image_path)

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return None, "No face detected"

    if len(face_locations) > 1:
        return None, "Only one face allowed"

    encodings = face_recognition.face_encodings(image, face_locations)

    if len(encodings) == 0:
        return None, "Face encoding failed"

    encoding = encodings[0]
    encoding_string = ",".join(map(str, encoding))

    return encoding_string, "success"


def string_to_encoding(encoding_string: str):
    return np.array([float(value) for value in encoding_string.split(",")])