import av
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(
    page_title="Face Orientation Detection",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
IMAGE_EXAMPLE = "images/photo-1544348817-5f2cf14b88c8.png"
IMAGE_TYPES = ["jpg", "jpeg", "png"]


class FaceMeshVideoProcessor:
    def __init__(self):
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_face_mesh(self, image, flip=True):
        if flip:
            image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_styles
                    .get_default_face_mesh_tesselation_style())
                self.drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_styles
                    .get_default_face_mesh_contours_style())
                self.drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_styles
                    .get_default_face_mesh_iris_connections_style())

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtx_r, mtx_q, qx, qy, qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Flip the image horizontally for a selfie-view display.
        return image

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.process_face_mesh(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def local_css(file_name):
    # Method for reading styles.css and applying necessary changes to HTML
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def choose_image():
    # st.sidebar.markdown('Upload your image ⬇')
    image_file = st.sidebar.file_uploader("", type=IMAGE_TYPES)

    if not image_file:
        text = """This is a detection example.
        Try your input from the left sidebar.
        """
        st.markdown(
            '<h6 align="center">' + text + "</h6>",
            unsafe_allow_html=True,
        )
        image_file = IMAGE_EXAMPLE
    else:
        st.sidebar.markdown(
            "__Image is uploaded successfully!__",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h4 align="center">Detection result</h4>',
            unsafe_allow_html=True,
        )

    pil_image = Image.open(image_file)
    st.image(pil_image, use_column_width=True)
    image = np.array(pil_image)
    image = FaceMeshVideoProcessor().process_face_mesh(image, flip=False)
    st.image(image, use_column_width=True)


def choose_webcam():
    st.sidebar.markdown('Click "START" to connect this app to a server')
    st.sidebar.markdown("It may take a minute, please wait...")
    webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=FaceMeshVideoProcessor,
        async_processing=True,
    )


def main():
    local_css("css/styles.css")
    st.markdown(
        '<h1 align="center">🤖 Face Orientation Detection</h1>',
        unsafe_allow_html=True,
    )
    st.set_option("deprecation.showfileUploaderEncoding", False)
    choice = st.sidebar.radio(
        "Select an input option:",
        ["Webcam", "Image"],
    )
    if choice == "Image":
        choose_image()

    if choice == "Webcam":
        choose_webcam()


if __name__ == "__main__":
    main()
