import av
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(
    page_title="Face Mesh Detector",
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

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
        # Flip the image horizontally for a selfie-view display.
        return cv2.flip(image, 1) if flip else image

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
        '<h1 align="center">🤖 Face Mesh Detection</h1>',
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
