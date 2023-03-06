import av
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from processing import process_face_mesh

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


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process_face_mesh(img)
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

    PIL_image = Image.open(image_file)
    st.image(PIL_image, use_column_width=True)
    image = np.array(PIL_image)
    image = process_face_mesh(image, flip=False)
    st.image(image, use_column_width=True)


def choose_webcam():
    st.sidebar.markdown('Click "START" to connect this app to a server')
    st.sidebar.markdown("It may take a minute, please wait...")
    webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
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
