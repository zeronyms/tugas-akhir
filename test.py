import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av

st.set_page_config(page_title="Kamera Live", layout="centered")
st.title("📷 Kamera Real-Time (WebRTC)")


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Tidak diproses apa-apa, hanya menampilkan frame
        return frame


webrtc_streamer(
    key="simple-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
