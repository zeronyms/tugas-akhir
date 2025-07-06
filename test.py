import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import time
import requests
import base64
from PIL import Image
from io import BytesIO
import queue
from streamlit_autorefresh import st_autorefresh

# Auto-refresh setiap 5 detik agar kolom kanan terus cek hasil
st.set_page_config(layout="wide")
st.title("📷 Deteksi Tingkat Fokus Mahasiswa Realtime")

st_autorefresh(interval=5000, key="refresh")

# <-- PERUBAHAN 1: Mengubah rasio kolom menjadi [2, 1]
# Kolom kiri (kamera) akan 2x lebih lebar dari kolom kanan (hasil).
col1, col2 = st.columns([2, 1])

# State awal
if "latest_label" not in st.session_state:
    st.session_state.latest_label = "Menunggu Prediksi"
    st.session_state.latest_face = None

if "latest_conf" not in st.session_state:
    st.session_state.latest_conf = 0.0


# VideoProcessor class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_sent = time.time()
        self._output = queue.Queue()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if now - self.last_sent > 5:
            self.last_sent = now
            print("📤 [INFO] Mengirim frame ke API...")

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, img_encoded = cv2.imencode(".jpg", img, encode_param)

            files = {"image": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}

            try:
                res = requests.post(
                    "https://tugas-akhir-production-8f2e.up.railway.app/predict",
                    files=files,
                )
                print("📬 [INFO] Status response:", res.status_code)

                if res.status_code == 200:
                    data = res.json()
                    label = "Fokus" if data["label"] == 1 else "Tidak Fokus"
                    print(
                        f"✅ [HASIL] Label: {label}, Confidence: {data['prediction']:.2f}"
                    )

                    self._output.put(
                        {
                            "label": label,
                            "confidence": float(data["prediction"]),
                            "face_image": data["face_image"],
                        }
                    )
                else:
                    print("⚠️ [WARN] API response:", res.text)
            except Exception as e:
                print(f"❌ [ERROR] Gagal koneksi ke API: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Kamera (kiri)
with col1:
    st.header("📹 Kamera Mahasiswa")
    webrtc_ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": 1280, "height": 720},
            "audio": False,
        },
        async_processing=True,
    )

# Hasil prediksi (kanan)
with col2:
    st.header("📊 Hasil Prediksi")

    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        try:
            msg = webrtc_ctx.video_processor._output.get(timeout=1.0)
            if msg:
                st.session_state.latest_label = msg.get("label", "Tidak diketahui")
                st.session_state.latest_conf = msg.get("confidence", 0.0)
                try:
                    face_bytes = base64.b64decode(msg["face_image"])
                    st.session_state.latest_face = Image.open(BytesIO(face_bytes))
                except Exception as e:
                    print(f"❌ [ERROR] Gagal decode wajah: {e}")
                    st.session_state.latest_face = None
        except queue.Empty:
            # Tidak melakukan apa-apa, cukup gunakan state terakhir
            pass

    # <-- PERUBAHAN 2: Menggunakan st.metric untuk tampilan yang lebih baik
    # Ini akan menggantikan st.markdown untuk tampilan yang lebih rapi.

    # Buat dua kolom di dalam kolom kanan untuk menata metrik
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Status", st.session_state.latest_label)
    metric_col2.metric("Keyakinan", f"{st.session_state.latest_conf:.2%}")

    st.write("---")  # Garis pemisah

    if st.session_state.latest_face:
        st.image(
            st.session_state.latest_face,
            caption="Wajah yang Terdeteksi",
            use_container_width=True,
        )
    else:
        st.info("Belum ada wajah yang terdeteksi dari server.")
