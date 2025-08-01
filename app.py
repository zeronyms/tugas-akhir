import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from twilio.rest import Client
import av
import cv2
import time
import requests
import base64
from PIL import Image
from io import BytesIO
import queue
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# === Konfigurasi halaman ===
st.set_page_config(layout="wide", page_title="Deteksi Fokus Mahasiswa")
st.title("Prototipe Dashboard Deteksi Tingkat Fokus Mahasiswa")
st_autorefresh(interval=5000, key="data_refresh")

# === Inisialisasi State ===
if "latest_label" not in st.session_state:
    st.session_state.latest_label = "Menunggu..."
    st.session_state.latest_conf = 0.0
    st.session_state.latest_face = None
    st.session_state.focus_history = []

if "measuring" not in st.session_state:
    st.session_state.measuring = False
    st.session_state.measure_values = []
    st.session_state.avg_focus = None

# === Ambil konfigurasi ICE server dari Twilio ===
TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]


@st.cache_resource
def get_twilio_ice_servers():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    token = client.tokens.create()
    return {"iceServers": token.ice_servers}


rtc_config = get_twilio_ice_servers()


# === Pemrosesan video ===
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_sent = time.time() - 4
        self._output_queue = queue.Queue()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if now - self.last_sent > 5:
            self.last_sent = now
            _, img_encoded = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            files = {"image": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}

            try:
                res = requests.post(
                    # "https://tugas-akhir-production-8f2e.up.railway.app/predict",
                    "http://127.0.0.1:5000/predict",
                    files=files,
                    timeout=10,
                )
                if res.status_code == 200:
                    data = res.json()
                    label = "Fokus" if data.get("label") == 1 else "Tidak Fokus"
                    confidence = float(data.get("prediction", 0.0))
                    face_image_b64 = data.get("face_image")

                    self._output_queue.put(
                        {
                            "label": label,
                            "confidence": confidence,
                            "face_image": face_image_b64,
                        }
                    )
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Gagal koneksi ke API: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# === Layout Utama ===
col1, col2 = st.columns([2, 1.2])

# === KOLOM KIRI: Kamera & Kontrol ===
with col1:
    st.subheader("🔴 Live Kamera")

    webrtc_ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": 1280, "height": 720},
            "audio": False,
        },
        rtc_configuration=rtc_config,
        async_processing=True,
    )

    with st.container(border=True):
        st.subheader("Menghitung Rata-rata Fokus")
        st.write("**Tekan tombol mulai untuk memulai penghitungan**")

        # Letakkan tombol di dalam kontainer yang sama
        col_start, col_stop, col_status = st.columns([1, 1, 3])

        with col_start:
            if st.button("Mulai", use_container_width=True):
                st.session_state.measuring = True
                st.session_state.measure_values = []  # Reset data sebelumnya
                st.session_state.avg_focus = None
                st.toast("Pengukuran dimulai!")

        with col_stop:
            if st.button("Stop", use_container_width=True):
                if st.session_state.measuring:
                    st.session_state.measuring = False
                    if st.session_state.measure_values:
                        # Hitung rata-rata
                        st.session_state.avg_focus = sum(
                            st.session_state.measure_values
                        ) / len(st.session_state.measure_values)
                        st.toast("Pengukuran selesai!")
                    else:
                        st.warning("Tidak ada data fokus yang terkumpul.")
                else:
                    st.info("Pengukuran belum dimulai.")

        st.write("---")  # Garis pemisah

        # --- Tampilan Hasil Rata-rata Fokus ---
        if st.session_state.avg_focus is not None:
            st.metric(
                label="Hasil Rata-rata Fokus", value=f"{st.session_state.avg_focus:.2%}"
            )
            # Tambahkan tombol untuk mereset
            if st.button("Ulangi Pengukuran", use_container_width=True):
                st.session_state.measuring = False
                st.session_state.measure_values = []
                st.session_state.avg_focus = None
                st.rerun()  # Refresh halaman untuk memulai dari awal

        elif st.session_state.measuring:
            st.info(
                "Pengukuran sedang berlangsung... Tekan 'Stop' untuk melihat hasil."
            )
        else:
            st.info("Tekan tombol 'Mulai' untuk memulai pengukuran fokus.")

# === KOLOM KANAN: Hasil Analisis ===
with col2:
    st.subheader("Analisis")

    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        try:
            result = webrtc_ctx.video_processor._output_queue.get(timeout=1.0)
            st.session_state.latest_label = result["label"]
            st.session_state.latest_conf = result["confidence"]

            st.session_state.focus_history.append(st.session_state.latest_conf)
            if len(st.session_state.focus_history) > 20:
                st.session_state.focus_history.pop(0)

            if result.get("face_image"):
                try:
                    face_bytes = base64.b64decode(result["face_image"])
                    st.session_state.latest_face = Image.open(BytesIO(face_bytes))
                except:
                    st.session_state.latest_face = None
            else:
                st.session_state.latest_face = None

            if st.session_state.measuring:
                st.session_state.measure_values.append(st.session_state.latest_conf)

        except queue.Empty:
            pass

    with st.container(border=True):
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Status Terakhir", st.session_state.latest_label)
        metric_col2.metric("Tingkat Keyakinan", f"{st.session_state.latest_conf:.2%}")

        st.write("---")
        st.write("**Wajah Terdeteksi**")
        if st.session_state.latest_face:
            col_spacer1, col_img, col_spacer2 = st.columns([1, 3.5, 1])
            with col_img:
                st.image(
                    st.session_state.latest_face,
                    caption="Wajah yang dianalisis",
                    use_container_width=True,
                )
        else:
            st.info("Belum ada wajah yang terdeteksi dari server.")

    with st.container(border=True):
        st.subheader("Fluktuasi Tingkat Fokus")
        if st.session_state.focus_history:
            chart_data = pd.DataFrame({"Tingkat Fokus": st.session_state.focus_history})
            st.area_chart(chart_data, height=200, use_container_width=True)
        else:
            st.info("Grafik akan muncul setelah data pertama diterima.")
