import streamlit as st
from pathlib import Path
from pipeline_vr import process_video_to_vr180
import base64

# ==============================
# Page config
# ==============================
st.set_page_config(page_title="VR180 Video Converter", layout="wide")

# ==============================
# Background CSS
# ==============================
def set_background_image(image_file):
    """Set .webp background for Streamlit app"""
    with open(image_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

dummy_bg = "background.jpeg"
if Path(dummy_bg).exists():
    set_background_image(dummy_bg)
else:
    st.warning("Background image not found. Using black dummy background.")

# ==============================
# Title
# ==============================
st.title("🎬 VR180 Video Converter")

# ==============================
# Upload input video
# ==============================
st.subheader("Upload Your Input Video")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

# ==============================
# Resolution select
# ==============================
st.subheader("Resolution Quality")
resolution = st.radio("Choose resolution:", options=[2880, 3084], index=0)

# ==============================
# Camera offset select
# ==============================
# st.subheader("Camera Position")
# camera_label_map = {
#     "Theater front seat mode(note:view close to you)": 0.3,
#     "Middle seat mode(note:view little far away to you)": 0.6,
#     "Back middle mode(note:view far away to you)": 0.9,
#     "Back seat mode(note:view very far away to you)": 1.2
# }
# camera_position = st.selectbox("Select camera position:", options=list(camera_label_map.keys()))
# camera_offset = camera_label_map[camera_position]

# ==============================
# Process button
# ==============================
if st.button("🚀 Convert to VR180"):
    if uploaded_file is None:
        st.warning("Please upload a video first!")
    else:
        # Save uploaded video temporarily
        temp_input = Path("temp_input_video.mp4")
        with open(temp_input, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Output path
        output_video = Path(f"output_vr180_{resolution}.mp4").resolve()

        # Live progress UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        indeterminate = st.empty()

        def show_indeterminate():
            indeterminate.markdown(
                
                """
                <style>
                .indeterminate-bar {
                    position: relative;
                    height: 10px;
                    background: #e6eef7;
                    border-radius: 6px;
                    overflow: hidden;
                    box-shadow: inset 0 0 3px rgba(0,0,0,0.1);
                }
                .indeterminate-bar:before {
                    content: "";
                    position: absolute;
                    left: -40%;
                    top: 0;
                    height: 100%;
                    width: 40%;
                    background: linear-gradient(90deg, rgba(66,135,245,0) 0%, rgba(66,135,245,0.6) 50%, rgba(66,135,245,0) 100%);
                    animation: slide 1.2s infinite;
                }
                @keyframes slide {
                    0% { left: -40%; }
                    50% { left: 60%; }
                    100% { left: 140%; }
                }
                </style>
                <div class="indeterminate-bar"></div>
                """,
                unsafe_allow_html=True,
            )

        def hide_indeterminate():
            indeterminate.empty()

        try:
            # Define callbacks for live updates
            def update_progress(pct: int):
                try:
                    progress_bar.progress(int(max(0, min(100, pct))))
                except Exception:
                    pass

            def update_status(msg: str):
                try:
                    status_text.text(str(msg))
                except Exception:
                    pass

            # Show moving stripes to indicate active processing
            show_indeterminate()

            # Run pipeline with callbacks
            meta_output = process_video_to_vr180(
                str(temp_input),
                str(output_video),
                fps=15,
                midas_model="MIDAS_SMALL",
                baseline=15,
                vr_output_size=resolution,
                compression_strength=0.3,
                camera_offset=0.3,
                panini_weight=0,
                stereo_weight=0,
                blur_offset=100,
                blur_mode="edge",
                blur_strength=50,
                progress_callback=update_progress,
                status_callback=update_status,
            )

            progress_bar.progress(100)
            status_text.text("Done.")
            hide_indeterminate()
            st.success("✅ VR180 video generated successfully!")
            st.video(str(meta_output))

        except Exception as e:
            hide_indeterminate()
            st.error(f"❌ Error occurred: {e}")
