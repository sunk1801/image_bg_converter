import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from rembg import new_session, remove

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_BACKGROUND_COLOR = "#7CA1BB"       # Change background color here
DEFAULT_MAX_FILE_SIZE_KB = 100             # Output must be strictly below this
DEFAULT_INITIAL_JPEG_QUALITY = 95
DEFAULT_MIN_JPEG_QUALITY = 25
DEFAULT_QUALITY_STEP = 5
DEFAULT_RESIZE_STEP = 0.90
DEFAULT_MIN_DIMENSION = 100

# AI background-removal model (LIGHTER for Streamlit).
# "u2net" is a lightweight model suitable for general foreground/background separation
# and runs efficiently on CPU/limited hardware.
# Other lighter alternatives:
# - "u2net_human_seg" (optimized for human segmentation)
# - "isnet-general-use" (general purpose, moderate size)
# - "bria-rmbg" (heavier, more accurate - not recommended for limited resources)
DEFAULT_AI_MODEL = "u2net"

# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
    ".gif", ".heic", ".heif"
}

AVAILABLE_AI_MODELS = {
    "u2net": "⚡ Lightweight (fastest, CPU-friendly)",
    "u2net_human_seg": "👤 Human Optimized (good for portraits)",
    "isnet-general-use": "🎯 General Purpose (balanced)",
    "bria-rmbg": "🔬 Heavy (most accurate, slower)",
}


@st.cache_resource(show_spinner=False)
def load_ai_session(model_name):
    """Load the AI model once and reuse it for all images."""
    return new_session(model_name)


def hex_to_rgb(hex_color):
    value = hex_color.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("Background color must be a 6-digit HEX value.")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def ai_replace_background(img, model_name):
    """
    Use AI segmentation to create a foreground alpha mask.
    The original foreground is then composited onto the configured
    solid background. This is much more robust than white-pixel
    replacement for non-white/uneven backgrounds.
    """
    session = load_ai_session(model_name)

    rgb = img.convert("RGB")
    source_buffer = io.BytesIO()
    rgb.save(source_buffer, format="PNG")

    result = remove(
        source_buffer.getvalue(),
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )

    foreground = Image.open(io.BytesIO(result)).convert("RGBA")

    # AI mask is the alpha channel.
    alpha = foreground.getchannel("A")

    bg = Image.new("RGBA", rgb.size, hex_to_rgb(st.session_state.get("background_color", DEFAULT_BACKGROUND_COLOR)) + (255,))
    composed = Image.composite(foreground, bg, alpha)

    return composed.convert("RGB")


def encode_jpeg(img, quality):
    buffer = io.BytesIO()
    img.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True
    )
    return buffer.getvalue()


def compress_under_limit(img, max_size_kb, initial_quality, min_quality, quality_step, resize_step, min_dimension):
    """
    Quality reduction first. If quality reaches the configured minimum,
    reduce dimensions and retry. The result is strictly below the
    configured byte limit.
    """
    limit = max_size_kb * 1024
    working = img.convert("RGB")

    while True:
        quality = initial_quality

        while quality >= min_quality:
            data = encode_jpeg(working, quality)
            if len(data) < limit:
                return data, working.size, quality

            quality -= quality_step

        if min(working.size) <= min_dimension:
            # Last-resort dimension reduction.
            scale = 0.80
        else:
            scale = resize_step

        new_width = max(1, int(working.width * scale))
        new_height = max(1, int(working.height * scale))

        if (new_width, new_height) == working.size:
            raise ValueError(
                f"Could not compress image below {max_size_kb} KB."
            )

        working = working.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )


def open_image(data):
    with Image.open(io.BytesIO(data)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def process_image(data, model_name, max_size_kb, initial_quality, min_quality, quality_step, resize_step, min_dimension):
    img = open_image(data)
    segmented = ai_replace_background(img, model_name)
    return compress_under_limit(segmented, max_size_kb, initial_quality, min_quality, quality_step, resize_step, min_dimension)


def extract_zip(uploaded_zip):
    items = []

    with zipfile.ZipFile(uploaded_zip) as z:
        for info in z.infolist():
            if info.is_dir():
                continue

            suffix = Path(info.filename).suffix.lower()

            if suffix in SUPPORTED_EXTENSIONS:
                items.append((info.filename, z.read(info.filename)))

    return items


def create_output_zip(results):
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        used_names = set()

        for result in results:
            name = result["output_path"]
            candidate = name
            counter = 1

            while candidate.lower() in used_names:
                p = Path(name)
                candidate = str(
                    p.with_name(f"{p.stem}_{counter}{p.suffix}")
                )
                counter += 1

            used_names.add(candidate.lower())
            z.writestr(candidate, result["data"])

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="AI Image → JPG <100KB",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Image Background + JPG <100 KB Converter")

st.markdown(
    f"""
**Pipeline:**

`Any Image → RGB → AI Foreground Segmentation → `
`Solid Background → JPG → < 100 KB`
"""
)

# Initialize session state with defaults
if "background_color" not in st.session_state:
    st.session_state.background_color = DEFAULT_BACKGROUND_COLOR
if "max_file_size_kb" not in st.session_state:
    st.session_state.max_file_size_kb = DEFAULT_MAX_FILE_SIZE_KB
if "ai_model" not in st.session_state:
    st.session_state.ai_model = DEFAULT_AI_MODEL
if "initial_jpeg_quality" not in st.session_state:
    st.session_state.initial_jpeg_quality = DEFAULT_INITIAL_JPEG_QUALITY
if "min_jpeg_quality" not in st.session_state:
    st.session_state.min_jpeg_quality = DEFAULT_MIN_JPEG_QUALITY
if "quality_step" not in st.session_state:
    st.session_state.quality_step = DEFAULT_QUALITY_STEP
if "resize_step" not in st.session_state:
    st.session_state.resize_step = DEFAULT_RESIZE_STEP

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🎨 Background Color")
    background_color = st.color_picker(
        "Select background color",
        value=st.session_state.background_color,
        key="bg_color_picker"
    )
    st.session_state.background_color = background_color

    st.subheader("🤖 AI Model")
    ai_model = st.selectbox(
        "Choose AI model (lighter = faster)",
        options=list(AVAILABLE_AI_MODELS.keys()),
        format_func=lambda x: AVAILABLE_AI_MODELS[x],
        index=list(AVAILABLE_AI_MODELS.keys()).index(st.session_state.ai_model),
        key="model_selector"
    )
    st.session_state.ai_model = ai_model

    if ai_model == "bria-rmbg":
        st.warning("⚠️ Heavy model - may be slow on limited hardware")
    elif ai_model == "u2net":
        st.success("✅ Lightweight & fast")

    st.subheader("📦 Compression Settings")
    max_file_size_kb = st.slider(
        "Max file size (KB)",
        min_value=50,
        max_value=500,
        value=st.session_state.max_file_size_kb,
        step=10,
        key="max_size_slider"
    )
    st.session_state.max_file_size_kb = max_file_size_kb

    initial_quality = st.slider(
        "Initial JPEG quality",
        min_value=50,
        max_value=100,
        value=st.session_state.initial_jpeg_quality,
        step=5,
        key="init_quality_slider"
    )
    st.session_state.initial_jpeg_quality = initial_quality

    min_quality = st.slider(
        "Minimum JPEG quality",
        min_value=10,
        max_value=50,
        value=st.session_state.min_jpeg_quality,
        step=5,
        key="min_quality_slider"
    )
    st.session_state.min_jpeg_quality = min_quality

    quality_step = st.slider(
        "Quality reduction step",
        min_value=1,
        max_value=10,
        value=st.session_state.quality_step,
        key="quality_step_slider"
    )
    st.session_state.quality_step = quality_step

    resize_step = st.slider(
        "Resize reduction factor",
        min_value=0.70,
        max_value=0.99,
        value=st.session_state.resize_step,
        step=0.02,
        key="resize_step_slider"
    )
    st.session_state.resize_step = resize_step

    st.divider()
    st.info(
        "The AI model runs locally. "
        "The first run downloads the model weights; subsequent runs "
        "reuse the cached model."
    )

    # Display current config
    st.markdown("**Current Configuration:**")
    st.code(
        f"""BACKGROUND_COLOR = "{st.session_state.background_color}"
MAX_FILE_SIZE_KB = {st.session_state.max_file_size_kb}
AI_MODEL = "{st.session_state.ai_model}"
INITIAL_JPEG_QUALITY = {st.session_state.initial_jpeg_quality}
MIN_JPEG_QUALITY = {st.session_state.min_jpeg_quality}
QUALITY_STEP = {st.session_state.quality_step}
RESIZE_STEP = {st.session_state.resize_step}""",
        language="python"
    )

uploaded_files = st.file_uploader(
    "Upload one or more images OR a ZIP containing images",
    type=[
        "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff",
        "gif", "heic", "heif", "zip"
    ],
    accept_multiple_files=True
)

if uploaded_files:

    image_items = []
    skipped = []

    for uploaded in uploaded_files:

        if uploaded.name.lower().endswith(".zip"):
            try:
                image_items.extend(extract_zip(uploaded))
            except zipfile.BadZipFile:
                skipped.append(f"{uploaded.name}: invalid ZIP")

        else:
            suffix = Path(uploaded.name).suffix.lower()

            if suffix in SUPPORTED_EXTENSIONS:
                image_items.append(
                    (uploaded.name, uploaded.getvalue())
                )
            else:
                skipped.append(
                    f"{uploaded.name}: unsupported extension"
                )

    st.write(f"**{len(image_items)} image(s) ready for AI processing.**")

    if skipped:
        st.warning("\n".join(skipped))

    if image_items:

        if st.button("🚀 AI Convert & Compress", type="primary"):

            results = []
            progress = st.progress(0)
            status = st.empty()

            for index, (input_path, data) in enumerate(
                image_items, start=1
            ):

                status.write(
                    f"AI processing `{input_path}` "
                    f"({index}/{len(image_items)}) using {st.session_state.ai_model}..."
                )

                try:
                    output_data, dimensions, quality = process_image(
                        data,
                        st.session_state.ai_model,
                        st.session_state.max_file_size_kb,
                        st.session_state.initial_jpeg_quality,
                        st.session_state.min_jpeg_quality,
                        st.session_state.quality_step,
                        st.session_state.resize_step,
                        DEFAULT_MIN_DIMENSION
                    )

                    source_path = Path(input_path)
                    parent = str(source_path.parent)

                    output_path = (
                        f"{parent}/{source_path.stem}.jpg"
                        if parent not in ("", ".")
                        else f"{source_path.stem}.jpg"
                    )

                    results.append(
                        {
                            "input": input_path,
                            "output_path": output_path,
                            "data": output_data,
                            "size_kb": len(output_data) / 1024,
                            "dimensions": dimensions,
                            "quality": quality,
                        }
                    )

                except Exception as exc:
                    results.append(
                        {
                            "input": input_path,
                            "error": str(exc)
                        }
                    )

                progress.progress(
                    index / len(image_items)
                )

            status.empty()
            st.session_state["results"] = results


if "results" in st.session_state:

    results = st.session_state["results"]

    successful = [
        r for r in results
        if "error" not in r
    ]

    failed = [
        r for r in results
        if "error" in r
    ]

    st.divider()
    st.subheader("✅ Converted Images")

    if successful:

        cols = st.columns(4)

        for i, result in enumerate(successful):

            with cols[i % 4]:

                st.image(
                    result["data"],
                    caption=result["output_path"],
                    width="stretch"
                )

                st.caption(
                    f"{result['size_kb']:.1f} KB · "
                    f"{result['dimensions'][0]}×"
                    f"{result['dimensions'][1]} · "
                    f"JPEG Q{result['quality']}"
                )

                st.download_button(
                    "⬇️ Download",
                    data=result["data"],
                    file_name=Path(
                        result["output_path"]
                    ).name,
                    mime="image/jpeg",
                    key=f"download_{i}"
                )

        zip_data = create_output_zip(successful)

        st.download_button(
            "📦 Download All as ZIP",
            data=zip_data,
            file_name="ai_converted_images.zip",
            mime="application/zip",
            type="primary"
        )

    if failed:

        st.error("Some images could not be processed:")

        for result in failed:
            st.write(
                f"- `{result['input']}`: "
                f"{result['error']}"
            )
