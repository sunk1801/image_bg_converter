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
BACKGROUND_COLOR = "#7CA1BB"       # Change background color here
MAX_FILE_SIZE_KB = 100             # Output must be strictly below this
INITIAL_JPEG_QUALITY = 95
MIN_JPEG_QUALITY = 25
QUALITY_STEP = 5
RESIZE_STEP = 0.90
MIN_DIMENSION = 100

# AI background-removal model.
# "bria-rmbg" is used by current rembg releases as the default
# model and is suitable for general foreground/background separation.
AI_MODEL = "bria-rmbg"

# Optional: use a lighter model if CPU processing is too slow.
# Examples supported by rembg include:
# "u2net", "u2net_human_seg", "isnet-general-use"
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
    ".gif", ".heic", ".heif"
}

@st.cache_resource(show_spinner=False)
def load_ai_session():
    """Load the AI model once and reuse it for all images."""
    return new_session(AI_MODEL)


def hex_to_rgb(hex_color):
    value = hex_color.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("Background color must be a 6-digit HEX value.")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def ai_replace_background(img):
    """
    Use AI segmentation to create a foreground alpha mask.
    The original foreground is then composited onto the configured
    solid background. This is much more robust than white-pixel
    replacement for non-white/uneven backgrounds.
    """
    session = load_ai_session()

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

    bg = Image.new("RGBA", rgb.size, hex_to_rgb(BACKGROUND_COLOR) + (255,))
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


def compress_under_limit(img):
    """
    Quality reduction first. If quality reaches the configured minimum,
    reduce dimensions and retry. The result is strictly below the
    configured byte limit.
    """
    limit = MAX_FILE_SIZE_KB * 1024
    working = img.convert("RGB")

    while True:
        quality = INITIAL_JPEG_QUALITY

        while quality >= MIN_JPEG_QUALITY:
            data = encode_jpeg(working, quality)
            if len(data) < limit:
                return data, working.size, quality

            quality -= QUALITY_STEP

        if min(working.size) <= MIN_DIMENSION:
            # Last-resort dimension reduction.
            scale = 0.80
        else:
            scale = RESIZE_STEP

        new_width = max(1, int(working.width * scale))
        new_height = max(1, int(working.height * scale))

        if (new_width, new_height) == working.size:
            raise ValueError(
                f"Could not compress image below {MAX_FILE_SIZE_KB} KB."
            )

        working = working.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )


def open_image(data):
    with Image.open(io.BytesIO(data)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def process_image(data):
    img = open_image(data)
    segmented = ai_replace_background(img)
    return compress_under_limit(segmented)


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
`Background {BACKGROUND_COLOR} → JPG → < {MAX_FILE_SIZE_KB} KB`
"""
)

with st.sidebar:
    st.header("⚙️ Configuration")

    st.code(
        f"""BACKGROUND_COLOR = "{BACKGROUND_COLOR}"
MAX_FILE_SIZE_KB = {MAX_FILE_SIZE_KB}
AI_MODEL = "{AI_MODEL}"
INITIAL_JPEG_QUALITY = {INITIAL_JPEG_QUALITY}
MIN_JPEG_QUALITY = {MIN_JPEG_QUALITY}
QUALITY_STEP = {QUALITY_STEP}
RESIZE_STEP = {RESIZE_STEP}""",
        language="python"
    )

    st.info(
        "The AI model runs locally after installation. "
        "The first run downloads the model weights; subsequent runs "
        "reuse the cached model."
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
                    f"({index}/{len(image_items)})..."
                )

                try:
                    output_data, dimensions, quality = process_image(data)

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
