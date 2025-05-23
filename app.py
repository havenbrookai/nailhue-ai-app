import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
from streamlit_drawable_canvas import st_canvas
import pandas as pd

@st.cache_data
def load_polish_data():
    df = pd.read_csv("Nail_Polish_Shades_with_Hex_Codes.csv")
    df.columns = [col.strip() for col in df.columns]
    return df

polish_df = load_polish_data()

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def hex_to_rgb(hex_val):
    hex_val = hex_val.strip().lstrip('#')
    return tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))

def extract_colors_from_mask(image, mask, num_colors=5):
    mask_np = np.array(mask.convert("L"))
    image_np = np.array(image)
    selected_pixels = image_np[mask_np > 10]
    if len(selected_pixels) < 10:
        return []
    kmeans = KMeans(n_clusters=min(num_colors, len(np.unique(selected_pixels, axis=0))), n_init='auto')
    kmeans.fit(selected_pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in colors]

def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def match_to_brands(user_rgb):
    best_matches = {}
    for tier in ["Premium", "Common", "Affordable"]:
        tier_df = polish_df[polish_df["Tier"].str.strip().str.lower() == tier.lower()]
        closest = None
        smallest_diff = float("inf")
        for _, row in tier_df.iterrows():
            try:
                db_rgb = hex_to_rgb(row["Hex Code"])
                dist = color_distance(user_rgb, db_rgb)
                if dist < smallest_diff:
                    smallest_diff = dist
                    closest = row
            except:
                continue
        best_matches[tier] = closest
    return best_matches

st.title("💅 NailHue AI – Draw to Detect Nail Colors")
st.write("Upload a photo, draw over nails to detect polish colors and match to brands by tier.")

uploaded_file = st.file_uploader("Upload a nail photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    st.markdown("### 🖌️ Step 1: Draw Over Nails to Select Color Zones")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 120)", 
        stroke_width=10,
        stroke_color="red",
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("🎯 Step 2: Match Colors from Drawn Area"):
        if canvas_result.image_data is not None:
            mask = Image.fromarray((canvas_result.image_data[:, :, 3] > 0).astype(np.uint8) * 255)
            drawn_colors = extract_colors_from_mask(image, mask, num_colors=5)

            if not drawn_colors:
                st.warning("Please draw over a visible part of the nail before matching.")
            else:
                for idx, color in enumerate(drawn_colors):
                    hex_code = rgb_to_hex(color)
                    st.subheader(f"🎨 Color {idx+1}: `{hex_code}`")
                    st.markdown(f"<div style='background-color:{hex_code}; width:100px; height:30px; border-radius:5px;'></div>", unsafe_allow_html=True)

                    matches = match_to_brands(color)
                    for tier in ["Premium", "Common", "Affordable"]:
                        match = matches[tier]
                        if match is not None:
                            st.markdown(f"**{tier}**: {match['Brand']} – *{match['Shade Name']}*")
                            st.markdown(f"<div style='background-color:{match['Hex Code']}; width:100px; height:30px; border-radius:5px;'></div>", unsafe_allow_html=True)