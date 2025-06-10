
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")
    return pipe

pipe = load_model()

st.title("🖼️ Text-to-Image Generator (Stable Diffusion)")

prompt = st.text_input("Enter your prompt:", "A futuristic city with flying cars")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
