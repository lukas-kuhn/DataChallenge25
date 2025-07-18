import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import time

from inference import load_model
from train import denormalize


def preprocess_image(image_input, image_size=224): #TODO anpassen an neues modell und wieder auslagern
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        image = Image.open(image_input).convert('RGB')

    return transform(image).unsqueeze(0)

def slerp(val, low, high):
    # Implementation der slerp zwischen low und high bei val (0 <= val <= 1)
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    val = torch.tensor([val], device=low.device)
    if so.item() == 0:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high

def linear(val, low, high):
    # Lineare Interpolation zwischen low und high bei val (0 <= val <= 1)
    return (1 - val) * low + val * high

def interpolate_latents(alpha, mu1_flat, mu2_flat, interp_mode, model):
    if interp_mode == "Linear":
        z_interp_flat = (1 - alpha) * mu1_flat + alpha * mu2_flat
    else:  # SLERP
        z_interp_flat = slerp(alpha, mu1_flat, mu2_flat)

    z_interp = z_interp_flat.view(1, 256, 14, 14)  # z. B. [1, 256, 14, 14]
    
    with torch.no_grad():
        recon = model.decode(z_interp)
    
    img_tensor = denormalize(recon.squeeze(0).cpu()).clamp(0, 1)
    img_pil = to_pil_image(img_tensor)
    return img_pil

def rotate_shift_zoom(image, rotation_degree, x_offset, y_offset, zoom=1.0, fill=(255, 255, 255)):
    orig_size = image.size
    zoomed_size = (int(orig_size[0] * zoom), int(orig_size[1] * zoom))
    zoomed_img = image.resize(zoomed_size, resample=Image.BICUBIC)

    canvas_size = (image.width, image.height)
    # 1. Bild rotieren mit weißem Hintergrund
    rotated = zoomed_img.rotate(rotation_degree, resample=Image.BICUBIC, fillcolor=fill)
    
    # 2. Neues weißes Canvas erstellen
    canvas = Image.new("RGB", canvas_size, fill)
    
    # 3. Das rotierte Bild mit Offsets auf das Canvas verschieben
    canvas.paste(rotated, (x_offset, y_offset))
    
    return canvas

def add_grid(img, grid_size=56, line_color=(255, 0, 0), line_width=1):
    """
    Zeichnet ein Raster direkt auf ein RGB-Bild.
    Transparenz ist dann nicht möglich, aber für einfarbige Linien reicht RGB.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Vertikale Linien
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)

    # Horizontale Linien
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)

    return img

def add_border(image, border_size=2, border_color=(0, 0, 0)):
    width, height = image.size
    new_size = (width + 2 * border_size, height + 2 * border_size)
    bordered = Image.new("RGB", new_size, border_color)
    bordered.paste(image, (border_size, border_size))
    return bordered

def resize_and_pad(img, target_size=(224, 224), background_color=(255, 255, 255)):
    """
    Skaliert das Bild proportional und zentriert es auf einem weißen Hintergrund der Zielgröße.
    """
    img = img.convert("RGB")
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Neues weißes Bild erstellen
    new_img = Image.new("RGB", target_size, background_color)

    paste_x = (target_size[0] - img.width) // 2
    paste_y = (target_size[1] - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))

    return new_img

def add_crosshair(img, color=(255, 0, 0), line_width=1):
    """
    Fügt dem gegebenen Bild ein Fadenkreuz in der Mitte hinzu.
    """
    w, h = img.size
    draw = ImageDraw.Draw(img)
    center_x, center_y = w // 2, h // 2
    draw.line([(center_x, 0), (center_x, h)], fill=color, width=line_width)
    draw.line([(0, center_y), (w, center_y)], fill=color, width=line_width)
    return img


def main():
    st.title("Bildinterpolation mit Slider")

    # Session State für Reset-Funktionalität initialisieren
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    if 'last_img1_name' not in st.session_state:
        st.session_state.last_img1_name = None
    if 'last_img2_name' not in st.session_state:
        st.session_state.last_img2_name = None

    # Bildauswahl durch den Nutzer
    img1_file = st.file_uploader("Lade Bild 1 hoch", type=["png", "jpg", "jpeg"], key="img1")
    img2_file = st.file_uploader("Lade Bild 2 hoch", type=["png", "jpg", "jpeg"], key="img2")


    if img1_file and img2_file:

        st.divider()

        # Prüfe, ob sich die Bilder geändert haben
        current_img1_name = img1_file.name
        current_img2_name = img2_file.name
        
        if (st.session_state.last_img1_name != current_img1_name or 
            st.session_state.last_img2_name != current_img2_name):
            # Bilder haben sich geändert - Reset auslösen
            st.session_state.reset_counter += 1
            st.session_state.last_img1_name = current_img1_name
            st.session_state.last_img2_name = current_img2_name


        img1_pil = resize_and_pad(Image.open(img1_file))
        img2_pil = resize_and_pad(Image.open(img2_file))
        

        sub_header, button_col = st.columns([5, 1])
        with sub_header:
            st.subheader("Bild-Transformationen")
        with button_col:
            if st.button("🔄 Reset", help="Setzt alle Transformationsparameter zurück"):
                st.session_state.reset_counter += 1
                st.rerun()
        
        rotation = st.slider("Rotation für Bild 1 (°)", -180, 180, 0, 1, key=f"rotation_{st.session_state.reset_counter}")

        x_offset = st.slider("X-Verschiebung", -100, 100, 0, 1, key=f"x_offset_{st.session_state.reset_counter}")
        
        y_offset = st.slider("Y-Verschiebung", -100, 100, 0, 1, key=f"y_offset_{st.session_state.reset_counter}")

        zoom = st.slider("Zoom", min_value=0.5, max_value=2.0, value=1.0, step=0.01, key=f"zoom_{st.session_state.reset_counter}")

        # Reset Button und Gitteranzeige
        col_left, col_right = st.columns([2, 1])
        with col_left:
            show_grid = st.checkbox("Gitter anzeigen", value=True, key=f"show_grid_{st.session_state.reset_counter}")
        with col_right:
            grid_type = st.radio("", ["Fadenkreuz", "Vollgitter"], horizontal=True, label_visibility="collapsed", key=f"grid_type_{st.session_state.reset_counter}")


        result_img = rotate_shift_zoom(img1_pil, rotation, x_offset, y_offset, zoom)

        if show_grid:
            if grid_type == "Fadenkreuz":
                overlay1 = add_crosshair(result_img)
                overlay2 = add_crosshair(img2_pil)
            else: 
                overlay1 = add_grid(result_img)
                overlay2 = add_grid(img2_pil)
        else:
            overlay1 = result_img
            overlay2 = img2_pil

        col1, col2 = st.columns(2)
        with col1:
            st.image(add_border(overlay1), caption=f"Bild 1: {current_img1_name}", use_container_width=True)
        with col2:
            st.image(add_border(overlay2), caption=f"Bild 2: {current_img2_name}", use_container_width=True)

        st.divider()
        st.subheader("Latent Space Interpolation")

        # Auswahl der Interpolationsmethode
        interp_mode = st.radio("Interpolationsmethode", options=["Linear", "SLERP"])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _ = load_model(f"best_model.pth", device)
        model.eval()

        image1 = preprocess_image(result_img).to(device)
        image2 = preprocess_image(img2_pil).to(device)

        with torch.no_grad():
            mu1, _ = model.encode(image1)
            mu2, _ = model.encode(image2)

        mu1_flat = mu1.view(1, -1)
        mu2_flat = mu2.view(1, -1)

        alpha = st.slider("Interpolationsfaktor", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

        if interp_mode == "Linear":
            z_interp_flat = linear(alpha, mu1_flat, mu2_flat)
        else:  # SLERP
            z_interp_flat = slerp(alpha, mu1_flat, mu2_flat)

        z_interp = z_interp_flat.view_as(mu1) # Reshape to original latent shape
        with torch.no_grad():
            recon = model.decode(z_interp)

        img_tensor = denormalize(recon.squeeze(0).cpu()).clamp(0, 1)  # [C, H, W]
        img_pil = to_pil_image(img_tensor)

        st.image(img_pil, caption=f"Manuelle Interpolation (α = {alpha:.2f})", use_container_width=True)

        st.write(f"Tensor Min: {img_tensor.min():.4f}, Max: {img_tensor.max():.4f}, Mean: {img_tensor.mean():.4f}")

        # # --- Animation ---
        # st.markdown("### Automatische Animation")
        # play = st.button("▶️ Animation starten")

        # if play:
        #     placeholder = st.empty()
        #     steps = 50  # Anzahl Frames
        #     delay = 0.05  # Sekundendelay pro Frame

        #     # Hin und zurück
        #     for direction in [1, -1]:
        #         for i in range(steps + 1):
        #             a = i / steps if direction == 1 else (steps - i) / steps
        #             img = interpolate_latents(a, mu1_flat, mu2_flat, interp_mode, model)
        #             placeholder.image(img, caption=f"Animation (α = {a:.2f})", use_container_width=True)
        #             time.sleep(delay)

    else:
        st.info("Bitte lade zwei Bilder hoch, um die Interpolation zu sehen.")


if __name__ == "__main__":
    main()