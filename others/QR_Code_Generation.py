import qrcode

# ──────────────────────────────────────────────────────────────────────────────
# Helper: create and save a QR-code for given URL and filename.
def generate_qr(url: str, filename: str) -> None:
    """
    url      – the link to encode
    filename – output file (PNG)
    """
    # Instantiate builder with custom sizing:
    qr = qrcode.QRCode(
        version=None,          # automatic size
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,           # pixel-dim of each “box”
        border=4,              # modules thick
    )

    qr.add_data(url)          # queue the payload
    qr.make(fit=True)         # collate & determine optimal version

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)        # write out PNG

if __name__ == "__main__":
    links = {
        "qr_github": "https://github.com/SensorsINI/physical-cartpole",
        "qr_publication": "https://drive.google.com/file/d/1E4Wk2n-Il384iHKY7QRkmt9OxP0j_vY8/view?usp=share_link",
        "qr_video": "https://drive.google.com/file/d/1YPlq_zzI9xr6D40q_EI-CGqs1K7LMWWk/view?usp=share_link",

    }
    for name, link in links.items():
        generate_qr(link, f"{name}.png")
        print(f"✔ Generated {name}.png")
