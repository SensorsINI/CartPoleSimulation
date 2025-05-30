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
        "physical_cartpole": "https://github.com/SensorsINI/physical-cartpole",
        "google_drive"     : "https://drive.google.com/drive/folders/1PbKjwNEYb3FqAs-qltQiflUjSwp6yNx1?usp=drive_link",
    }
    for name, link in links.items():
        generate_qr(link, f"{name}.png")
        print(f"✔ Generated {name}.png")
