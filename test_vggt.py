import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
print("Device:", device, "dtype:", dtype)

# Replace with your real image paths (1+ images are fine)
image_paths = [
    r"C:\path\to\img1.jpg",
    r"C:\path\to\img2.jpg"
]

# Load model (this will download ~5GB weights the first time)
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
images = load_and_preprocess_images(image_paths).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=(device=="cuda"), dtype=dtype):
        preds = model(images)

print("Prediction keys:", list(preds.keys()))
