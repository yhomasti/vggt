from flask import Flask, request, send_file
from pathlib import Path
app = Flask(__name__)

OUT = Path("output"); OUT.mkdir(exist_ok=True)
# Put a small test GLB at output/result.glb (see step 3)

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if f: (OUT / "last.jpg").write_bytes(f.read())
    return {"ok": True}

@app.get("/output/result.glb")
def get_glb():
    return send_file(OUT / "result.glb", mimetype="model/gltf-binary")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
