# Import necessary libraries
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from clip_video_classifier.preprocess.video_to_frames import video_to_frames
from clip_video_classifier.models.transformer_encoder import (
    TransformerEncoder,
    collate_fn,
)
from clip_video_classifier.models.frame_embeddings import Frame2Embeddings
from clip_video_classifier.data.dataset import ClipEmbeddingsDataset
from clip_video_classifier.models.inference import Inference
from torch_snippets import *
from functools import lru_cache

# Create a FastAPI instance
app = FastAPI()


request_dir = P(os.environ.get("REQUEST_DIR", "/tmp/video-uploads/"))
predictions_dir = P(
    os.environ.get("PREDICTIONS_DIR", "/tmp/transformer-encoder-predictions/")
)
artifacts_dir = P(
    os.environ.get(
        "ARTIFACTS_DIR",
        "/mnt/347832F37832B388/projects/ssbd/cogniable-assignment/artifacts/",
    )
)


# Load your trained video classification model
inference = Inference(artifacts_dir / "a/pytorch_model.bin")


# Define an endpoint for video classification
@app.post("/classify_video/")
async def classify_video(file: UploadFile = File(...)):
    try:
        # Read the uploaded video file
        request_id = rand()
        # Define the file path to save the video
        video_path = request_dir / f"{request_id}/video.mp4"
        makedir(parent(video_path))
        video_path = str(video_path)  # Convert to a string
        contents = file.file.read()
        with open(video_path, "wb") as video_file:
            video_file.write(contents)
        Info(f"Video written at {video_path}")
        # Process the video frames and classify
        content = inference.predict_on_video_path(video_path)
        return JSONResponse(content=content)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
