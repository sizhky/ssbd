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
from transformers import Trainer, TrainingArguments
from torch_snippets import *

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
model = TransformerEncoder(4, 512, 128)
load_torch_model_weights_to(model, artifacts_dir / "a/pytorch_model.bin", device="cpu")
model.eval()

clip = Frame2Embeddings(device="cpu")

training_args = TrainingArguments(
    output_dir=predictions_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    label_names=["targets"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
)


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
        frames = video_to_frames(
            video_path, frames_path=None, start_sec=0, clip_duration=5, verbose=True
        )

        # Make predictions
        with torch.no_grad():
            input = clip.frames2clip_image_embeddings(frames)
            input = {"embeddings": input}
            Info(f"{input=}")
            input = collate_fn([input])
            predictions = model(**input)
            predictions = F.softmax(predictions["logits"], dim=-1)
            Info(f"{predictions=}")
            confidence, predicted_class = predictions.max(1)
            predicted_class = ClipEmbeddingsDataset.id2label[predicted_class.item()]
            confidence = confidence.item()
            Info(f"{predicted_class=} @ {confidence=}")

        # Return the predicted class
        return JSONResponse(
            content={
                "predicted_class": predicted_class,
                "confidence": f"{confidence:.2f}",
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
