# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/data/02_video_to_frames.ipynb.

# %% auto 0
__all__ = [
    "get_transform",
    "get_video",
    "get_video_duration",
    "video_to_frames",
    "extract_frames_for_annotations",
]

# %% ../../nbs/data/02_video_to_frames.ipynb 3
from ..cli import cli
from torch_snippets import *
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
)
from functools import lru_cache


# %% ../../nbs/data/02_video_to_frames.ipynb 4
def get_transform(num_frames, side_size, crop_size):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size)),
            ]
        ),
    )


@lru_cache
def get_video(video_path):
    video = EncodedVideo.from_path(video_path)
    return video


def get_video_duration(video_path):
    video = get_video(video_path)
    return video.duration


def video_to_frames(
    video_path,
    frames_path,
    start_sec: int,
    clip_duration: float,
    num_frames_per_sec: int = 10,
    side_size: int = 256,
    crop_size: int = 256,
    verbose=False,
):
    video = get_video(video_path)
    if clip_duration is None:
        clip_duration = video.duration
    end_sec = start_sec + clip_duration
    with notify_waiting("Loading Video Clip"):
        if verbose:
            Info(f"Video duration: {video.duration} s")
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        num_frames = clip_duration * num_frames_per_sec
        video_data = get_transform(num_frames, side_size, crop_size)(video_data)
        tensor_data = video_data["video"].permute(1, 0, 2, 3)
    Info(f"{tensor_data=}")
    if frames_path is None:
        return tensor_data
    if exists(frames_path):
        Info(f"Skipping extraction for {frames_path}")
        return
    makedir(parent(frames_path))
    dumpdill(tensor_data, frames_path)


@cli.command()
def extract_frames_for_annotations(
    annotations_path,
    videos_folder,
    frames_folder,
    num_frames_per_sec: int = 5,
    side_size: int = 256,
    crop_size: int = 256,
    exclude_others: bool = True,
    row_index: int = None,
):
    videos_folder = P(videos_folder)
    frames_folder = P(frames_folder)
    annotations = pd.read_csv(annotations_path).rename({"class": "activity"}, axis=1)
    if exclude_others:
        annotations = annotations.query('activity != "others"')
    for _, row in (
        tracker := track2(
            annotations.sort_values("video").iterrows(), total=len(annotations)
        )
    ):
        if row_index is not None and row.name != row_index:
            continue
        video = videos_folder / f"{row.video}.mp4"
        frames = frames_folder / f"{row.name}.frames.tensor"
        tracker.send(f"Processing {frames}")
        video_to_frames(
            video,
            frames,
            row.start,
            row.clip_duration,
            num_frames_per_sec=num_frames_per_sec,
            side_size=side_size,
            crop_size=crop_size,
        )
