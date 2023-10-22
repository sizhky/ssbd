# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/model/00_frames_to_embeddings.ipynb.

# %% auto 0
__all__ = ["Frame2Embeddings", "frames_to_embeddings"]

# %% ../../nbs/model/00_frames_to_embeddings.ipynb 3
from ..cli import cli
from torch_snippets import *
import clip


# %% ../../nbs/model/00_frames_to_embeddings.ipynb 4
class Frame2Embeddings:
    def __init__(self, model="ViT-B/32", device="cuda", batch_size=16):
        self.model, self.preprocess = clip.load(model, device=device)
        self.batch_size = batch_size

    def __call__(self, frames_tensor_path_or_folder, **kwargs):
        if os.path.isdir(frames_tensor_path_or_folder):
            return self.extract_clip_embeddings_for_folder(
                frames_tensor_path_or_folder, **kwargs
            )
        else:
            return self.frames2clip_image_embeddings(
                frames_tensor_path_or_folder, **kwargs
            )

    @torch.no_grad()
    def frames2clip_image_embeddings(self, frames_tensor_path):
        frames = loaddill(frames_tensor_path)
        frames = [
            (np.array(a) * 255).astype(np.uint8).transpose(1, 2, 0) for a in frames
        ]
        frames = [Image.fromarray(im) for im in frames]
        frames = torch.stack([self.preprocess(im) for im in frames]).to(device)
        batches = torch.split(frames, self.batch_size)
        embeddings = []
        for batch in batches:
            embeddings.append(self.model.encode_image(batch).cpu().detach())
        embeddings = torch.cat(embeddings)

        return embeddings

    def extract_clip_embeddings_for_folder(
        self, frames_folder, embeddings_folder, n=None
    ):
        frames_folder = P(frames_folder)
        embeddings_folder = P(embeddings_folder)
        makedir(embeddings_folder)
        for ix, frames_tensor_path in E((tracker := track2(frames_folder.ls()))):
            tracker.send(f"Processing {frames_tensor_path}")
            if n is not None and ix >= n:
                return
            to = f"{embeddings_folder}/{stem(frames_tensor_path)}.embeddings.tensor"
            if exists(to):
                Info(f"Skipping {to} as it already exists")
                continue
            embeddings = self(frames_tensor_path)
            dumpdill(embeddings, to)


@cli.command()
def frames_to_embeddings(frames_folder, embeddings_folder, model, device):
    f2e = Frame2Embeddings(model, device)
    f2e(frames_folder, embeddings_folder=embeddings_folder)
