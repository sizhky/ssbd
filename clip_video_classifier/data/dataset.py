# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/data/03_embeddings_dataset.ipynb.

# %% auto 0
__all__ = ["ClipEmbeddingsDataset"]

# %% ../../nbs/data/03_embeddings_dataset.ipynb 3
from torch_snippets import *
from functools import lru_cache


# %% ../../nbs/data/03_embeddings_dataset.ipynb 4
class ClipEmbeddingsDataset(Dataset):
    labels = ["ArmFlapping", "HeadBanging", "Spinning", "others"]
    label2id = {l: ix for ix, l in enumerate(labels)}
    id2label = {ix: l for l, ix in label2id.items()}

    def __init__(
        self,
        embeddings_dir: str,
        annotations: str,
        average_embeddings: bool = False,
        frames_dir: str = None,
        binary_mode: bool = False,
    ):
        self.average_embeddings = average_embeddings
        self.embeddings_dir = P(embeddings_dir)
        if isinstance(annotations, (str, P)):
            self.annotations = pd.read_csv(annotations)
        else:
            self.annotations = annotations
        available_embeddings = [
            int(stem(f).split(".")[0]) for f in self.embeddings_dir.ls()
        ]
        available_annotations = self.annotations.index.tolist()

        self.annotations = self.annotations.loc[
            list(common(available_annotations, available_embeddings))
        ]
        self.frames_dir = frames_dir
        self.binary_mode = binary_mode
        Info(f"created dataset of {len(self)} items")

    def __len__(self):
        return len(self.annotations)

    @lru_cache
    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        embedding = loaddill(
            self.embeddings_dir / f"{row.name}.frames.embeddings.tensor"
        ).cpu()
        label = row["label"]
        if self.binary_mode:
            label = int(label != "others")
        else:
            label = self.label2id[label]
        if 0:
            frames = loaddill(self.frames_dir / f"{row.name}.frames.tensor")
        return {
            "embeddings": embedding.cpu().detach(),
            "input": embedding.mean(0).cpu().detach(),
            "targets": label,
            "label": label,
        }
