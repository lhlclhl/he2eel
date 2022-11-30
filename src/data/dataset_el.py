import os

import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset

# entity set for negative entity sampling
entities = []
efreqs = []
erandbuf = []
entname_path = "data/entities.txt"
if os.path.exists(entname_path):
    with open(entname_path, encoding="utf-8") as fin:
        for line in fin:
            ent, freq = line.strip("\r\n").rsplit("\t", 1)
            entities.append(ent)
            efreqs.append(int(freq))
    entities.append("NIL")
    fs = sum(efreqs)
    efreqs.append(fs/len(efreqs))
    efreqs = np.array(efreqs)
    efreqs **= 0.75
    efreqs /= efreqs.sum()
print(len(entities), "entities loaded")

def get_negative_entity():
    global erandbuf
    if not erandbuf:
        erandbuf = np.random.choice(len(entities), 1000, p=efreqs).tolist()
    return entities[erandbuf.pop()]

class DatasetEL(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=32,
        max_length_span=15,
        test=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        with jsonlines.open(data_path) as f:
            self.data = list(f)

        self.max_length = max_length
        self.max_length_span = max_length_span
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):

        batch = {
            **{
                f"src_{k}": v
                for k, v in self.tokenizer(
                    [b["input"] for b in batch],
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_offsets_mapping=True,
                ).items()
            },
            "offsets_start": (
                [
                    i
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                ],
                [
                    a[0]
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                ],
            ),
            "offsets_end": (
                [
                    i
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                ],
                [
                    a[1]
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                ],
            ),
            "offsets_inside": (
                [
                    i
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                    for j in range(a[0] + 1, a[1] + 1)
                ],
                [
                    j
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                    for j in range(a[0] + 1, a[1] + 1)
                ],
            ),
            "raw": batch,
        }

        if not self.test:

            negatives = [
                np.random.choice([e for e in cands if e != a[2]])
                if len([e for e in cands if e != a[2]]) > 0
                else get_negative_entity()
                for b in batch["raw"]
                for a, cands in zip(b["anchors"], b["candidates"])
                if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
            ]

            targets = [
                a[2]
                for b in batch["raw"]
                for a in b["anchors"]
                if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
            ]

            assert len(targets) == len(negatives)

            batch_upd = {
                **(
                    {
                        f"trg_{k}": v
                        for k, v in self.tokenizer(
                            targets,
                            return_tensors="pt",
                            padding=True,
                            max_length=self.max_length,
                            truncation=True,
                        ).items()
                    }
                    if not self.test
                    else {}
                ),
                **(
                    {
                        f"neg_{k}": v
                        for k, v in self.tokenizer(
                            [e for e in negatives if e],
                            return_tensors="pt",
                            padding=True,
                            max_length=self.max_length,
                            truncation=True,
                        ).items()
                    }
                    if not self.test
                    else {}
                ),
                "neg_mask": torch.tensor([e is not None for e in negatives]),
            }

            batch = {**batch, **batch_upd}

        return batch
