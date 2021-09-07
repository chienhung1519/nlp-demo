from typing import List, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        label_mapping: Dict[str, int],
        max_len: int,
        model_name_or_path: str,
    ):
    """
    Arguments:
     - data: list of instances containng text and label keys.
     - label_mapping: transfer label from text to int.
     - max_len: maximum length of text input.
     - model_name_or_path: transformers model name or path.
    """
        self.data = data
        self._label2idx = label_mapping
        self._idx2label = {idx: label for label, idx in self._label2idx.items()}
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]

    @property
    def num_classes(self) -> int:
        return len(self._label2idx)

    def label2idx(self, label: str):
        return self._label2idx[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

    def collate_fn(self, samples: List[Dict]) -> Dict:
        texts = [sample["text"] for sample in samples]
        result = self.tokenizer(texts, max_length=self.max_len, padding=True, truncation=True)
        result["labels"] = [self.label2idx(sample["label"]) for sample in samples]
        return result