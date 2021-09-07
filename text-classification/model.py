from torch.nn import Module
from transformers import AutoConfig, AutoModelForSequenceClassification


class SeqClsModel(Module):
    def __init__(self, model_name_or_path, num_labels):
        super(SeqClsModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        
    def forward(self, **kwargs):
        return self.model(**kwargs)