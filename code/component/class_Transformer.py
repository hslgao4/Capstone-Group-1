import pdb
from huggingface_hub import hf_hub_download
import torch
from transformers import TimeSeriesTransformerModel

file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly")

# during training, one provides both past and future values
# as well as possible additional features
pdb.set_trace()
outputs = model(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    # static_categorical_features = torch.zeros((64, 1), dtype=torch.int),
    static_real_features=batch["static_real_features"],
    # static_real_features =torch.zeros((64, 1), dtype=torch.int),
    future_values=batch["future_values"],
    future_time_features=batch["future_time_features"]
)

last_hidden_state = outputs.last_hidden_state

