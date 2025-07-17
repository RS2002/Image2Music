from EasyABC.midi2abc import my_midi2abc
import json
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np


meta_path = "./midicaps/meta.txt"

# def parse(meta_path):
#     training_set = []
#     index = 0
#     with open(meta_path, 'r') as file:
#         pbar = tqdm.tqdm(file)
#         for line in pbar:
#             try:
#                 data_dict = json.loads(line)
#                 path = data_dict['location']
#                 caption = data_dict['caption']
#                 abc = my_midi2abc("./midicaps/" + path)
#                 data = {"index": index, "music": abc, "description": caption}
#                 training_set.append(data)
#             except Exception as e:
#                 continue
#             index += 1
#     with open("train.json", 'w') as json_file:
#         json.dump(training_set, json_file, indent=4)
#
# parse(meta_path)

def text_encode(model, processor, meta_path, device, batch_size = 32):
    texts = []
    with open(meta_path, 'r') as file:
        pbar = tqdm(file)
        for line in pbar:
            data_dict = json.loads(line)
            caption = data_dict['caption']
            texts.append(caption)

    num_texts = len(texts)
    all_embeddings = None
    batches = (num_texts + batch_size - 1) // batch_size

    for i in tqdm(range(batches)):
        batch_texts = texts[i * batch_size: (i + 1) * batch_size]
        inputs = processor(text=batch_texts, padding=True, return_tensors="pt").to(device)
        embeddings = model.get_text_features(**inputs)
        embeddings = embeddings.cpu().detach().numpy()

        if all_embeddings is None:
            all_embeddings = embeddings
        else:
            all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)

    print(f"Embedding shapes: {all_embeddings.shape}")
    np.save("embedding.npy", all_embeddings)

device = torch.device("cuda:2")
model_id = "zer0int/LongCLIP-GmP-ViT-L-14"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()
torch.set_grad_enabled(False)
text_encode(model, processor, meta_path, device, 32)