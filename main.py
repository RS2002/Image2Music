import numpy as np
import torch
import argparse
import faiss
from transformers import AutoModel, AutoTokenizer, AutoProcessor, CLIPProcessor, CLIPModel
from keye_vl_utils import process_vision_info
# from qwen_vl_utils import process_vision_info
# from transformers import Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
import json
from pipeline3 import pipeline

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--meta_path", type=str, default="./midicaps/meta.txt")
    parser.add_argument("--midi_root", type=str, default="./midicaps/")

    parser.add_argument("--embedding_path", type=str, default="./embedding.npy")
    parser.add_argument("--model_path", type=str, default="Kwai-Keye/Keye-VL-8B-Preview")
    # parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--clip_path", type=str, default="zer0int/LongCLIP-GmP-ViT-L-14")
    parser.add_argument("--clip_length", type=int, default=248)
    parser.add_argument("--max_length", type=int, default=8192)

    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--cuda_device", type=int, default=2)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # device setup
    torch.set_grad_enabled(False)
    cuda_device = args.cuda_device
    if not args.cpu and cuda_device is not None:
        device_name = "cuda:" + str(cuda_device)
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    print("Device Setup Successfully.")

    # VLM setup
    model_path = args.model_path
    model = AutoModel.from_pretrained(
        model_path,
        # torch_dtype="auto",
        # device_map="auto",
        trust_remote_code=True,
    ).to(device)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct"
    #     # , torch_dtype="auto", device_map="auto"
    # ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("VLM Setup Successfully.")

    # CLIP setup
    clip_path = args.clip_path
    clip_model = CLIPModel.from_pretrained(clip_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_path)
    clip_model.eval()
    print("CLIP Setup Successfully.")

    # faiss setup
    description_embedding = np.load(args.embedding_path)
    dimension = description_embedding.shape[1]
    faiss.normalize_L2(description_embedding)
    index = faiss.IndexFlatIP(dimension)
    index.add(description_embedding)
    print("FAISS Setup Successfully.")

    # load meta data
    meta_path = args.meta_path
    texts = []
    midi_path = []
    midi_root = args.midi_root
    with open(meta_path, 'r') as file:
        pbar = tqdm(file)
        for line in pbar:
            data_dict = json.loads(line)
            caption = data_dict['caption']
            texts.append(caption)
            path = midi_root + data_dict['location']
            midi_path.append(path)
    print("DATA Setup Successfully.")

    pipeline(device, model, processor, process_vision_info, clip_model, clip_processor, index, texts, midi_path,args.clip_length,args.max_length)


if __name__ == '__main__':
    main()