from EasyABC.midi2abc import my_midi2abc
from PIL import Image
import faiss
import json
import os

def pipeline(device, model, processor, process_vision_info, clip_model, clip_processor, index, texts, midi_path, clip_length, max_length):
    print("System Initialized. ")
    while True:
        print("Type your img path or 'exit / quit' to quit.\n")
        img_path = input("Please Input: ")
        if img_path.lower() in ["exit", "quit"]:
            break

        if not os.path.exists(img_path):
            print("No Such Imgage!")
            continue

        # 1. generate the description for the image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": "Describe this image./no_think"},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        # inputs = processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        #     **video_kwargs,
        # )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=clip_length)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        description = output_text[0]

        print("Image Description: "+description)

        # 2. RAG by img and generated description
        index_list = []

        inputs = clip_processor(text=[description], padding=True, return_tensors="pt").to(device)
        text_embedding = clip_model.get_text_features(**inputs)
        text_embedding = text_embedding.cpu().detach().numpy()
        faiss.normalize_L2(text_embedding)
        _, I = index.search(text_embedding, 1)
        I = I.reshape(-1)
        index_list.append(I[0])

        image = Image.open(img_path)
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        image_embedding = clip_model.get_image_features(**inputs)
        image_embedding = image_embedding.cpu().detach().numpy()
        faiss.normalize_L2(image_embedding)
        _, I = index.search(image_embedding, 1)
        I = I.reshape(-1)
        if I[0] != index_list[0]:
            index_list.append(I[0])
        print("Retrieved Index: ", index_list)

        # 3. set prompt:
        prompt = "Generate a music with ABC-format for the given image. The description of the image is: "+description + "\n"
        prompt += "You just need to create music that corresponds to the image, regardless of the terms used, such as emotions or scenarios."
        example = []

        for i in I:
            text_description = texts[i]
            path = midi_path[i]

            try:
                abc = my_midi2abc(path)
            except Exception as e:
                continue

            example.append("Text description: "+text_description+"\n Corresponding Music: "+abc+"\n")

        num = len(example)
        if num != 0:
            prompt = prompt + f"{num} examples are provided as: \n"
            for i in range(num):
                prompt = prompt + example[i]
            prompt += "Please note the example is just provided for reference. Do not generate same song as the example."


        print("Final Prompt: " + prompt)

        # 4. generate music:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": prompt+"/think"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        # inputs = processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        #     **video_kwargs,
        # )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=max_length)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        description = output_text[0]

        print("Generated Result:")
        print(description)