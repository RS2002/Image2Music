from EasyABC.midi2abc import my_midi2abc
from PIL import Image
import faiss
import json
import os
from utils import *

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
        img_description = output_text[0]

        print("-"*50)
        print("Image Description: "+img_description)
        print("-"*50)

        # 2. RAG by img and generated description
        index_list = []

        inputs = clip_processor(text=[img_description], padding=True, return_tensors="pt").to(device)
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
        prompt = "Generate a music with ABC-format for the given image. The description of the image is: "+img_description + "\n"
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
            prompt += "Please note the example is just provided for reference. Do not generate same song as the example. "
            prompt += "Please describe the creating motivation after the ABC-format music. "

        print("Primary Prompt: " + prompt)
        print("-"*50)

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

        while True:
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

            print("Primary Result:")
            print(description)
            print("-"*50)

            # 5. refine music:
            abc_music, generation_motivation = generation_parse(description)
            if abc_music is None:
                print("Re-generate!")
                continue

            try:
                score = parse_abc(abc_music)
                score.write('midi', fp='./temp.mid')
                mid = muspy.read_midi('./temp.mid')
                metrics = evaluate(mid)
            except:
                print("ABC Parse Fail")
                print("Re-generate!")
                continue

            break

        prompt = f"""
        Please refine the ABC music based on the specified metrics to enhance its suitability for the accompanying image. \n
        The image is described as: {img_description} \n
        The ABC music is: {abc_music} \n
        The evaluated metrics are: \n
        - Pitch Range: {metrics[0]} \n
        - Number of Pitches Used: {metrics[1]} \n
        - Number of Pitch Classes Used: {metrics[2]} \n
        - Polyphony: {metrics[3]} \n
        - Scale Consistency: {metrics[4]} \n
        - Pitch Entropy: {metrics[5]} \n
        - Pitch Class Entropy: {metrics[6]} \n
        - Empty Beat Rate: {metrics[7]} \n
        Metric Definitions: \n
        - Pitch Range: Measures the span between the highest and lowest notes in semitones. A higher value is beneficial for expressive genres like orchestral music or jazz solos, while a lower value suits minimalist pieces or children's music, where a limited range creates focus and accessibility. \n
        - Number of Pitches Used: Counts distinct MIDI pitches (0-127) in the composition. Higher values are advantageous for complex genres like avant-garde or progressive metal, while lower values work well for folk music or pop melodies, where repetitive motifs require fewer distinct pitches. \n
        - Number of Pitch Classes Used: Tracks unique note names (C, C♯, etc., ignoring octaves). Higher counts are desirable in jazz fusion or atonal works exploring dissonance, while lower counts suit diatonic music like blues or beginner exercises, where tonal clarity is achieved through fewer pitch classes. \n
        - Polyphony: Calculates the average number of simultaneous notes. Higher density excels in contrapuntal genres like fugues or film scores, while lower values better serve monophonic traditions like Gregorian chant or vocal solos, prioritizing melodic purity. \n
        - Scale Consistency: Quantifies adherence to a single musical scale. Higher consistency is preferable for hymn-like tonal pieces seeking harmonic stability, while lower consistency benefits Romantic symphonies or prog rock, where dramatic key changes enhance emotional impact. \n
        - Pitch Entropy: Measures unpredictability in pitch selection using Shannon entropy. Higher entropy suits serialist compositions aiming for atonal equality, while lower entropy benefits pop music, where predictable pitch centers create catchy melodies. \n
        - Pitch Class Entropy: Assesses unpredictability across note names (ignoring octaves). Higher values serve dodecaphonic works demanding equal use of all 12 pitch classes, whereas lower values fit blues or folk genres that intentionally emphasize a subset of notes. \n
        - Empty Beat Rate: Indicates the proportion of beats without note onsets. Higher rates benefit reggae or ambient music, where strategic silences define rhythmic character, while lower rates drive high-energy genres like salsa or EDM, which require constant rhythmic momentum. \n
        Please refine the ABC music and provide your reasoning for the adjustments made.
        """
        print("Final Prompt: " + prompt)
        print("-" * 50)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": prompt + "/think"},
                ],
            }
        ]

        while True:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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

            print("Final Result:")
            print(description)
            print("-" * 50)

            # 6. compare result:
            abc_music, generation_motivation = generation_parse(description)
            if abc_music is None:
                print("Re-refine!")
                continue

            try:
                score = parse_abc(abc_music)
                score.write('midi', fp='./temp2.mid')
                mid = muspy.read_midi('./temp2.mid')
                metrics2 = evaluate(mid)
                print("Metrics: Pitch Range, Number of Pitches Used, Number of Pitch Classes Used, Polyphony, Scale Consistency, Pitch Entropy, Pitch Class Entropy, Empty Beat Rate")
                print("Before Refining: ", metrics)
                print("After Refining: ", metrics2)
            except:
                print("ABC Parse Fail")
                print("Re-refine!")
                continue

            break
        print("-" * 50)






