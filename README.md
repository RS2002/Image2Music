# Image2Music

**Article:** Zijian Zhao*, Dian Jin, Zijing Zhou"[Zero-Effort Image-to-Music Generation: An Interpretable RAG-based VLM Approach](https://arxiv.org/abs/2509.22378)" (under review)



## 1. Worflow

![](./image/main.png)



## 2. Dataset

[amaai-lab/MidiCaps · Datasets at Hugging Face](https://huggingface.co/datasets/amaai-lab/MidiCaps)

Please rename the `train.json` as `meta.txt`.

The data process part is based on the code of [jwdj/EasyABC: EasyABC (github.com)](https://github.com/jwdj/EasyABC).



## 3. Run the Model

```shell
python main.py
```



## 4. Citation

```
@misc{zhao2025zeroeffortimagetomusicgenerationinterpretable,
      title={Zero-Effort Image-to-Music Generation: An Interpretable RAG-based VLM Approach}, 
      author={Zijian Zhao and Dian Jin and Zijing Zhou},
      year={2025},
      eprint={2509.22378},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2509.22378}, 
}
```



## 5. Links

Some websites provide the service for abc2midi and midi2abc:

[midi2abc (marmooo.github.io)](https://marmooo.github.io/midi2abc/)

[ABC notation converter - Nota ABC (notabc.app)](https://notabc.app/abc-converter/)