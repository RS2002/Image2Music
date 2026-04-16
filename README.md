# Image2Music

**Article:** Zijian Zhao*, Dian Jin, Zijing Zhou, "[Zero-Effort Image-to-Music Generation: An Interpretable RAG-based VLM Approach](https://arxiv.org/abs/2509.22378)", ACM ICMR 2026



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
@article{zhao2025zero,
  title={Zero-Effort Image-to-Music Generation: An Interpretable RAG-based VLM Approach},
  author={Zhao, Zijian and Jin, Dian and Zhou, Zijing},
  journal={arXiv preprint arXiv:2509.22378},
  year={2025}
}
```



## 5. Links

Some websites provide the service for abc2midi and midi2abc:

[midi2abc (marmooo.github.io)](https://marmooo.github.io/midi2abc/)

[ABC notation converter - Nota ABC (notabc.app)](https://notabc.app/abc-converter/)