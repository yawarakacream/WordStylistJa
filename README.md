# WordStylistJa

日本語文字の学習に対応させる

- `--dataset` データセット名

## [etlcdb](http://etlcdb.db.aist.go.jp/?lang=ja)

前処理として，`{--etlcdb_path}/ETL{*}.json` に etlcdb のバイナリデータを json 化したものを置いておく．  
抽出した画像も適当に置いておく．

`character.py` に対応する文字を予め書いておく．

### 学習

- `--etlcdb_path` etlcdb のパス
- `--etlcdb_names` etlcdb の名前
- `--etlcdb_process_type` 前処理の種類
- `--save_path` 保存先

入力

```
python train.py --etlcdb_path ./etlcdb_path --etlcdb_process_type original --etlcdb_names ETL4 ETL5 --save_path ./save_path/original
```

実行結果

1. `./etlcdb_path/ETL4.json`, `./etlcdb_path/ETL5.json` から画像の相対パスを取得
2. `./etlcdb_path/original/{1. で得た相対パス}` の画像を利用して学習
3. `./save_path/original` 以下に保存

### サンプリング

- `--save_path` 学習時と合わせる
- `--writers` `{--save_path}/writer2idx.json` を参照（キーの方を入れる）
- `--words` 出力する文字（ターミナルにはうまく打てないかも）

入力

```
python sampling.py --save_path ./save_path/original --writers ETL4_5001 ETL5_6001 --words "ね" "コ"
```

実行結果

1. `./save_path/original` の結果を利用（学習時と合わせる）
2. ETL4 の 5001 番と ETL5 の 6001 番が書いたつもりの "ね" と "コ" の画像を生成
3. `./save_path/original/generated` 以下に保存

---

# Official PyTorch Implementation of "WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models" - ICDAR 2023

<!-- 
[arXiv](https://arxiv.org/pdf/2303.16576.pdf) 
  -->
 <p align='center'>
  <b>
    <a href="https://arxiv.org/pdf/2303.16576.pdf">ArXiv Paper</a>
  </b>
</p> 

 
 <p align="center">
<img src=figs/wordstylist.png width="600"/>
</p>

> **Abstract:** 
>*Text-to-Image synthesis is the task of generating an image according to a specific text description. Generative Adversarial Networks have been considered the standard method for image synthesis virtually since their introduction. Denoising Diffusion Probabilistic Models are recently setting a new baseline, with remarkable results in Text-to-Image synthesis, among other fields. Aside its usefulness per se, it can also be particularly relevant as a tool for data augmentation to aid training models for other document image processing tasks. In this work, we present a latent diffusion-based method for styled text-to-text-content-image generation on word-level. Our proposed method is able to generate realistic word image samples from different writer styles, by using class index styles and text content prompts without the need of adversarial training, writer recognition, or text recognition. We gauge system performance with the Fréchet Inception Distance, writer recognition accuracy, and writer retrieval. We show that the proposed model produces samples that are aesthetically pleasing, help boosting text recognition performance, and get similar writer retrieval score as real data.*


## Dataset & Pre-processing

Download the ```data/words.tgz``` of IAM Handwriting Database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database.

Then, pre-process the word images by running:
```
python prepare_images.py
```
Before running the ```prepare_images.py``` code make sure you have changed the ```iam_path``` and ```save_dir``` to the corresponding ```data/words.tgz``` and the path to save the processed images.

## Training from scratch

To train the diffusion model run:
```
python train.py --iam_path path/to/processed/images --save_path path/to/save/models/and/results
```

## Trained Model

We provide the weights of a trained model, which you can download from: [trained_model](https://drive.google.com/file/d/1XVRUXSJw0PaNgrtFH_mNHceFO-Ouf_xz/view?usp=share_link).

## Sampling - Regenerating IAM

If you want to regenerate the full IAM training set you can run:
```
python full_sampling.py --save_path path/to/save/generated/images --models_path /path/to/trained/models
```

## Sampling - Single image

If you want to generate a single word with a random style you can run:
```
python sampling.py --save_path path/to/save/generated/images --models_path /path/to/trained/models --words ['hello']
```

## Citation

If you find the code useful for your research, please cite our paper:
```
@article{nikolaidou2023wordstylist,
  title={{WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models}},
  author={Nikolaidou, Konstantina and Retsinas, George and Christlein, Vincent and Seuret, Mathias and Sfikas, Giorgos and Smith, Elisa Barney and Mokayed, Hamam and Liwicki, Marcus},
  journal={arXiv preprint arXiv:2303.16576},
  year={2023}
}
```

## Acknowledgements

We would like to thank the researchers of [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [GANwriting](https://github.com/omni-us/research-GANwriting/tree/9e0d8a3a8327f00c67029dbf4a2fc1b0a88f730d), [SmartPatch](https://github.com/MattAlexMiracle/SmartPatch), and [HTR best practices](https://github.com/georgeretsi/HTR-best-practices/tree/main) for releasing their code.
