<h1 align="center">
ReVision: Multimodal Instruction Rewriting with Tiny Vision language Models 
</h1>

## 🛠️ Install

1. Clone this repository and navigate to MobileVLM folder

2. Install Package
    ```Shell
    conda create -n vir python=3.10 -y
    conda activate vir
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Dataset

Data is here : https://huggingface.co/datasets/anonymoususerrevision/multimodal_query_rewrites

## Pretrain 
(This step is optional and a pretrained version of the model is already shared in the paper. If you still want to pretrain)
First prepare a model with randomly initialized parameters

    ```Shell
    python prepare_model_for_pretraining.py --vision_model_name_or_path google/siglip-base-patch16-256 --text_model_name_or_path OuteAI/Lite-Mistral-150M-v2-Instruct --dest ./ReVision-250M-64-16-random
    ```

In `pretrain.py`, change the huggingface cache appropriately `os.environ["HF_DATASETS_CACHE"]` and point to your local directoy. Also, if you planning to push the pretrained model to huggingface hub, change this `anonymoususerrevision/ReVision-250M-64-16` to your desired model identifier. It is strongly advise to thoroughly go through the pretraining code and change variable values as needed. For changing arguments and training hyper parameters, check `args.py`

For pretraining run the following command. 

    ```Shell
    python pretrain.py
    ```
    

## Fine Tune

Similar to pretrainig above, just change code and provide link to the appropriate dataset and run `python finetune.py` (or other variants provided)

## Infernece

Various inference scripts are provided under `test_*.py` and `evaluate.py`. For running baseline experiments with PaliGemma or Qwen, use appropriate processors and conditional generators from huggingface in `evaluate.py` and not use `ReVisionProcessor`, `ReVisionForConditionalGeneration`. Also the format of the prompt needs to be changed (specifically for Paligemma) in `datautils.py`. 

## Terms of Use

The code is released under Apache License 2.0 but the data is not. The image portion of the dataset comes from existing resources. To serve the research community better, we uploaded images.zip for better reproducing our work in research community. It must not be used for any other purposes. The use of these images must comply with the respective licenses attached with the image sources. This may be taken down at any time when requested by the original owner or owners of the referenced images.

