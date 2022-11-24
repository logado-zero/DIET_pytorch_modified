import torch
import re
import yaml

from tqdm import tqdm
import numpy as np
from src.models.classifier import DIETClassifier, DIETClassifierConfig
from src.models.wrapper import DIETClassifierWrapper
from transformers import BertTokenizerFast


if __name__=="__main__":

    #Load config 
    config_file = "src/config.yml"
    f = open(config_file, "r")
    config = yaml.load(f)
    model_config_dict = config.get("model", None)
    util_config = config.get("util", None)

    device = torch.device(model_config_dict["device"]) if torch.cuda.is_available() else torch.device("cpu")
    # Intents list
    intents = model_config_dict["intents"]
    # Entities list
    entities = ["O"] + model_config_dict["entities"]

    #Load model
    # model_config_attributes = ["model", "intents", "entities"]
    # model_config = DIETClassifierConfig(**{k: v for k, v in model_config_dict.items() if k in model_config_attributes})

    # tokenizer = BertTokenizerFast.from_pretrained(model_config_dict["tokenizer"])
    # model = DIETClassifier(config=model_config)

    # model.to(device)

    wrapper = DIETClassifierWrapper(config=config_file)



    #train
    #after training, wrapper will load best model automatically
    wrapper.train_model(save_folder="/content/drive/MyDrive/CLV/Chatbot/WeiBot/DIET_atis_org_model")