from src.models.wrapper import DIETClassifierWrapper
import torch

config_file = "src/config.yml"
save_path="/home/ubuntu/LongDo/DIET/model/_cosin_crf_entropy"

if __name__=="__main__":
    wrapper = DIETClassifierWrapper(config=config_file)


    #train
    #after training, wrapper will load best model automatically
    wrapper.train_model(save_folder=save_path)