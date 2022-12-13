import torch
import re
import yaml

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from src.models.classifier import DIETClassifier, DIETClassifierConfig
from transformers import BertTokenizerFast

def load_test_dataset(list_intents, list_entities, path_of_dataset):
  """
        Load file test dataset and prepare formated data for evaluation
        :param list_intents: list of intents
        :param list_entities: list of entities
        :param path_of_dataset: path of file dataset
        :return: List[Dict[]] of data follow each intent
  """
  print("Intents:",list_intents)
  print("Entities:", list_entities)
  with open(path_of_dataset) as file:
    documents = yaml.full_load(file)

  formated_data_test = []
  for item, docs in documents.items():
    target_intents = [list_intents.index(item)]* len(docs)
    target_entities = []
    texts =[]

    for doc in docs:
        entities = re.findall(r'\[\w+\]\(\w+\)', doc)
        text = doc

        text_split = doc.split()
        tar_en_sen = [0]* len(text_split)

        for entity in entities:
          text = text.replace(entity,re.findall(r'\[(\w+)\]', entity)[0])
          tar_en_sen[text_split.index(entity)] = list_entities.index(re.findall(r'\((\w+)\)', entity)[0])

        target_entities.append(tar_en_sen)
        texts.append(text)
    formated_data_test.append({"intent": item, "target_intents": target_intents, "target_entities": target_entities, "texts": texts})

  return formated_data_test

def tokenize(tokenizer,device,sentences):
        """
        Tokenize sentences using tokenizer.
        :param sentences: list of sentences
        :return: tuple(tokenized sentences, offset_mapping for sentences)
        """
        inputs = tokenizer(sentences, return_tensors="pt", return_attention_mask=True, return_token_type_ids=True,
                                return_offsets_mapping=True,
                                padding=True, truncation=True)

        offset_mapping = inputs["offset_mapping"]
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"}

        return inputs, offset_mapping

def convert_intent_logits(intent_logits: torch.tensor):
        """
        Convert logits from model to predicted intent,

        :param intent_logits: output from model
        :return: dictionary of predicted intent
        """
        softmax = torch.nn.Softmax(dim=-1)
        softmax_intents = softmax(intent_logits)

        predicted_intents = []
        for sentence in softmax_intents:
            # sentence = sentence[0]

            # sorted_sentence = sentence.clone()
            # sorted_sentence, _ = torch.sort(sorted_sentence)

            
            max_probability = torch.argmax(sentence)
            

            predicted_intents.append(max_probability)  

        return predicted_intents

def convert_entities_logits( entities_logits: torch.tensor, offset_mapping: torch.tensor):
        """
        Convert logits to predicted entities

        :param entities_logits: entities logits from model
        :param offset_mapping: offset mapping for sentences
        :return: list of predicted entities
        """
        # softmax = torch.nn.Softmax(dim=-1)
       
        # softmax_entities = softmax(entities_logits)

        predicted_entities = []

        for sentence, offset in zip(entities_logits, offset_mapping):
            predicted_entities.append([])
            latest_entity = None
            for word, token_offset in zip(sentence, offset[1:]):
                # max_probability = torch.argmax(word)
                # if word[max_probability] >= 0.5  and max_probability != 0:
                #     if max_probability != latest_entity:
                #         latest_entity = max_probability
                #         predicted_entities[-1].append({
                #             "entity_name": max_probability,
                #             "start": token_offset[0].item(),
                #             "end": token_offset[1].item()
                #         })
                #     else:
                #         predicted_entities[-1][-1]["end"] = token_offset[1].item()
                # else:
                #     latest_entity = None 
                
                if  word != 0:
                    if word != latest_entity:
                        latest_entity = word
                        predicted_entities[-1].append({
                            "entity_name": word,
                            "start": token_offset[0].item(),
                            "end": token_offset[1].item()
                        })
                    else:
                        predicted_entities[-1][-1]["end"] = token_offset[1].item()
                else:
                    latest_entity = None 
                
        return predicted_entities

def predict(tokenizer,device,model,sentences):
        """
        Predict intent and entities from sentences.

        :param sentences: list of sentences
        :return: list of prediction
        """
        inputs, offset_mapping = tokenize(tokenizer= tokenizer,device= device ,sentences=sentences)
        outputs = model(**inputs)
        logits = outputs["logits"]
        predicted_intents = convert_intent_logits(intent_logits=logits[1])
        predicted_entities = convert_entities_logits(entities_logits=logits[0], offset_mapping=offset_mapping)
        predicted_outputs = {}
        predicted_outputs["intents"] = []
        predicted_outputs["entities"] = []
        predicted_outputs["texts"] = []
        for sentence, intent_sentence, entities_sentence in zip(sentences, predicted_intents, predicted_entities):
            
            predicted_outputs["intents"].append(intent_sentence)
            sentence_split = sentence.split()
            entity_sen = [0] * len(sentence_split)
            for entity in entities_sentence:
                text_entity = sentence[entity["start"]: entity["end"]]
                if text_entity in sentence_split:
                  entity_sen[sentence_split.index(text_entity)] = entity['entity_name']
            
            predicted_outputs["entities"].append(entity_sen)
            # predicted_outputs[-1].update("intent:",intent_sentence)
            # predicted_outputs[-1].update({"entities": entities_sentence})
            # for entity in predicted_outputs[-1]["entities"]:
            #     entity["text"] = sentence[entity["start"]: entity["end"]]

            #     if self.synonym_dict.get(entity["text"], None):
            #         entity["original_text"] = entity["text"]
            #         entity["text"] = self.synonym_dict[entity["text"]]

            predicted_outputs["texts"].append(sentence) 
        return predicted_outputs
def evaluation(tokenizer,device,model,test_dataset):
    intents_true = []
    intents_pre = []
    entities_true = []
    entities_pre = []
    for i in tqdm(test_dataset, desc ="Predict Processing ...."):
        intents_true.extend(i['target_intents'])
        entities_true.extend([entity for sen_en in i['target_entities'] for entity in sen_en])
        if len(i['target_intents']) > 300:
          ind_last =0 
          for ind in range(20,len(i['texts']),50):
              predicted_output = predict(tokenizer,device,model,i["texts"][ind_last:ind])
              intents_pre.extend(predicted_output["intents"])
              entities_pre.extend([entity for sen_en in predicted_output["entities"] for entity in sen_en])
              ind_last = ind
          predicted_output = predict(tokenizer,device,model,i["texts"][ind_last:])
          intents_pre.extend(predicted_output["intents"])
          entities_pre.extend([entity for sen_en in predicted_output["entities"] for entity in sen_en])
            
        else: 
          
          predicted_output = predict(tokenizer,device,model,i["texts"])
          intents_pre.extend(predicted_output["intents"])
          entities_pre.extend([entity for sen_en in predicted_output["entities"] for entity in sen_en])

    intents_true = np.array(intents_true)
    intents_pre = np.array(torch.tensor(intents_pre,device='cpu'))
    entities_true = np.array(entities_true)
    entities_pre = np.array(torch.tensor(entities_pre,device='cpu'))

    precision_intent, recall_intent, f1_intent, _ = precision_recall_fscore_support(intents_true, intents_pre, average='macro')
    print("Result Evaluation of Intent Predict: \n Precision: {:.2f} \t Recall: {:.2f} \t F1-score: {:.2f}".format(precision_intent, recall_intent, f1_intent))
    precision_entity, recall_entity, f1_entity, _ = precision_recall_fscore_support(entities_true, entities_pre, average='macro')
    print("Result Evaluation of Entity Predict: \n Precision: {:.2f} \t Recall: {:.2f} \t F1-score: {:.2f}".format(precision_entity, recall_entity, f1_entity))
    
if __name__=="__main__":

    #Load config 
    config_file = "src/config.yml"
    f = open(config_file, "r")
    config = yaml.load(f)
    model_config_dict = config.get("model", None)
    print("Load model from : ",model_config_dict['model'])
    util_config = config.get("util", None)

    device = torch.device(model_config_dict["device"]) if torch.cuda.is_available() else torch.device("cpu")
    # Intents list
    intents = model_config_dict["intents"]
    # Entities list
    entities = ["O"] + model_config_dict["entities"]

    #Load model
    model_config_attributes = ["model", "intents", "entities"]
    model_config = DIETClassifierConfig(**{k: v for k, v in model_config_dict.items() if k in model_config_attributes})

    tokenizer = BertTokenizerFast.from_pretrained(model_config_dict["tokenizer"])
    model = DIETClassifier(config=model_config, use_dot_product= model_config_dict["use_dot_product"])

    model.to(device)

    test_dataset = load_test_dataset(list_intents= intents, list_entities= entities, path_of_dataset= "dataset/test_dataset.yml")
    evaluation(tokenizer= tokenizer,device= device,model= model,test_dataset= test_dataset)
