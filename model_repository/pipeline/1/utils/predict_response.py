import numpy as np
import os
import torch
import pickle
import pandas as pd

from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.preprocessing import LabelEncoder

from flair.data import Sentence
from flair.models import SequenceTagger



# mapping_data = pd.read_excel().dropna(how='all')

def get_response(self, intent, predicted_entities):

        entities = {ent['entity']: ent['value'] for ent in predicted_entities}
        intent_rows = self.mapping_data[self.mapping_data['intent'] == intent]

        for _, row in intent_rows.iterrows():
            all_match = True
            for i in range(1, 4):
                entity_col = f'entity_{i}'
                value_col = f'entity_{i}_value'
                if pd.isna(row[entity_col]):
                    continue
                elif row[entity_col] in entities and (pd.isna(row[value_col]) or entities[row[entity_col]] == row[value_col]):
                    continue
                else:
                    all_match = False
                    break
            if all_match:
                return row['expected prompt']

        if not intent_rows.empty:
            return intent_rows.sample(n=1)['expected prompt'].values[0]

        return "Sorry, I couldn't find a response for your request."


class text_response():
    def __init__(self, args):
        self.args = args
        self.mapping_data = pd.read_excel(args.excel_response).dropna(how='all')

        self.entity_model = SequenceTagger.load(args.bert_entity_model)

        with open(args.label_encoder, 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.intent_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.intent_model = RobertaForSequenceClassification.from_pretrained(args.bert_intent_model, local_files_only=True).to(args.device)
        self.intent_model.eval()

        self.para_tokenizer = AutoTokenizer.from_pretrained(args.paraphrase_model, local_files_only=True)
        self.para_model =  AutoModelForSeq2SeqLM.from_pretrained(args.paraphrase_model, local_files_only=True).to(args.device)

    def get_response(self, intent, predicted_entities):

        entities = {ent['entity']: ent['value'] for ent in predicted_entities}
        intent_rows = self.mapping_data[self.mapping_data['intent'] == intent]

        for _, row in intent_rows.iterrows():
            all_match = True
            for i in range(1, 4):
                entity_col = f'entity_{i}'
                value_col = f'entity_{i}_value'
                if pd.isna(row[entity_col]):
                    continue
                elif row[entity_col] in entities and (pd.isna(row[value_col]) or entities[row[entity_col]] == row[value_col]):
                    continue
                else:
                    all_match = False
                    break
            if all_match:
                return row['expected prompt']

        if not intent_rows.empty:
            return intent_rows.sample(n=1)['expected prompt'].values[0]

        return "Sorry, I couldn't find a response for your request."

    def generate_responses(self, input_sentence, number_return_sequence=5):
        sentence = Sentence(input_sentence)
        self.entity_model.predict(sentence)
        predicted_entities = [{"entity": entity.tag, "value": entity.text} for entity in sentence.get_spans('ner')]

        # Intent classification
        encoded_inputs = self.intent_tokenizer([input_sentence], padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].to(self.args.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.args.device)

        with torch.no_grad():
            outputs = self.intent_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            _, predicted_indices = torch.max(logits, dim=1)
            predicted_intent = self.label_encoder.inverse_transform(predicted_indices.cpu().numpy())[0]

        # print(predicted_entities, predicted_intent)

        # Response mapping
        base_response = self.get_response(predicted_intent, predicted_entities)
        if base_response == 'No matching response found.':
            return [base_response]

        # Paraphrasing
        batch = self.para_tokenizer(['Paraphrasing this sentence: ' + base_response],
                            truncation=True,
                            padding='longest',
                            max_length=256,
                            return_tensors="pt").to(self.args.device)
        translated = self.para_model.generate(**batch,
                                        max_length=256,
                                        num_beams=number_return_sequence+2,
                                        num_return_sequences=number_return_sequence,
                                        temperature=1.3)
        paraphrased_responses = self.para_tokenizer.batch_decode(translated, skip_special_tokens=True)
        # print(paraphrased_responses)
        return paraphrased_responses
    

# mapping_data = pd.read_excel("model_repository/test_cases.xlsx").dropna(how='all')

# # Load entities model
# entities_ckpt = 'model_repository/bert_entity_model/best-model.pt'
# entities_model = SequenceTagger.load(entities_ckpt)

# # Load intent model
# with open('model_repository/bert_intent_model/label_encoder.pkl', 'rb') as f:
#     label_encoder = pickle.load(f)
# device = 'cuda'
# ckpt = "model_repository/bert_intent_model"
# intent_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# intent_model = RobertaForSequenceClassification.from_pretrained(ckpt, local_files_only=True).to(device)
# intent_model.eval()

# # Load paraphrase model
# checkpoints_dir = "model_repository/t5_model"
# para_tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir, local_files_only=True)
# para_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoints_dir, local_files_only=True).to(device)

# def get_response(intent, predicted_entities):
#     entities = {ent['entity']: ent['value'] for ent in predicted_entities}

#     intent_rows = mapping_data[mapping_data['intent'] == intent]

#     for _, row in intent_rows.iterrows():
#         all_match = True
#         for i in range(1, 4):
#             entity_col = f'entity_{i}'
#             value_col = f'entity_{i}_value'
#             if pd.isna(row[entity_col]):
#                 continue
#             elif row[entity_col] in entities and (pd.isna(row[value_col]) or entities[row[entity_col]] == row[value_col]):
#                 continue
#             else:
#                 all_match = False
#                 break
#         if all_match:
#             return row['expected prompt']

#     if not intent_rows.empty:
#         return intent_rows.sample(n=1)['expected prompt'].values[0]

#     return "Sorry, I couldn't find a response for your request."


# def generate_responses(input_sentence, number_return_sequence=8):
#     sentence = Sentence(input_sentence)
#     entities_model.predict(sentence)
#     predicted_entities = [{"entity": entity.tag, "value": entity.text} for entity in sentence.get_spans('ner')]

#     # Intent classification
#     encoded_inputs = intent_tokenizer([input_sentence], padding='max_length', truncation=True, return_tensors='pt')
#     input_ids = encoded_inputs['input_ids'].to(device)
#     attention_mask = encoded_inputs['attention_mask'].to(device)
#     with torch.no_grad():
#         outputs = intent_model(input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         probabilities = torch.softmax(logits, dim=1)
#         _, predicted_indices = torch.max(logits, dim=1)
#         predicted_intent = label_encoder.inverse_transform(predicted_indices.cpu().numpy())[0]

#     # print(predicted_entities, predicted_intent)

#     # Response mapping
#     base_response = get_response(predicted_intent, predicted_entities)
#     if base_response == 'No matching response found.':
#         return [base_response]

#     # Paraphrasing
#     batch = para_tokenizer(['Paraphrasing this sentence: ' + base_response],
#                            truncation=True,
#                            padding='longest',
#                            max_length=256,
#                            return_tensors="pt").to(device)
#     translated = para_model.generate(**batch,
#                                      max_length=256,
#                                      num_beams=number_return_sequence+2,
#                                      num_return_sequences=number_return_sequence,
#                                      temperature=1.3)
#     paraphrased_responses = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
#     # print(paraphrased_responses)
#     return paraphrased_responses