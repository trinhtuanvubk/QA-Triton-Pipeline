
import os
import pandas as pd

def get_intent_labels(path):
    return [label.strip() for label in open(os.path.join(path), 'r', encoding='utf-8')]


def get_slot_labels(path):
    return [label.strip() for label in open(os.path.join(path), 'r', encoding='utf-8')]

def post_processing(input_ids, slots, tokenizer):
    list_raw_slots = []
    for index in range(input_ids.shape[0]):
        list_tokens = tokenizer.convert_ids_to_tokens(input_ids[index])
        list_index_g = [index for index, i in enumerate(list_tokens) if 'Ġ' in i]
        list_index_g = [1] + list_index_g
        raw_slots = []
        for j, token in enumerate(list_tokens):
            if j in list_index_g:
                raw_slots.append(slots[index][j])
        list_raw_slots.append(raw_slots)
    return list_raw_slots




def get_response(intent, predicted_entities, mapping_df):
    # get response from intent and entity
    entities = {ent['entity']: ent['value'] for ent in predicted_entities}
    intent_rows = mapping_df[mapping_df['intent'] == intent]

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