import json
import numpy as np
import torch
from torch import autocast
from pathlib import Path
import pandas as pd
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import triton_python_backend_utils as pb_utils

from torch.utils.dlpack import to_dlpack, from_dlpack

from transformers import RobertaTokenizer, Wav2Vec2Processor, T5ForConditionalGeneration, AutoTokenizer
from loguru import logger
from utils import crf_decode, util
from utils.decoder import *
from utils.text import text_to_sequence

ROBERTA_CONFIG = "config/roberta_config"
WAV2VEC2_CONFIG = "config/wav2vec2_config"
T5_CONFIG = "config/t5_config"
INTENT_LABEL = "config/roberta_labels/intent_label.txt"
SLOT_LABEL = "config/roberta_labels/slot_label.txt"
MAPPING_DATA = "config/test_cases.xlsx"
T5_PATH = "t5_paraphrase"
LM_PATH = "ngram_model/4gram_small.arpa"

cur_folder = Path(__file__).parent
roberta_tokenizer_path = str(cur_folder/ROBERTA_CONFIG)
wav2vec_processor_path = str(cur_folder/WAV2VEC2_CONFIG)
t5_tokenizer_path = str(cur_folder/T5_CONFIG)
intent_label_path = str(cur_folder/INTENT_LABEL)
slot_label_path = str(cur_folder/SLOT_LABEL)
mapping_data_path = str(cur_folder/MAPPING_DATA)
t5_path = str(cur_folder/T5_PATH)
lm_path = str(cur_folder/LM_PATH)


# load mapping excel file
mapping_df = pd.read_excel(mapping_data_path).dropna(how='all')


class TritonPythonModel:
    def initialize(self, args):
        # parse model_config
        self.model_config = model_config = json.loads(args["model_config"])
        # get last configuration
        last_output_config = pb_utils.get_output_config_by_name(model_config, "output")
        #convert triton type to numpy type/
        self.last_output_dtype = pb_utils.triton_string_to_numpy(last_output_config["data_type"])

        # load wav2vec2 processor
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_processor_path, local_files_only=True)

        # load roberta tokenizer
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_tokenizer_path, local_files_only=True)

        # load dict intent and dict tag
        intent_label_lst = util.get_intent_labels(intent_label_path)
        self.dict_intents = {i: intent for i, intent in enumerate(intent_label_lst)}

        slot_label_lst = util.get_slot_labels(slot_label_path)
        self.dict_tags = {i: tag for i, tag in enumerate(slot_label_lst)}

        # load t5 tokenizer and t5 model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_path, local_files_only=True).to('cuda')
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_tokenizer_path, local_files_only=True)
        vocab_dict = self.processor.tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())

        # Lower case ALL letters
        vocab = []
        for _, token in sort_vocab:
            vocab.append(token.lower())

        # replace the word delimiter with a white space since the white space is used by the decoders
        vocab[vocab.index(self.processor.tokenizer.word_delimiter_token)] = ' '
        
        self.beam_decoder = BeamCTCDecoder(vocab, lm_path=lm_path,
                                 alpha=1.5, beta=0.5,
                                 cutoff_top_n=40, cutoff_prob=1.0,
                                 beam_width=32, num_processes=16,
                                 blank_index=vocab.index(self.processor.tokenizer.pad_token))
        
        
        # string dtype
        self._dtypes = [np.bytes_, np.object_]

    def execute(self, requests):
        responses = []
        for request in requests:
            # ==== Wav2Vec2 ====
            inp = pb_utils.get_input_tensor_by_name(request, "input")
            audio_array = inp

            w2v2_request = pb_utils.InferenceRequest(
                model_name="wav2vec2",
                requested_output_names=["output"],
                inputs=[audio_array],
            )

            trans_response = w2v2_request.exec()

            if trans_response.has_error():
                raise pb_utils.TritonModelException(trans_response.error().message())
            else:
                logits = pb_utils.get_output_tensor_by_name(
                    trans_response, "output")
                logits = from_dlpack(logits.to_dlpack()).clone()
                
            logger.debug("logit shape:{}".format(logits.shape))
            
            beam_decoded_outputs, _ = self.beam_decoder.decode(logits)
            transcriptions = [output[0].strip() for output in beam_decoded_outputs]

            # logger.debug("beam dec:{}".format(beam_decoded_outputs.shape))
            # transcriptions = beam_decoded_output[0]
            
            
            # prediction = torch.argmax(logits, axis=-1)
            # logger.debug("prediction shape:{}".format(prediction.shape))
            # raw_transcription = self.processor.batch_decode(prediction)
            # transcriptions = [sen.lower() for sen in raw_transcription]
            logger.debug("transcription:{}".format(transcriptions))
            
            # ==== JointBert ====
            # string = "customer service"
            joinbert_inputs = self.roberta_tokenizer(transcriptions, padding=True, truncation=True, return_tensors='pt')
            _input_ids = joinbert_inputs.input_ids
            _attention_mask = joinbert_inputs.attention_mask

            input_ids = pb_utils.Tensor(
                "input_ids",
                _input_ids.numpy().astype(np.int64)
            )
            
            attention_mask = pb_utils.Tensor(
                "attention_mask",
                _attention_mask.numpy().astype(np.int64)
            )

            jointbert_request = pb_utils.InferenceRequest(
                model_name="jointbert",
                requested_output_names=["intent_logits", "slot_logits", "transitions", "start_transition", "end_transition"],
                inputs=[input_ids, attention_mask],
            )
            response = jointbert_request.exec()


            intent_logits = pb_utils.get_output_tensor_by_name(response, "intent_logits")
            intent_logits = from_dlpack(intent_logits.to_dlpack()).clone()

            slot_logits = pb_utils.get_output_tensor_by_name(response, "slot_logits")
            slot_logits = from_dlpack(slot_logits.to_dlpack()).clone()

            transitions = pb_utils.get_output_tensor_by_name(response, "transitions")
            transitions = from_dlpack(transitions.to_dlpack()).clone()

            start_transition = pb_utils.get_output_tensor_by_name(response, "start_transition")
            start_transitions = from_dlpack(start_transition.to_dlpack()).clone()

            end_transition = pb_utils.get_output_tensor_by_name(response, "end_transition")
            end_transitions = from_dlpack(end_transition.to_dlpack()).clone()

            # decode crf 
            best_tags_list = crf_decode.decode(slot_logits, transitions, start_transitions, end_transitions)

            intent_preds = intent_logits.argmax(dim=-1).numpy().tolist()
            logger.debug(f"{intent_preds}-{best_tags_list}")

            intentions =  [self.dict_intents[i] for i in intent_preds]
            slots = util.post_processing(_input_ids, best_tags_list, self.roberta_tokenizer)
            slots = [[self.dict_tags[i] for i in tag] for tag in slots]

            logger.debug(f"{intentions}-{slots}")

            # ==== T5 ====
            # preprocess slots
            # [['O', 'O', 'O', 'O', 'B-TOP_UP', 'B-MY', 'B-NUMBER', 'O']]
            slots = ([list(set([element.replace("B-","").replace("I-","") for element in slot if element != "O"])) for slot in slots])
            tokenized_transcriptions = [sent.split() for sent in transcriptions]
            dict_slot = [[{"entity":s,"value":tokenized_transcriptions[id_sent][index]} for index, s in enumerate(slot)] for id_sent, slot in enumerate(slots)]
            answers = [util.get_response(intent, slot, mapping_df) for (intent, slot) in zip(intentions,dict_slot)]
            
            num_return_sequence = 10
            batch = self.t5_tokenizer(['Paraphrasing this sentence: ' + answer for answer in answers],
                    truncation=True,
                    padding='longest',
                    max_length=256,
                    return_tensors="pt").to("cuda")
            
            # print(model.generate.__dict__)
            translated = self.t5_model.generate(**batch,
                                max_length=256,
                                num_beams=10,
                                num_return_sequences=num_return_sequence,
                                do_sample=True,
                                temperature=1.3)
            # num_sentences = len(answers)
            all_tgt_texts = [self.t5_tokenizer.batch_decode(translated[i*num_return_sequence:(i+1)*num_return_sequence], skip_special_tokens=True) for i in range(len(answers))]
            
            final_anwers = [random.choice(answer) for answer in all_tgt_texts]

            # # ==== Sending Response ====
            # last_output_tensor = pb_utils.Tensor(
            #     "paraphrase_answer", np.array([i.encode('utf-8') for i in final_anwers], dtype=self._dtypes[0]))
            
            # inference_response = pb_utils.InferenceResponse([last_output_tensor])
        
            # responses.append(inference_response)

            # return responses

            # ==== VIT2 ====
            text_norm = text_to_sequence(final_anwers[0], ["english_cleaners2"])
            phoneme_ids = torch.LongTensor(text_norm)
            text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
            text = np.asarray(text, dtype=np.int64)
            tensor_text = pb_utils.Tensor("input", text)

            text_lengths = np.array([text.shape[1]], dtype=np.int64)
            tensor_text_lengths = pb_utils.Tensor("input_lengths", text_lengths)

            scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
            tensor_scales = pb_utils.Tensor("scales", scales)

            vits_request = pb_utils.InferenceRequest(
                model_name="vits2",
                requested_output_names=["output"],
                inputs=[tensor_text, tensor_text_lengths, tensor_scales],
            )
            vits2_response = vits_request.exec()
            audio = pb_utils.get_output_tensor_by_name(vits2_response, "output")
            audio = from_dlpack(audio.to_dlpack()).clone()
            audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)

            logger.debug(audio.shape)
            

            # ==== Sending Response ====
            last_output_tensor = pb_utils.Tensor("output", audio)
            
            inference_response = pb_utils.InferenceResponse([last_output_tensor])
        
            responses.append(inference_response)

            return responses



