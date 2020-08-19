from typing import Any
import importlib
import random
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper

from allennlp.data.iterators import BucketIterator, MultiprocessIterator
from allennlp.training.trainer import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler

from allennlp.nn.util import get_text_field_mask
from allennlp.common.util import START_SYMBOL, END_SYMBOL, prepare_global_logging
from allennlp.common.params import Params

from allennlp.predictors import Seq2SeqPredictor
from allennlp.nn import util as nn_util
from allennlp.common.util import sanitize

from datasetreaders import MathDatasetReader
from models import TextClassifier
from predictors import TextClassifierPredictor

import pdb
from tqdm import tqdm

import os
import argparse
import time
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)


serialization_path = './runs/classifier_' + time.strftime('%Y%m%d_%H:%M:%S', time.localtime())
parser = argparse.ArgumentParser(description='VAE Text Generation')
parser.add_argument('--model', '-m', type=str, default='lstm', choices=['lstm', 'cnn'])
parser.add_argument('--serialization_dir', '-s', type=str, default=serialization_path)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#CUDA_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
torch.backends.cudnn.benchmark = True


# PARAMETERS
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

NUM_LAYERS = 3   # for lstm classifier
NUM_FILTERS = 3	 # for cnn classifier


def main():
	###############################################################################################
	prepare_global_logging(serialization_dir=args.serialization_dir, file_friendly_logging=False)
	#DATA
	reader = MathDatasetReader(source_tokenizer=CharacterTokenizer(),
	                        target_tokenizer=CharacterTokenizer(),
	                        source_token_indexers={'tokens': SingleIdTokenIndexer(namespace='tokens')},
	                        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='tokens')},
	                        target=False,
	                        label=True,
	                        lazy=False)
	# train_data = reader.read("../../datasets/math/label-data/train-all")
	# val_data = reader.read("../../datasets/math/label-data/interpolate")
	val_data = reader.read("./generate_files")


	vocab = Vocabulary()
	vocab.add_tokens_to_namespace([START_SYMBOL, END_SYMBOL, ' ', '!', "'", '(', ')', '*', '+', ',', '-', '.', '/',
	                                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?',
	                                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
	                                    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
	                                    'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
	                                    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}'], namespace='tokens')
	vocab.add_tokens_to_namespace(['algebra', 'arithmetic', 'calculus', 'comparison',
	  								 'measurement', 'numbers', 'polynomials', 'probability'], namespace='labels')



	# MODEL
	embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
	                             embedding_dim=EMBEDDING_DIM)
	source_embedder = BasicTextFieldEmbedder({"tokens": embedding})

	if args.model == 'lstm':
		encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, 
											num_layers=NUM_LAYERS, batch_first=True))
	elif args.model == 'cnn':
		encoder = CnnEncoder(embedding_dim=EMBEDDING_DIM, num_filters=NUM_FILTERS, output_dim=HIDDEN_DIM)
	else:
		raise NotImplemented("The classifier model should be LSTM or CNN")


	model = TextClassifier(vocab=vocab,
				source_text_embedder=source_embedder,
	            encoder=encoder,
	            )
	model.to(device)


	if not Path(args.serialization_dir).exists() or not Path(args.serialization_dir).is_dir():
  		raise NotImplementedError("The model seems not to exist")
	with open(Path(args.serialization_dir) / "best.th", "rb") as model_path:
  		model_state = torch.load(model_path, map_location=nn_util.device_mapping(-1))
  		model.load_state_dict(model_state)
	model.eval()

	predictor = TextClassifierPredictor(model, dataset_reader=reader)

	# TEST
	correct = 0
	total = 0

	pbar = tqdm(val_data)
	batch_instance = list()
	batch_gt = list()

	idx_last = 0
	for idx, instance in enumerate(pbar):
		if idx != (idx_last + BATCH_SIZE):
			batch_instance.append(instance)
			batch_gt.append(instance.fields["labels"].label) # str
		else:
			idx_last = idx
			outputs = predictor.predict(batch_instance)
			for i, output in enumerate(outputs):
				if batch_gt[i] == output['predict_labels']:
					correct += 1
				total += 1
			batch_instance = list()
			batch_gt = list()
			pbar.set_description("correct/total %.3f" % (correct / total))

    

if __name__ == '__main__':
	main()



