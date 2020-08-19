from typing import Any
import importlib
import random
import numpy as np

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

from datasetreaders import MathDatasetReader
from models import TextClassifier

import pdb

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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

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

NUM_LAYERS = 3
NUM_FILTERS = 3


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
	                        lazy=True)
	train_data = reader.read("../../datasets/math/label-data/train-all")
	# val_data = reader.read("../../datasets/math/label-data/interpolate")


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


	optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.995), eps=1e-6)

	train_iterator = BucketIterator(batch_size=BATCH_SIZE,
	                                max_instances_in_memory=1024,
	                                sorting_keys=[("source_tokens", "num_tokens")])
	train_iterator = MultiprocessIterator(train_iterator, num_workers=16)
	train_iterator.index_with(vocab)

	val_iterator = BucketIterator(batch_size=BATCH_SIZE,
	                              max_instances_in_memory=1024,
	                              sorting_keys=[("source_tokens", "num_tokens")])
	val_iterator = MultiprocessIterator(val_iterator, num_workers=16)
	val_iterator.index_with(vocab)
	#pdb.set_trace()

	LR_SCHEDULER = {
	"type": "exponential",
	"gamma": 0.5,
	"last_epoch": -1
	}
	lr_scheduler = LearningRateScheduler.from_params(optimizer, Params(LR_SCHEDULER))
	

	# TRAIN
	trainer = Trainer(model=model,
	                  optimizer=optimizer,
	                  iterator=train_iterator,
	                  validation_iterator=None,
	                  train_dataset=train_data,
	                  validation_dataset=None,
	                  patience=None,
	                  shuffle=True,
	                  num_epochs=1,
	                  summary_interval=100,
	                  learning_rate_scheduler=lr_scheduler,
	                  cuda_device=CUDA_DEVICES,
	                  grad_norm=5,
	                  grad_clipping=5,
	                  model_save_interval=600,
	                  serialization_dir=args.serialization_dir,
	                  keep_serialized_model_every_num_seconds=3600,
	                  should_log_parameter_statistics=True,
	                  should_log_learning_rate=True
	                  )
	trainer.train()


if __name__ == '__main__':
	main()



