from typing import Dict, List

from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor

@Predictor.register("text_classifier_predictor")
class TextClassifierPredictor(Predictor):
	def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
		super().__init__(model, dataset_reader)

	def predict(self, instances: Instance):
		return self.predict_batch_instance(instances)