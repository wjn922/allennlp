from allennlp.training.metrics import BLEU
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from allennlp.training.metrics import CategoricalAccuracy, F1Measure

@Model.register("text_classifier")
class TextClassifier(Model):
    """
    This ``TextClassifier`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.
    The ``ComposedSeq2Seq`` class composes separate ``Seq2SeqEncoder`` and ``SeqDecoder`` classes.
    These parts are customizable and are independent from each other.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_text_embedders : ``TextFieldEmbedder``, required
        Embedders for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    decoder : ``SeqDecoder``, required
        The decoder of the "encoder/decoder" model
    tied_source_embedder_key : ``str``, optional (default=``None``)
        If specified, this key is used to obtain token_embedder in `source_text_embedder` and
        the weights are shared/tied with the decoder's target embedding weights.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    positive_label: the label you are interested in for prec, rec, f1.
        algebra, arithmetic, comparison, ...
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_text_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 tied_source_embedder_key: Optional[str] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 positive_label: str = "algebra",  
                 target_namespace: str = "tokens")-> None:

        super(TextClassifier, self).__init__(vocab, regularizer)

        self._source_text_embedder = source_text_embedder
        self._target_namespace = target_namespace
        self._encoder = encoder
        self._linear = torch.nn.Linear(in_features=encoder.get_output_dim(), 
                                        out_features=vocab.get_vocab_size('labels'))
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        
        self.accuracy = CategoricalAccuracy()
        positive_label = vocab.get_token_index(positive_label, namespace='labels')
        # for comnputing precision, recall and f1
        self.f1_measure = F1Measure(positive_label)

        # the loss function combines logsoftmax and NLLloss, the input to this function is logits
        self.loss_function = torch.nn.CrossEntropyLoss()  

        
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass on the encoder and decoder for producing the entire target sequence.
        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.
        Returns
        -------
        Dict[str, torch.Tensor]
            The output tensors from the decoder.
        """
        embedded_input = self._source_text_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size,  encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        # shape: (batch_size, vocab_size)
        logits = self._linear(encoder_outputs)

        output_dict = {"logits": logits}

        if labels is not None:
            self.accuracy(logits, labels)
            self.f1_measure(logits, labels)
            output_dict["loss"] = self.loss_function(logits, labels)

        if not self.training:   
            """
            during test,
            do the argmax over the probabilities of labels, and convert the index to label string
            """
            probs = F.softmax(output_dict["logits"], dim=-1) # (bs, vocab_size(label))
            predictions = probs.argmax(dim=-1) # (bs, )
            num_instances, _ = probs.size()

            predict_labels = []
            for i in range(num_instances):
                label_string = self.vocab.get_token_from_index(predictions[i].item(), namespace='labels')
                predict_labels.append(label_string)

            output_dict["probs"] = probs
            output_dict["predictions"] = predictions  # torch.Tensor: index
            output_dict["predict_labels"] = predict_labels    # string

        return output_dict

    """
    def predict(self, source_tokens: Dict[str, torch.LongTensor])-> Dict[str, torch.Tensor]:
        
        do the argmax over the probabilities of labels, and conver the index to label string
        
        output_dict = self.forward(source_tokens)
        probs = F.softmax(output_dict["logits"], dim=-1) # (bs, vocab_size(label))
        predictions = probs.argmax(dim=-1) # (bs, )
        num_instances, _ = prob.size()

        predict_labels = []
        for i in range(num_instances):
            label_string = self.vocab.get_token_from_index(predictions[i].item(), namespace='labels')
            predict_labels.append(label_string)

        output_dict["probs"] = probs
        output_dict["predictions"] = predictions  # torch.Tensor: index
        output_dict["predict_labels"] = predict_labels    # string

        return output_dict
    """

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        precision, recall, f1_score = self.f1_measure.get_metric(reset)
        all_metrics = {'accuracy': self.accuracy.get_metric(reset),
                       'precision': precision,
                       'recall': recall,
                       'f1_score': f1_score}

        return all_metrics





