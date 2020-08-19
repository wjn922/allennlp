from typing import Dict
import logging
from overrides import overrides
import glob
import os

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField, LabelField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("math_datasetreader")
class MathDatasetReader(DatasetReader):
    """
    ``MathDatasetReader`` class inherits Seq2SeqDatasetReader as the only
    difference is when dealing with autoencoding tasks i.e., the target equals the source.
    
    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 target: bool = True,
                 label: bool = False,
                 delimiter: str = "\t",
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._delimiter = delimiter

        self._target = target
        self._label = label

    @overrides
    def _read(self, file_path):
        files = glob.glob(file_path + "/*.txt")
        for file in files:
            with open(cached_path(file), "r") as data_file:
                logger.info("Reading instances from lines in file at: %s", file)
    
                for line_num, line in enumerate(data_file):
                    if line_num % 3 == 0:
                        line = line.strip("\n")
                        label = line

                    if line_num % 3 == 1:
                        line = line.strip("\n")

                        if self._label:
                            yield self.text_to_instance(line, label)
                        else:
                            yield self.text_to_instance(line, None)

        

    @overrides
    def text_to_instance(self, input_string: str, label_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source_string = self._source_tokenizer.tokenize(input_string)
        tokenized_source = tokenized_source_string.copy()
        tokenized_target_string = self._target_tokenizer.tokenize(input_string)
        tokenized_target = tokenized_target_string.copy()
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._target_token_indexers)

        if label_string is not None:  # there are 8 math topics
            label = label_string.split('_')[0]
            label_field = LabelField(label)

        if self._target:
            if self._label:
                fields = Instance({"source_tokens": source_field, "target_tokens": target_field, "labels": label_field})
            else:
                fields = Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            if self._label:
                fields = Instance({"source_tokens": source_field,  "labels": label_field})
            else:
                fields = Instance({"source_tokens": source_field})

        return fields