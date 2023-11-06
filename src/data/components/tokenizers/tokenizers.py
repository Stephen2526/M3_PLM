"""
Tokenizer class
Modified from TAPE
"""

from typing import List
import logging
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)

IUPAC_CODES = OrderedDict([
    ('Ala', 'A'),
    ('Asx', 'B'),
    ('Cys', 'C'),
    ('Asp', 'D'),
    ('Glu', 'E'),
    ('Phe', 'F'),
    ('Gly', 'G'),
    ('His', 'H'),
    ('Ile', 'I'),
    ('Lys', 'K'),
    ('Leu', 'L'),
    ('Met', 'M'),
    ('Asn', 'N'),
    ('Pro', 'P'),
    ('Gln', 'Q'),
    ('Arg', 'R'),
    ('Ser', 'S'),
    ('Thr', 'T'),
    ('Sec', 'U'),
    ('Val', 'V'),
    ('Trp', 'W'),
    ('Xaa', 'X'),
    ('Tyr', 'Y'),
    ('Glx', 'Z')])

IUPAC_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29)])

UNIREP_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("M", 1),
    ("R", 2),
    ("H", 3),
    ("K", 4),
    ("D", 5),
    ("E", 6),
    ("S", 7),
    ("T", 8),
    ("N", 9),
    ("Q", 10),
    ("C", 11),
    ("U", 12),
    ("G", 13),
    ("P", 14),
    ("A", 15),
    ("V", 16),
    ("I", 17),
    ("F", 18),
    ("Y", 19),
    ("W", 20),
    ("L", 21),
    ("O", 22),
    ("X", 23),
    ("Z", 23),
    ("B", 23),
    ("J", 23),
    ("<cls>", 24),
    ("<sep>", 25)])

MINI_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("A", 4),
    ("C", 5),
    ("D", 6),
    ("E", 7),
    ("F", 8),
    ("G", 9),
    ("H", 10),
    ("I", 11),
    ("K", 12),
    ("L", 13),
    ("M", 14),
    ("N", 15),
    ("O", 16),
    ("P", 17),
    ("Q", 18),
    ("R", 19),
    ("S", 20),
    ("T", 21),
    ("U", 22),
    ("V", 23),
    ("W", 24),
    ("Y", 25),
    ("X", 28)])

# vocab dict of Tranception Model
TRANCEP_VOCAB = OrderedDict([
("[UNK]", 0),
("[CLS]", 1),
("[SEP]", 2),
("[PAD]", 3),
("[MASK]", 4),
("A", 5),
("C", 6),
("D", 7),
("E", 8),
("F", 9),
("G", 10),
("H", 11),
("I", 12),
("K", 13),
("L", 14),
("M", 15),
("N", 16),
("P", 17),
("Q", 18),
("R", 19),
("S", 20),
("T", 21),
("V", 22),
("W", 23),
("Y", 24)])

PFAM_VOCAB_20AA_IDX = [4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,23,24,25]
PFAM_VOCAB_20AA_IDX_MAP = {4:0,5:1,6:2,7:3,8:4,9:5,10:6,11:7,12:8,13:9,14:10,15:11,17:12,18:13,19:14,20:15,21:16,23:17,24:18,25:19,-1:-1,28:-1}

aaCodes = OrderedDict([
    ('Ala', 'A'),
    ('Asx', 'B'),
    ('Cys', 'C'),
    ('Asp', 'D'),
    ('Glu', 'E'),
    ('Phe', 'F'),
    ('Gly', 'G'),
    ('His', 'H'),
    ('Ile', 'I'),
    ('Lys', 'K'),
    ('Leu', 'L'),
    ('Met', 'M'),
    ('Asn', 'N'),
    ('Pro', 'P'),
    ('Gln', 'Q'),
    ('Arg', 'R'),
    ('Ser', 'S'),
    ('Thr', 'T'),
    ('Sec', 'U'),
    ('Val', 'V'),
    ('Trp', 'W'),
    ('Xaa', 'X'),
    ('Tyr', 'Y'),
    ('Glx', 'Z')])

# structure property classes
SS3_class = OrderedDict([('H',0),('E',1),('C',2),('-',-1)])
SS8_class = OrderedDict([('G',0),('I',1),('H',2),('B',3),('E',4),('T',5),('S',6),('-',-1)])
RSA2_class = OrderedDict([('B',0),('E',1),('-',-1)])


class BaseTokenizer():
    """Basic Tokenizer. Can use different vocabs depending on the model.
    """

    def __init__(self, vocab: str = 'pfam'):
        if vocab == 'iupac':
            self.vocab = IUPAC_VOCAB
        elif vocab == 'unirep':
            self.vocab = UNIREP_VOCAB
        elif vocab == 'pfam':
            self.vocab = PFAM_VOCAB
        else:
            raise Exception("vocab not known!")
        self.tokens = list(self.vocab.keys())
        self._vocab_type = vocab
        assert self.start_token in self.vocab and self.stop_token in self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        return "<cls>"

    @property
    def stop_token(self) -> str:
        return "<sep>"

    @property
    def mask_token(self) -> str:
        if "<mask>" in self.vocab:
            return "<mask>"
        else:
            raise RuntimeError(f"{self._vocab_type} vocab does not support masking")

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.tokens[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.convert_id_to_token(id_) for id_ in indices]

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return ''.join(tokens)

    def add_special_tokens(self, token_ids: List[str]) -> List[str]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        cls_token = [self.start_token]
        sep_token = [self.stop_token]
        return cls_token + token_ids + sep_token

    def add_placeholder_tokens(self, token_ids: List[str], placeholder: str):
        """In order to be consistent with amino acid seq tokenization,
        add placeholder token '-' at start and end position for structure element seq
        """
        return [placeholder] + token_ids + [placeholder]

    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenize(text)
        tokens = self.add_special_tokens(tokens)
        token_ids = self.convert_tokens_to_ids(tokens)
        return np.array(token_ids, np.int64)
    
    def get_normal_token_ids(self) -> List[int]:
        ids2return = []
        for tok, idx in self.vocab.items():
            if '<' not in tok:
                ids2return.append(idx)
        return ids2return

    def get_normal_token_ids(self) -> List[int]:
        ids2return = []
        for tok, idx in self.vocab.items():
            if '<' not in tok:
                ids2return.append(idx)
        return ids2return

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls()


    def struct_convert_tokens_to_ids(self, tokens: List[str], type: str) -> List[int]:
        if type == 'ss3':
            token_dict = SS3_class
        elif type == 'ss8':
            token_dict = SS8_class
        elif type == 'rsa2':
            token_dict = RSA2_class
        else:
            raise Exception(f"invalid input type {type}")
        return [token_dict[token] for token in tokens]

