from pathlib import Path

import pytest
import torch, os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace, Split, WhitespaceSplit, CharDelimiterSplit
from tokenizers.processors import TemplateProcessing

#from src.data.mnist_datamodule import MNISTDataModule
from src.data.threeModality_datamodule import ThreeModalitySingleDataset
from src.data.components.structure_utils.residue_constants import restypes, restype_order_with_x, restype_order_with_special_tokens, aa_ss8_with_special_tokens_map

# @pytest.mark.parametrize("batch_size", [32, 128])
# def test_mnist_datamodule(batch_size: int) -> None:
#     """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
#     attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
#     correctly match.

#     :param batch_size: Batch size of the data to be loaded by the dataloader.
#     """
#     data_dir = "data/"

#     dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
#     dm.prepare_data()

#     assert not dm.data_train and not dm.data_val and not dm.data_test
#     assert Path(data_dir, "MNIST").exists()
#     assert Path(data_dir, "MNIST", "raw").exists()

#     dm.setup()
#     assert dm.data_train and dm.data_val and dm.data_test
#     assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

#     num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
#     assert num_datapoints == 70_000

#     batch = next(iter(dm.train_dataloader()))
#     x, y = batch
#     assert len(x) == batch_size
#     assert len(y) == batch_size
#     assert x.dtype == torch.float32
#     assert y.dtype == torch.int64


def test_aa_span_mask():
    seq_len = input(">input the sequence length:")
    while int(seq_len) != 0:
        seq_str = np.random.choice(restypes,size=int(seq_len),replace=True).tolist()
        seq_str = ''.join(seq_str)

        tokenizer = Tokenizer(BPE(vocab=aa_ss8_with_special_tokens_map,merges=[],unk_token="[UNK]"))
        #tokenizer.add_special_tokens(['[CLS]','[SEP]','[STA]','[END]','[MASK]','[UNK]','[PAD]'])
        tokenizer.pre_tokenizer = Whitespace()
        #CharDelimiterSplit('+')
        tokenizer.post_processor = TemplateProcessing(single="[CLS] $0", pair="[CLS] $A [SEP] $B:1", special_tokens=[("[CLS]", 0), ("[SEP]", 1)])

        pretrain_tokenizer = PreTrainedTokenizerFast(model_max_length=1025, padding_side='right', bos_token="[STA]", eos_token="[END]", unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MKAA]", additional_special_tokens=["[MKSS]"],split_special_tokens=True, tokenizer_object=tokenizer)
        pretrain_tokenizer.mask_ss_token = "[MKSS]"
        pretrain_tokenizer.mask_ss_token_id = pretrain_tokenizer.convert_tokens_to_ids("[MKSS]")
        
        pretrain_tokenizer.convert_tokens_to_ids
        #pretrain_tokenizer.save_pretrained('tests/save_tokenizer')
        #pretrain_tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.abspath('tests/save_tokenizer'))

        aa_int_tokens = np.array(pretrain_tokenizer.encode(seq_str, add_special_tokens=False, truncation=False, padding=False), np.uint8)
        
        config = OmegaConf.create({'aa_max_mask_ratio': 0.2, 'span_context_prob': 0.2, 'permute_spans': True, 'max_span_len': 50, 'label_ignore': -1})
        test_dataset = ThreeModalitySingleDataset(config=config, tokenizer=pretrain_tokenizer)
        masked_inputs, labels, perm_idx, input_types, masked_pairs = test_dataset._apply_aa_span_mask(aa_int_tokens)
        
        masked_inputs = np.insert(masked_inputs, 0, pretrain_tokenizer.cls_token_id, axis=0)
        labels = np.insert(labels, 0, config.label_ignore, axis=0)
        perm_idx = np.insert(perm_idx + 1, 0, 0, axis=0)
        input_types = np.insert(input_types, 0, 0, axis=0)

        print(f"inputs: {aa_int_tokens}")
        print(f"masked_pairs: {masked_pairs}")
        mask_lens = [mp[-1]-mp[0] for mp in masked_pairs]
        print(f"masked_len: {mask_lens}, {sum(mask_lens)}/{len(aa_int_tokens)}, {sum(mask_lens)/len(aa_int_tokens) : .2f}")
        # print(f"masked_inputs: {masked_inputs}")
        # print(f"labels: {labels}")
        # print(f"aa_perm_idx: {perm_idx}")
        # print(f"input_types: {input_types}")
        print(f"masked_inputs, labels, aa_perm_idx, input_types")
        for mi, l, ai, it in zip(masked_inputs, labels, perm_idx, input_types):
            print(f"{mi : >5}{l : >5}{ai : >5}{it : >5}")

        seq_len = input(">input the sequence length:")

if __name__ == "__main__":
    print('test span mask')
    test_aa_span_mask()