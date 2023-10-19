from typing import Any, Dict, Optional, Tuple, Union, List

import torch, os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from components.tokenizers import BaseTokenizer
from components.openfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    templates,
)

class ThreeModalitySingleDataset(Dataset):
    """Creates the dataset for three modalities
    
    OpenFold data organization at PDB level (https://docs.google.com/document/d/1R90-VJSLQEbot7tgXF3zb068Y1ZJAmsckQ_t2sJTv2c/edit?pli=1):
    |__pdb # MSAs/template hits for every sequence in PDB (2021-12)
        |__<PDB_ID_1>
            |__a3m
                |__bfd_uniclust_hits.a3m # BFD/Uniclust30 MSA by HHblits
                |__mgnify_hits.a3m # Mgnify MSA by JackHMMER
                |__uniref90_hits.a3m # UniRef90 MSA by JackHMMER
            |__hhr
                |__pdb70_hits.hhr # PDB70 template hits by HHSearch
        |__ . . .
    
    Sample a batch of size [b,]
        1 PDB structure |-> b/2 seqs from bfd_uniclust_hits.a3m
                        |-> (b/2-1) PDB70 templates
        in this way ==> b seqs & b/2 structs
    
    Args:
        data_dir:
            A path to a directory containing mmCIF files (in train
            mode) or FASTA files (in inference mode).
        alignment_dir:
            A path to a directory containing only data in the format 
            output by an AlignmentRunner 
            (defined in openfold.features.alignment_runner).
            I.e. a directory of directories named {PDB_ID}_{CHAIN_ID}
            or simply {PDB_ID}, each containing .a3m and .hhr
            files.
        config:
            A dataset config object (defined by Hydra)
        alignment_info (dataframe):
            meta info of alignment data with keys:
            - chain_nm: {PDB_ID}_{CHAIN_ID} or {PDB_ID}
            - seq_len: seq length
            - ext: structure file extension, e.g. ".cif", ".core", ".pdb"
            - ...

        mode:
            "fit", "val", "test", or "predict" 
    """

    def __init__(self,
                data_dir: str = None,
                config: DictConfig = None,
                mode: str = "train",
                tokenizer: Union[str, BaseTokenizer, PreTrainedTokenizerBase] = PreTrainedTokenizerFast,
                 **kwargs):
        super(ThreeModalitySingleDataset, self).__init__()
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        
        self.data_dir = data_dir
        self.config = config
        self.mode = mode
        self.tokenizer = tokenizer

        self.supported_exts = [".cif", ".core", ".pdb"]
        valid_modes = ["train", "eval", "predict"]
        if(mode not in valid_modes):
            raise ValueError(f'mode must be one of {valid_modes}')
    
        if(alignment_meta is not None):
            self._chain_nms = alignment_meta['chain_nm'].to_list()
        else:
            self._chain_nms = list(os.listdir(alignment_dir))

        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_nms)
        }
        

    def __len__(self) -> int:
        return len(self._chain_nms)

    def _init_metadata(self):
        """Initialize metadata."""
        # Process metadata CSV with different filtering criterions.
        filter_conf = self.config.filtering
        pdb_chain_metadata = pd.read_csv(self.config.metadata_path)
        self.raw_pdb_chain_metadata = pdb_chain_metadata
        
        if filter_conf.max_len is not None:
            pdb_chain_metadata = pdb_chain_metadata[pdb_chain_metadata.modeled_seq_len <= filter_conf.max_len]
        if filter_conf.min_len is not None:
            pdb_chain_metadata = pdb_chain_metadata[pdb_chain_metadata.modeled_seq_len >= filter_conf.min_len]
        if filter_conf.max_helix_percent is not None:
            pdb_chain_metadata = pdb_chain_metadata[
                pdb_chain_metadata.helix_percent < filter_conf.max_helix_percent]
        if filter_conf.max_loop_percent is not None:
            pdb_chain_metadata = pdb_chain_metadata[
                pdb_chain_metadata.coil_percent < filter_conf.max_loop_percent]
        if filter_conf.min_beta_percent is not None:
            pdb_chain_metadata = pdb_chain_metadata[
                pdb_chain_metadata.strand_percent > filter_conf.min_beta_percent]
        
        self._create_split_train_val_test(pdb_chain_metadata)
        

    def _create_split_train_val_test(self, pdb_chain_metadata):
        """ Split train, val and test sets of single chains
        train - OpenProteinSet
        val - random single chains out of OPS
        test - random single chains after 2021-12
        """
        # filter out pdb-chains in OpenProteinSet
        pdb_chain_metadata_OPS = pdb_chain_metadata[pdb_chain_metadata.auth_chain_in_OPS == True]
        pdb_chain_metadata_not_OPS = pdb_chain_metadata[pdb_chain_metadata.auth_chain_in_OPS == False]
        
        # Training set
        self.train_metadata = pdb_chain_metadata_OPS
        self._log.info(
            f'TRaining data size (OPS pdb-chains): {len(self.train_metadata)}')
        
        # Test set
        

        # Validation set
        self.eval_metadata = pdb_chain_metadata_not_OPS.sample(len(pdb_chain_metadata_not_OPS)//8,replace=False, random_state=self.config.seed).sort_values('modeled_seq_len', ascending=False)
        self.eval_pdb_chain_ids = self.eval_metadata.pdb_auth_chain.to_list()
        self._log.info(
            f'VALidation data size (pdb_chains): {len(self.eval_metadata)}')

    def _split_by_modeled_seqLen(self, metadata_df, num_len_bins, samples_per_len_bin):
        """subset split based on modeled seq len"""
        all_lengths = np.sort(metadata_df.modeled_seq_len.unique())
        length_indices = (len(all_lengths) - 1) * np.linspace(
            0.0, 1.0, num_len_bins)
        length_indices = length_indices.astype(int)
        eval_lengths = all_lengths[length_indices]
        sub_df = metadata_df[metadata_df.modeled_seq_len.isin(eval_lengths)]
        # Fix a random seed to get the same split each time.
        sub_df = sub_df.groupby('modeled_seq_len').sample(
            samples_per_len_bin, replace=True, random_state=self.config.seed)
        sub_df = sub_df.sort_values('modeled_seq_len', ascending=False)
        return sub_df
        

    def idx_to_chain_nm(self, idx):
        return self._chain_nms[idx]

    def __getitem__(self, index):
        chain_nm = self.idx_to_chain_nm(index)
        pdb_file_ext = self.alignment_meta.loc[self.alignment_meta['chain_nm'] == chain_nm,'ext'].iloc[0]
        if (pdb_file_ext not in self.supported_exts):
            raise ValueError("Invalid structure file type")
        alignment_tar_dir = os.path.join(self.alignment_dir, chain_nm)

        if (self.mode == 'train') or (self.mode == 'eval'):
            spl = chain_nm.rsplit('_', 1)
            if(len(spl) == 2):
                pdb_id, chain_id = spl
            else:
                pdb_id, = spl
                chain_id = None
            pdb_path = os.path.join(self.data_dir, pdb_id)
            pdb_path += pdb_file_ext

            if (pdb_file_ext == ".cif"):
                    data = self._parse_mmcif(
                        pdb_path, pdb_id, chain_id, alignment_tar_dir, self.alignment_meta,
                    )
            elif (pdb_file_ext == ".core"):
                    data = self.data_pipeline.process_core(
                        path, alignment_dir, alignment_index,
                    )
            elif (pdb_file_ext == ".pdb"):
                    structure_index = None
                    if(self._structure_index is not None):
                        structure_index = self._structure_index[name]
                    data = self.data_pipeline.process_pdb(
                        pdb_path=path,
                        alignment_dir=alignment_dir,
                        is_distillation=self.treat_pdb_as_distillation,
                        chain_id=chain_id,
                        alignment_index=alignment_index,
                        _structure_index=structure_index,
                    )
            else:
                raise ValueError("Extension branch missing") 
        else:
            path = os.path.join(name, name + ".fasta")
            data = self.data_pipeline.process_fasta(
                fasta_path=path,
                alignment_dir=alignment_dir,
                alignment_index=alignment_index,
            )
    

        return 

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        

        return 

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token(not special tokens)
                    token = self.tokenizer.convert_id_to_token(
                        random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels

    def _apply_bert_mask_copy(self, tokens: List[str], copy_num: int) -> Tuple[List[List], List[List]]:
        """
        return multiple copies of masked seqs for the single input seq
        """
        masked_tokens_aug = []
        labels_aug = []
        for cpy in range(copy_num):
            masked_tokens = copy(tokens)
            labels = np.zeros([len(tokens)], np.int64) - 1
            for i, token in enumerate(tokens):
                # Tokens begin and end with start_token and stop_token, ignore these
                if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                    continue
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        token = self.tokenizer.mask_token
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)

        return masked_tokens_aug, labels_aug


class ThreeModalityDataModule(LightningDataModule):
    """`LightningDataModule` for dataset of three modalities:
        - sequence
        - spatial topology
        - pdb structure
    
    TODO: the pipeine
    sequence:
        - span masking: refer to https://github.com/huggingface/transformers/blob/bffac926ca6bc6c965a92bfbfd00c567a2c0fb90/src/transformers/data/data_collator.py#L1234
        - attention_mask: refer to https://github.com/THUDM/GLM/blob/4f61ed7237a3b0187f4d62062429348276a78c84/tasks/seq2seq/dataset.py#L853
        - padding: may use https://huggingface.co/docs/transformers/v4.33.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `ThreeModalityDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)


    def setup(self, stage: Optional[str] = None) -> None:
        """Setup pytorch datasets for `self.data_train`, `self.data_val`, `self.data_test`, `self.data_predict`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ThreeModalityDataModule()
