from typing import Any, Dict, Optional, Tuple, Union, List
import torch, os, logging, random, copy, tree, math
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
import pandas as pd
import functools as fn
from itertools import chain
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


from src.data.components.tokenizers.tokenizers import BaseTokenizer
from src.data.components.openfold.utils import rigid_utils
from src.data.components.openfold.data import (
    data_transforms
)
import src.data.components.structure_utils.utils as du
from src.data.components.structure_utils import residue_constants

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
        |__<PDB_ID_2>
    
    Sample a batch of size [b,]
        1 PDB structure |-> b/2 seqs from bfd_uniclust_hits.a3m
                        |-> (b/2-1) PDB70 templates
        in this way ==> b seqs & b/2 structs
    
    Args:
        data_dir:
            A path to a directory containing mmCIF files (in train
            mode) or FASTA files (in inference mode).
        config:
            A dataset config object (defined by Hydra)
        mode:
            "fit", "val", "test", or "predict"
    """

    def __init__(self,
                metadata: pd.DataFrame = None,
                config: DictConfig = None,
                mode: str = "fit",
                struct_diffuser: Any = None,
                tokenizer: PreTrainedTokenizerFast = None,
                 **kwargs):
        super(ThreeModalitySingleDataset, self).__init__()
        self._log = logging.getLogger(__name__)
        
        self.metadata = metadata
        self.struct_diffuser = struct_diffuser        
        self.config = config
        self.mode = mode
        self.tokenizer = tokenizer
        
    @property
    def current_mode(self):
        return self.mode

    @property
    def get_struct_diffuser(self):
        return self.struct_diffuser

    @property
    def data_conf(self):
        return self.config

    def __len__(self) -> int:
        return len(self.metadata)
    
    @fn.lru_cache(maxsize=10000)
    def _process_data_row(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)

        # Run through OpenFold data transforms.
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype'][min_idx:(max_idx+1)]).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions'][min_idx:(max_idx+1)]).double(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask'][min_idx:(max_idx+1)]).double(),
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        # To speed up processing, only take necessary features
        final_feats = {
            'seqres': processed_feats['seqres'], # str
            'modeled_min': min_idx,
            'modeled_max': max_idx,
            #'aatype_arr': processed_feats['aatype'],
            #'seq_idx_arr': np.arange(len(processed_feats['seqres'])), # mmcif seq idx: min_idx to max_idx
            'sse8_type_ids_arr': processed_feats['sse8_type_ids'],
            'depth_resi_arr': processed_feats['depth_resi'],
            'b_factors_arr': processed_feats['b_factors'],
            'atom_pos_arr': processed_feats['atom_positions'],
            'sse_contacts': processed_feats['sse_contacts'],
            'homoSeq_aatype': processed_feats['homoSeq_aatype'],
            'pdb_temp': processed_feats['pdb_temp'],
            'atom_mask_arr': processed_feats['atom_mask'],
            'res_mask': processed_feats['bb_mask'],
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
            'atom37_pos': chain_feats['all_atom_positions'],
            'atom37_mask': chain_feats['all_atom_mask'],
            'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
            #'residue_index': processed_feats['residue_index'],
        }
        return final_feats
    
    def renumber_chain_resi_indices(self,chain_idx,res_idx):
        """"Re-number residue indices for each chain such that it starts from 1.
        Randomize chain indices.
        
        chain_idx: 1 1 1 1 1 3 3 3 6 6 6 -> 3 3 3 3 3 6 6 6 1 1 1
        resi_idx:  0 1 2 3 4 6 7 8 9 10 11 -> 1 2 3 4 5 1 2 3 1 2 3
        """
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = np.array(
            random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        for i,chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask
        return new_chain_idx, new_res_idx

    def __getitem__(self, index):
        data_row = self.metadata.iloc[index]
        if 'pdb_auth_chain' in data_row:
            pdb_auth_chain = data_row['pdb_auth_chain']
        else:
            raise ValueError('Need pdb-chain identifier.')

        processed_file_path = data_row,getattr('pkl_file_path')
        if processed_file_path is None:
            self._log.warning(f"No processed feature file for {pdb_auth_chain}")
        chain_feats = self._process_data_row(processed_file_path)
        
        out_feats = {}

        # homologs
        chain_feats['homoSeq_aatype'] # list[list[int]]
        chain_feats['pdb_temp'] # list[str]

        # modeled indices
        modeled_min = chain_feats['modeled_min']
        modeled_max = chain_feats['modeled_max']

        ### sequence processing ###
        #aa_ids = chain_feats['aatype_arr']
        #aa_id2str = lambda x: residue_constants.restypes[x] if x < residue_constants.restype_num else 'X'
        #aa_str = ''.join([aa_id2str(si) for si in aa_ids]).strip('X')
        
        aa_int_tokens = np.array(self.tokenizer.encode(chain_feats['seqres'][modeled_min:(modeled_max+1)], add_special_tokens=False, truncation=False, padding=False), np.uint8)
        aa_masked_ids, aa_labels, aa_permu_idx, aa_types, _ = self._apply_aa_span_mask(aa_int_tokens)
        # add the [CLS].id at beginning
        aa_masked_ids = np.insert(aa_masked_ids, 0, self.tokenizer.cls_token_id, axis=0)
        aa_unmasked_ids = np.insert(aa_int_tokens, 0, self.tokenizer.cls_token_id, axis=0)
        aa_labels = np.insert(aa_labels, 0, self.config.label_ignore, axis=0)
        aa_permu_idx = np.insert(aa_permu_idx + 1, 0, 0, axis=0)
        aa_types = np.insert(aa_types, 0, 0, axis=0)

        out_feats['aa_input_ids'] = aa_masked_ids
        out_feats['aa_label_ids'] = aa_labels
        out_feats['aa_permu_idx'] = aa_permu_idx
        out_feats['aa_type_ids'] = aa_types
        
        ### Spatial topology ###
        ss_masked_ids, label_ss_types, label_dist, label_depth, label_bfactor = self._apply_topo_mask(chain_feats)
        

        # Backbone diffusion
        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(index)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats['rigidgroups_0'])[:, 0]
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())

        # Sample t and diffuse.
        if self.is_training:
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
        else:
            t = 1.0
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )
        chain_feats.update(diff_feats_t)
        chain_feats['t'] = t
        
        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)
        final_feats = du.pad_feats(final_feats, data_row['modeled_seq_len']) #TODO: change padding
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        
        return
    
    def _get_chain_feat():

        return

    def _apply_aa_span_mask(self, inputs: np.ndarray):
        """Span masking for autoregressive blank infilling
        
        Adapted from huggingface.transformers.data.data_collator#L1234

        The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

            0. Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
            1. Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
            2. Reserve a context of length `context_length = span_length / plm_probability` to surround span to be
               masked
            3. Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length -
               span_length]` and mask tokens `start_index:start_index + span_length`
            4. Set `cur_len = cur_len + context_length`. If `cur_len < max_len` (i.e. there are tokens remaining in the
               sequence to be processed), repeat from Step 1.
            
        Args:
            inputs: integer id array of amino acid seq, [num_resi,]
        
        Returns:
            masked_inputs: amino acid seq ids with masks added as inputs
            labels: true amino acid ids for masked positions (-1 for unmasked postions)
            permu_idx: positional indices with permutation added (for ALiBi)
            input_types: segment types (context-0, spans-1,2,3..)
            
        """
        labels = np.copy(inputs)
        pos_idx = np.arange(labels.shape[0])
        seq_len = labels.shape[0]
        # Creating the mask and target_mapping tensors
        masked_indices = np.full(labels.shape[0], 0, dtype=np.bool_)
        masked_pairs = [[0, 0]]
        max_mask_pos = int(seq_len * self.config.aa_max_mask_ratio)

        # Start from the beginning of the sequence jump [CLS] by setting `cur_len = 1` (number of tokens processed so far).
        cur_len = 0
        max_len = labels.shape[0]

        while (cur_len < max_len) and (np.sum(masked_indices) < max_mask_pos):
            # Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
            span_length = np.random.randint(1, min(self.config.max_span_len, max_mask_pos) + 1, (1,)).item()
            # check number of masked positions
            cur_mask_num = np.sum(masked_indices).item()
            if (cur_mask_num + span_length) > max_mask_pos:
                if cur_mask_num > int(max_mask_pos*0.95):
                    break
                else:
                    span_length = np.random.randint(1, int(max_mask_pos*0.1) + 2, (1,)).item()

            # Reserve a context of length `context_length = span_length / span_context_prob` to surround the span to be masked
            context_length = int(math.ceil(span_length / self.config.span_context_prob))
            # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
            start_index = cur_len + np.random.randint(0, context_length - span_length + 1, (1,)).item()
            masked_indices[start_index : start_index + span_length] = 1
            masked_pairs.append([start_index, start_index + span_length])
            # Set `cur_len = cur_len + context_length`
            cur_len += context_length
        masked_pairs.append([max_len, max_len])

        special_tokens_mask = np.array(self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True),dtype=bool)
        masked_indices[special_tokens_mask] = 0
        if self.tokenizer._pad_token is not None:
            padding_mask = np.equal(labels, self.tokenizer.pad_token_id)
            masked_indices[padding_mask] = 0

        # Mask indicating non-functional tokens, where functional tokens are [SEP], [CLS], padding, etc.
        non_func_mask = ~(padding_mask | special_tokens_mask)

        # assemble masked_inputs and labels
        input_contexts = []
        input_spans = []
        label_spans = []
        input_pos_contexts = []
        input_pos_spans = []
        input_type_spans = []
        for mask_p in range(1,len(masked_pairs)):
            pre_pair = masked_pairs[mask_p-1]
            cur_pair = masked_pairs[mask_p]
            # 'context_ids' + [M].id
            input_contexts.extend(inputs[pre_pair[-1]:cur_pair[0]].tolist() + [self.tokenizer.mask_token_id])
            # 'context_pos' + span_first_pos
            input_pos_contexts.extend(pos_idx[pre_pair[-1]:cur_pair[0]].tolist() + [pos_idx[min(cur_pair[0],max_len-1)].item()])
            # skip last mask pair which is [seq_L, seq_L]
            if mask_p == len(masked_pairs)-1:
                continue
            # [STA].id + 'span_ids'
            input_spans.append([self.tokenizer.bos_token_id] + inputs[cur_pair[0]:cur_pair[-1]].tolist())
            # span_first_pos + 'span_ids'
            input_pos_spans.append([pos_idx[cur_pair[0]].item()] + pos_idx[cur_pair[0]:cur_pair[-1]].tolist())
            # 'span_ids' + [END].id
            label_spans.append(inputs[cur_pair[0]:cur_pair[-1]].tolist() + [self.tokenizer.eos_token_id])
            # 'span_type_ids'
            input_type_spans.append([mask_p]*(cur_pair[-1]-cur_pair[0]+1))
        # remove redundant [M] at the end of contexts
        del input_contexts[-1]
        del input_pos_contexts[-1]
        
        # permute span segments
        if self.config.permute_spans:
            span_perm_idx = sorted(list(range(len(input_spans))), key=lambda x: random.random())
            input_spans = [input_spans[i] for i in span_perm_idx]
            input_pos_spans = [input_pos_spans[i] for i in span_perm_idx]
            label_spans = [label_spans[i] for i in span_perm_idx]
            type_spans = [input_type_spans[i] for i in span_perm_idx]
        # flat lists
        flat_input_spans = list(chain(*input_spans))
        flat_input_pos_spans = list(chain(*input_pos_spans))
        flat_label_spans = list(chain(*label_spans))
        flat_input_type_spans = list(chain(*type_spans))

        masked_inputs = np.array(input_contexts + flat_input_spans, dtype=np.uint8)
        labels = np.array([self.config.label_ignore]*len(input_contexts) + flat_label_spans, dtype=np.int8)
        perm_idx = np.array(input_pos_contexts + flat_input_pos_spans, dtype=np.uint16)
        input_types = np.array([0]*len(input_contexts)+flat_input_type_spans, dtype=np.uint16)

        return masked_inputs, labels, perm_idx, input_types, masked_pairs

    def _apply_aa_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
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

    def _apply_topo_mask(self,
                         chain_feats: dict):
        """apply SSE masking and prepare prediction labels for topology learning
        SS8 seq tokenization and masking. The masking entities follow PTGL's node assignments
        Prediction labels include SS8 types for masked positions, pairwise Cb distance bins among helices and sheets, residue depth for helices and sheets, Cb b_factors for coils)

        Args:
            chain_feats:
                dictionary of chain features
        
        Returns:
            ss8_masked_ids: [num_resi,]
            label_ss8_ids: [num_resi,]
            label_dist: [num_resi, num_resi]
            label_depth: [num_resi,]
            label_bfactor: [num_resi,]
        """
        seqRes_str = chain_feats['seqres']
        sse_contacts = chain_feats['sse_contacts'] # dict(dict), utils.trim_PTGL_gml
        ss8_ids_arr = chain_feats['sse8_type_ids_arr'] # [num_res,]

        # get a sse type masking array (0-coil, 1-helix, 2-sheet)
        type_mask_map = {'H':1, 'E':2}
        sse_type_mask = np.zeros(ss8_ids_arr.shape[0], dtype=np.uint8)
        sse_node_ids_clc = {1: [], 2: [], 0: []}
        sse_node_len_clc = {1: [], 2: [], 0: []}
        for sse_node_id, sse_node_value in sse_contacts['sse_nodes'].items():
            node_res_pos = sse_node_value['seqRes_ids']
            node_type_id = type_mask_map.get(sse_node_value['sse_type'],0)
            sse_node_ids_clc[node_type_id].append(sse_node_id)
            sse_node_len_clc[node_type_id].append(len(node_res_pos))
            sse_type_mask[node_res_pos] = node_type_id

        # select SSEs for masking
        # probability from rescaled SSE length with ln(x)
        helix_mask_size = max(1, int(len(sse_node_ids_clc[1])*self.config.sse_mask_ratio))
        sheet_mask_size = max(1, int(len(sse_node_ids_clc[2])*self.config.sse_mask_ratio))
        helix_rescale_len = np.log(sse_node_len_clc[1])
        sheet_rescale_len = np.log(sse_node_len_clc[2])
        helix_selected_ids = np.random.choice(sse_node_ids_clc[1], size=helix_mask_size, replace=False, p=helix_rescale_len/np.sum(helix_rescale_len))
        sheet_selected_ids = np.random.choice(sse_node_ids_clc[2], size=sheet_mask_size, replace=False, p=sheet_rescale_len/np.sum(sheet_rescale_len))
    
        ss8_masked_ids = np.array([self.tokenizer.convert_tokens_to_ids(residue_constants.SS8_id2char[ssi].lower()) for ssi in ss8_ids_arr], dtype=np.uint8)
        label_ss8_ids = np.ones(ss8_ids_arr.shape[0], dtype=np.int8) * self.config.label_ignore
        for ss_m in np.concatenate((helix_selected_ids,sheet_selected_ids)):
            pos2mask = sse_contacts['sse_nodes'][ss_m]['seqRes_ids']
            ss8_masked_ids[pos2mask] = self.tokenizer.mask_ss_token_id
            for pos_i in pos2mask:
                label_ss8_ids[pos_i] = self.tokenizer.convert_tokens_to_ids(residue_constants.SS8_id2char[ss8_ids_arr[pos_i]].lower())
        
        # pairwise-distance between SSEs (helix and sheet)
        atom_pos_arr = chain_feats['atom_pos_arr'] # [num_res,37,3]
        atom_mask_arr = chain_feats['atom_mask_arr'] # [num_res,37]
        Cb_atom_idx = self._get_Cb_atom_idx(seqRes_str)
        resi_distMap = self._get_distanceMap(
                                atom_pos_arr=atom_pos_arr,
                                atom_mask_arr=atom_mask_arr,
                                atom_idx=Cb_atom_idx,
                                sse_type_mask=sse_type_mask,
                                coil_exclude=True)
        label_dist = self._discretize_any(
                input_arr=resi_distMap,
                first_cutoff=self.config.distogram.first_cutoff,
                last_cutoff=self.config.distogram.last_cutoff,
                num_bins=self.config.distogram.num_bins,
                ignore_index=self.config.label_ignore)

        # residue depth (discretize into bins)
        label_depth = np.copy(chain_feats['depth_resi_arr']) # [num_res,]
        label_depth = self._discretize_any(
                input_arr=label_depth,
                first_cutoff=self.config.depth.first_cutoff,
                last_cutoff=self.config.depth.last_cutoff,
                num_bins=self.config.depth.num_bins,
                ignore_index=self.config.label_ignore)
        coil_indx = np.where(sse_type_mask==0)[0]
        label_depth[coil_indx] = self.config.label_ignore

        # coil Ca b-factors (discretize into bins)
        cb_b_factor_arr = self._get_Cb_b_factor(chain_feats['b_factors_arr'],Cb_atom_idx) # [num_res, num_atom_type(37)]
        label_bfactor = self._discretize_any(
                input_arr=cb_b_factor_arr,
                first_cutoff=self.config.Bfactor.first_cutoff,
                last_cutoff=self.config.Bfactor.last_cutoff,
                num_bins=self.config.Bfactor.num_bins,
                ignore_index=self.config.label_ignore)
        non_coil_indx = np.where(sse_type_mask!=0)[0]
        label_bfactor[non_coil_indx] = self.config.label_ignore

        return ss8_masked_ids, label_ss8_ids, label_dist, label_depth, label_bfactor
    
    def _get_Cb_atom_idx(
            self,
            seqRes_str: str) -> np.ndarray:
        """Acquire Cb atom indices (Ca for GLY)
        """
        cb_idx = residue_constants.atom_order['CB']
        ca_idx = residue_constants.atom_order['CA']
        num_res = len(seqRes_str)
        # GLY position
        gly_pos = [p for p, aa in enumerate(seqRes_str) if aa=='G']
        atom_idx = np.array([cb_idx]*num_res,dtype=np.uint8)
        atom_idx[gly_pos] = ca_idx

        return atom_idx

    def _discretize_any(
            self,
            input_arr: np.ndarray,
            first_cutoff: float,
            last_cutoff: float,
            num_bins: int,
            ignore_index: int=-1,
            dtype: np.dtype = np.uint8) -> np.ndarray:
        """discretize residue depth value into bins. 
        """
        nan_indices = np.nonzero(np.isnan(input_arr))
        bin_cutoffs = np.linspace(first_cutoff,last_cutoff,num=num_bins-1)
        assign_bin = np.sum([input_arr > cutof for cutof in bin_cutoffs],axis=0,dtype=dtype)
        assign_bin[nan_indices] = ignore_index
        return assign_bin

    def _get_distanceMap(self,
                         atom_pos_arr: np.ndarray,
                         atom_mask_arr: np.ndarray,
                         atom_idx: np.ndarray,
                         sse_type_mask: np.ndarray,
                         coil_exclude: bool,) -> np.ndarray:
        """Generate C_beta distance map
        """
        num_res = atom_pos_arr.shape[0]
        coil_idx = np.where(sse_type_mask == 0)[0]
        res_idx = np.arange(num_res,dtype=np.uint16)
        cb_atom_pos_arr = atom_pos_arr[res_idx,atom_idx] #[num_res, 3]
        cb_atom_mask_arr = atom_mask_arr[res_idx,atom_idx] #[num_res,]
        # replace non-existing atom pos with nan
        res_mask_idx = np.nonzero(cb_atom_mask_arr == 0)[0]
        cb_atom_pos_arr[res_mask_idx] = np.nan
        if coil_exclude:
            cb_atom_pos_arr[coil_idx] = np.nan
        cb_dist_matrix = cdist(cb_atom_pos_arr, cb_atom_pos_arr, 'euclidean')
        return cb_dist_matrix

    def _discretize_distogram_fast(
            self,
            distance_map: np.ndarray,
            first_cutoff: float,
            last_cutoff: float,
            num_bins: int,
            ignore_index: int=-1,
            dtype: np.dtype = np.uint8) -> np.ndarray:
        """discretize distance value into bins. 
        """
        nan_indices = np.nonzero(np.isnan(distance_map))
        bin_cutoffs = np.linspace(first_cutoff,last_cutoff,num=num_bins-1)
        assign_bin = np.sum([distance_map > cutof for cutof in bin_cutoffs],axis=0,dtype=dtype)
        assign_bin[nan_indices] = ignore_index
        return assign_bin

    def _nor_b_factor(self, b_factors_arr: np.ndarray):
        """Normalized atom-level b-factor with respect to all atoms in the protein
            (b_i - b_u) / b_\sigma
        and extract Ca atom b-facotr

        Args:
            b_factors_arr: [num_res, num_atom_type(37)]
        
        Returns:
            normalized b-factor array, [num_res, num_atom_type(37)]
            Ca atom b-factor array, [num_res,]

            np.nan for non-existing atoms
        """
        b_factors_arr[b_factors_arr == 0.] = np.nan
        b_mean = np.nanmean(b_factors_arr)
        b_sd = np.nanstd(b_factors_arr)
        nor_b_factors_arr = (b_factors_arr - b_mean) / b_sd
        ca_idx = residue_constants.atom_order['CA']
        ca_b_factors_arr = nor_b_factors_arr[:, ca_idx]
        return nor_b_factors_arr, ca_b_factors_arr

    def _get_Cb_b_factor(
            self,
            b_factor_arr: np.ndarray,
            atom_idx: np.ndarray):
        res_idx = np.arange(b_factor_arr.shape[0],dtype=np.uint16)
        Cb_b_factor_arr = b_factor_arr[res_idx, atom_idx]
        return Cb_b_factor_arr

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
        config: DictConfig = None,
        mode: str = 'fit',
    ) -> None:
        """Initialize a `ThreeModalityDataModule`.

        Args:
            config: configurations for data module.
            mode: current running mode, one of "fit", "val", "test" and "predict".
        """
        super().__init__()

        # this line allows to access init params with 'self.config' attribute
        # also ensures init params will be stored in ckpt
        #self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.config = config
        self.mode = mode
        
        if config.load_tokenizer_path:
            self.tokenizer = self._load_tokenizer(config.load_tokenizer_path)
        else:
            self.tokenizer = self._init_tokenizer(config.load_tokenizer_path)

        self.supported_exts = [".cif"]
        valid_modes = ["fit", "val", "test", "predict"]

        if(mode not in valid_modes):
            raise ValueError(f'mode must be one of {valid_modes}')
        
        # init metadata and split for train, val and test
        if self.mode not in ['predict']:
            self._init_metadata(self.config.split_strategy)
            if self.mode == 'fit':
                self.target_metadata = self.train_metadata
            elif self.mode == 'val':
                self.target_metadata = self.val_metadata
            elif self.mode == 'test':
                self.target_metadata = self.test_metadata

    def _load_tokenizer(self, load_tokenizer_path: str=None):
        pretrain_tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.abspath(load_tokenizer_path))
        return pretrain_tokenizer

    def _init_tokenizer(self, save_tokenizer_path: str=None):
        tokenizer = Tokenizer(BPE(vocab=residue_constants.aa_ss8_with_special_tokens_map,merges=[],unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(single="[CLS] $0", pair="[CLS] $A [SEP] $B:1", special_tokens=[("[CLS]", 0), ("[SEP]", 1)])
        pretrain_tokenizer = PreTrainedTokenizerFast(model_max_length=self.config.filtering.max_len + 1, padding_side='right', bos_token="[STA]", eos_token="[END]", unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MKAA]", additional_special_tokens=["[MKSS]"],split_special_tokens=True, tokenizer_object=tokenizer)
        pretrain_tokenizer.mask_ss_token = "[MKSS]"
        pretrain_tokenizer.mask_ss_token_id = pretrain_tokenizer.convert_tokens_to_ids("[MKSS]")
        if save_tokenizer_path:
            pretrain_tokenizer.save_pretrained(save_tokenizer_path)
        return pretrain_tokenizer

    def _init_metadata(self, split_strategy: str='default'):
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
        # split metadata
        if split_strategy == 'default':
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
        
        # Held-out test set
        pdb_after_OPS = pdb_chain_metadata_not_OPS[pdb_chain_metadata_not_OPS.release_date > '2021-12-31']
        self.test_metadata = pdb_after_OPS.sample(len(pdb_after_OPS)//5, replace=False, random_state=self.config.seed).sort_values('modeled_seq_len', ascending=False)
        self.test_pdb_chain_ids = self.test_metadata.pdb_auth_chain.to_list()
        self._log.info(
            f'TEst data size (pdb_chains): {len(self.test_metadata)}')

        # Validation set
        self.val_metadata = pdb_chain_metadata_not_OPS[~pdb_chain_metadata_not_OPS.pdb_auth_chain.isin(self.test_pdb_chain_ids)].sample(len(pdb_chain_metadata_not_OPS)//10, replace=False, random_state=self.config.seed).sort_values('modeled_seq_len', ascending=False)
        self.val_pdb_chain_ids = self.val_metadata.pdb_auth_chain.to_list()
        self._log.info(
            f'VALidation data size (pdb_chains): {len(self.val_metadata)}')

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

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        return


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
            if self.config.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.config.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.config.batch_size // self.trainer.world_size



    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
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
