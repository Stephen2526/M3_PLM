import pickle, os, glob, dataclasses, string, io, gzip, collections
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Any
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from Bio import PDB
from Bio.PDB.Chain import Chain
from torch.utils import data
import torch

from src.data.components.structure_utils import chemical, residue_constants, protein, so3_utils
from src.data.components.openfold.utils import rigid_utils
from src.data.components.openfold.data import parsers

Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]
UNPADDED_FEATS = [
    't', 'rot_score_scaling', 'trans_score_scaling', 't_seq', 't_struct'
]
RIGID_FEATS = [
    'rigids_0', 'rigids_t'
]
PAIR_FEATS = [
    'rel_rots'
]
PLM_FEATS = [
    'input_aa_ids', 'input_aa_mask', 'target_aa_ids'
]
# features padded with -1
PAD_NEG_ONE_FEATS = [
    'target_aa_ids'
]

move_to_np = lambda x: x.cpu().detach().numpy()
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])

class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def write_pkl(
        save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)

def compare_conf(conf1, conf2):
    return OmegaConf.to_yaml(conf1) == OmegaConf.to_yaml(conf2)

def parse_pdb(filename):
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines)

def parse_pdb_lines(lines):

    # indices of residues observed in the structure
    idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    seq = []
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        seq.append(residue_constants.restype_3to1[aa])
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(chemical.aa2long[chemical.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return xyz, mask, np.array(idx_s), ''.join(seq)

def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        #chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
        chain_int += CHAIN_TO_INT[chain_char] * len(ALPHANUMERIC)**i
    return chain_int

def chain_int_to_str(chain_int: int):
    chain_str = ''
    char_num = len(ALPHANUMERIC)
    if chain_int < char_num:
        return INT_TO_CHAIN[chain_int]
    curr_num = chain_int
    while curr_num > 0:
        curr_mod = curr_num % char_num
        chain_str += INT_TO_CHAIN[curr_mod]
        curr_num = (curr_num - curr_mod) // char_num
    return chain_str

def parse_pdb_feats(
        pdb_name: str,
        pdb_path: str,
        scale_factor=1.,
        # TODO: Make the default behaviour read all chains.
        chain_id='A',
    ):
    """
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        mean_center: whether to mean center atom positions.
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {
        chain.id: chain
        for chain in structure.get_chains()}

    def _process_chain_id(x):
        chain_prot = process_chain(struct_chains[x], x)
        chain_dict = dataclasses.asdict(chain_prot)

        # Process features
        feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
        return parse_chain_feats(
            feat_dict, scale_factor=scale_factor)

    if isinstance(chain_id, str):
        return _process_chain_id(chain_id)
    elif isinstance(chain_id, list):
        return {
            x: _process_chain_id(x) for x in chain_id
        }
    elif chain_id is None:
        return {
            x: _process_chain_id(x) for x in struct_chains
        }
    else:
        raise ValueError(f'Unrecognized chain list {chain_id}')

def rigid_frames_from_atom_14(atom_14):
    n_atoms = atom_14[:, 0]
    ca_atoms = atom_14[:, 1]
    c_atoms = atom_14[:, 2]
    return rigid_utils.Rigid.from_3_points(
        n_atoms, ca_atoms, c_atoms
    )

def compose_rotvec(r1, r2):
    """Compose two rotation euler vectors."""
    R1 = rotvec_to_matrix(r1)
    R2 = rotvec_to_matrix(r2)
    cR = np.einsum('...ij,...jk->...ik', R1, R2)
    return matrix_to_rotvec(cR)

def rotvec_to_matrix(rotvec):
    return Rotation.from_rotvec(rotvec).as_matrix()

def matrix_to_rotvec(mat):
    return Rotation.from_matrix(mat).as_rotvec()

def rotvec_to_quat(rotvec):
    return Rotation.from_rotvec(rotvec).as_quat()

def pad_feats(raw_feats, max_len, use_torch=False):
    # padded_feats = {
    #     feat_name: pad(feat, max_len, use_torch=use_torch)
    #     for feat_name, feat in raw_feats.items()
    #     if feat_name not in UNPADDED_FEATS + RIGID_FEATS
    # }
    padded_feats = {}
    for feat_name, feat in raw_feats.items():
        if feat_name not in UNPADDED_FEATS + RIGID_FEATS:
            if feat_name not in PAD_NEG_ONE_FEATS:
                padded_feats[feat_name] = pad(feat, max_len, use_torch=use_torch)
            else:
                padded_feats[feat_name] = pad(feat, max_len, pad_value=-1, use_torch=use_torch)

    for feat_name in PAIR_FEATS:
        if feat_name in padded_feats:
            padded_feats[feat_name] = pad(padded_feats[feat_name], max_len, pad_idx=1)
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    for feat_name in RIGID_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = pad_rigid(raw_feats[feat_name], max_len)
    return padded_feats

def pad_rigid(rigid: torch.tensor, max_len: int):
    num_rigids = rigid.shape[0]
    pad_amt = max_len - num_rigids
    pad_rigid = rigid_utils.Rigid.identity(
        (pad_amt,), dtype=rigid.dtype, device=rigid.device, requires_grad=False)
    return torch.cat([rigid, pad_rigid.to_tensor_7()], dim=0)

def pad(x: np.ndarray, max_len: int, pad_idx=0, pad_value=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        pad_value: value used to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        #raise ValueError(f'Invalid pad amount {pad_amt}')
        return x
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename):

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    #print(filename)

    if filename.split('.')[-1] == 'gz':
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename, 'r')

    # read file line by line
    for line in fp:

        # skip labels
        if line[0] == '>':
            continue

        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)
        if len(msa) == 10000:
            break

    # convert letters into numbers
    alphabet = list("ARNDCQEGHILKMFPSTWYV-")
    encoding = np.array(alphabet, dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for letter, enc in zip(alphabet, encoding):
        res_cat = residue_constants.restype_order_with_x.get(
            letter, residue_constants.restype_num)
        msa[msa == enc] = res_cat

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins

def parse_a3m_rawSeq(file_path: str):
    """Parses raw(unaligned) sequences from a3m format alignment.

    Args:
        file_path: path to the a3m file. The first sequence in the
            file should be the query sequence.

    Returns:
        A tuple of:
            * A list of sequences that have been aligned to the query. These
                might contain duplicates.
            * The deletion matrix for the alignment as a list of lists. The element
                at `deletion_matrix[i][j]` is the number of residues deleted from
                the aligned sequence i at residue position j.
    """
    if file_path.split('.')[-1] == 'gz':
        with gzip.open(file_path, 'rt') as fp:
            a3m_string= fp.read()
    else:
        with open(file_path, 'r') as fp:
            a3m_string= fp.read()
    
    sequences, _ = parsers.parse_fasta(a3m_string)
    
    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans("", "", "-")
    raw_sequences = []
    max_seq_len = 0
    for s in sequences:
        raw_sequences.append(s.translate(deletion_table).upper())
        if len(raw_sequences[-1]) > max_seq_len:
            max_seq_len = len(raw_sequences[-1])

    # tokenize seqs    
    alphabet = list("ARNDCQEGHILKMFPSTWYVBZJUO")
    encoding = np.array(alphabet, dtype='|S1').view(np.uint8)
    rawSeq_pad_ids = np.array([list(s+'x'*(max_seq_len-len(s))) for s in raw_sequences], dtype='|S1').view(np.uint8)
    for letter, enc in zip(alphabet, encoding):
        if letter in 'BZJUO':
            letter = residue_constants.ambiguous_restype_1to1(letter)
        res_cat = residue_constants.restype_order.get(
                    letter, residue_constants.restype_num) # default value: 20
        rawSeq_pad_ids[rawSeq_pad_ids == enc] = res_cat
    
    # treat all unknown characters as gaps
    rawSeq_pad_ids[(rawSeq_pad_ids > 20) & (rawSeq_pad_ids != 120)] = 20 # uint8
    rawSeq_ids = [s[s != 120].tolist() for s in rawSeq_pad_ids]

    return rawSeq_ids[1:], raw_sequences[1:]

def write_checkpoint(
        ckpt_path: str,
        model,
        conf,
        optimizer,
        epoch,
        step,
        logger=None,
        use_torch=True,
    ):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    for fname in os.listdir(ckpt_dir):
        if '.pkl' in fname or '.pth' in fname:
            os.remove(os.path.join(ckpt_dir, fname))
    if logger is not None:
        logger.info(f'Serializing experiment state to {ckpt_path}')
    else:
        print(f'Serializing experiment state to {ckpt_path}')
    write_pkl(
        ckpt_path,
        {
            'model': model,
            'conf': conf,
            'optimizer': optimizer,
            'epoch': epoch,
            'step': step
        },
        use_torch=use_torch)

def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict

def length_batching(
        np_dicts: List[Dict[str, np.ndarray]],
        max_squared_res: int,
    ):
    get_len = lambda x: x['res_mask'].shape[0]
    dicts_by_length = [(get_len(x), x) for x in np_dicts]
    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0]
    ## adaptively reduce batch size based on sequence length
    max_batch_examples = int(max_squared_res // max_len**2)
    pad_example = lambda x: pad_feats(x, max_len)
    padded_batch = [
        pad_example(x) for (_, x) in length_sorted[:max_batch_examples]]
    return torch.utils.data.default_collate(padded_batch)

def create_data_loader(
        torch_dataset: data.Dataset,
        batch_size,
        shuffle,
        sampler=None,
        num_workers=0,
        np_collate=False,
        max_squared_res=1e6,
        length_batch=False,
        drop_last=False,
        prefetch_factor=2):
    """Creates a data loader with jax compatible data structures."""
    if np_collate:
        collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)
    elif length_batch:
        collate_fn = lambda x: length_batching(
            x, max_squared_res=max_squared_res)
    else:
        collate_fn = None
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return data.DataLoader(
        torch_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context='fork' if num_workers != 0 else None,
        )

def parse_chain_feats(chain_feats: protein.ProteinChain, scale_factor=1.) -> protein.ProteinChain:
    ca_idx = residue_constants.atom_order['CA']
    chain_feats.bb_mask = chain_feats.atom_mask[:, ca_idx]
    bb_pos = chain_feats.atom_positions[:, ca_idx]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats.bb_mask) + 1e-5)
    centered_pos = chain_feats.atom_positions - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats.atom_positions = scaled_pos * chain_feats.atom_mask[..., None]
    chain_feats.bb_positions = chain_feats.atom_positions[:, ca_idx]
    return chain_feats

def rigid_frames_from_all_atom(all_atom_pos):
    rigid_atom_pos = []
    for atom in ['N', 'CA', 'C']:
        atom_idx = residue_constants.atom_order[atom]
        atom_pos = all_atom_pos[..., atom_idx, :]
        rigid_atom_pos.append(atom_pos)
    return rigid_utils.Rigid.from_3_points(*rigid_atom_pos)

def pad_pdb_feats(raw_feats, max_len):
    padded_feats = {
        feat_name: pad(feat, max_len)
        for feat_name, feat in raw_feats.items() if feat_name not in UNPADDED_FEATS
    }
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    return padded_feats

def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]
                          ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors))

def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected

def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram

def quat_to_rotvec(quat, eps=1e-6):
    # w > 0 to ensure 0 <= angle <= pi
    flip = (quat[..., :1] < 0).float()
    quat = (-1 * quat) * flip + (1 - flip) * quat

    angle = 2 * torch.atan2(
        torch.linalg.norm(quat[..., 1:], dim=-1),
        quat[..., 0]
    )

    angle2 = angle * angle
    small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_angle_scales = angle / torch.sin(angle / 2 + eps)

    small_angles = (angle <= 1e-3).float()
    rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
    rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
    return rot_vec

def quat_to_rotmat(quat, eps=1e-6):
    rot_vec = quat_to_rotvec(quat, eps)
    return so3_utils.Exp(rot_vec)

def save_fasta(
        pred_seqs,
        seq_names,
        file_path,
    ):
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as f:
        for x,y in zip(seq_names, pred_seqs):
            f.write(f'>{x}\n{y}\n')


def run_PTGLtools(pdb_id: str, pdb_path: str, tool_obj_path: str, tmp_dir: str):
    """Run PTGLgraphComputation tools to generate SSE contacts and interaction types

    Args:
        pdb_id:
            4-letter identifier for pdb structure
        pdb_path:
            file path of pdb structure (default use mmcif format)
        tool_obj_path:
            PTGLgraphComputation java program file path
        tmp_path:
            dir path to save intermediate output files
    
    Returns:
        chain_gml: Dict[auth_chain_id, gml_object]
    """
    if not Path(pdb_path).is_absolute():
        pdb_path = os.path.abspath(pdb_path)
    if not Path(tool_obj_path).is_absolute():
        tool_obj_path = os.path.abspath(tool_obj_path)
    if not Path(tmp_dir).exists():
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = os.path.abspath(tmp_dir)
    pdb_basename = os.path.basename(pdb_path)

    # os.system(f"cd {tmp_dir}")
    # os.system(f"cp {pdb_path} .")
    # os.system(f"mkdssp -i {pdb_basename} -o {pdb_id}.dssp")
    # os.system(f"java -jar {tool_obj_path}/PTGLgraphComputation.jar {pdb_id} --settingsfile {tool_obj_path}/PTGLgraphComputation_settings.txt")

    os.system(f"cd {tmp_dir}; cp {pdb_path} .; mkdssp -i {pdb_basename} -o {pdb_id}.dssp; java -jar {tool_obj_path}/PTGLgraphComputation.jar {pdb_id} --settingsfile {tool_obj_path}/PTGLgraphComputation_settings.txt --no-warn --silent")
    
    # load GML outputs
    gml_file_list = glob.glob(f"{tmp_dir}/{pdb_id}*albe_PG.gml")
    out_dict = {}
    for gml_file in gml_file_list:
        gml_file = os.path.basename(gml_file)
        chain_id = gml_file.split('_')[1] # auth_chain_id
        gml_out = nx.read_gml(f'{tmp_dir}/{gml_file}')
        out_dict[chain_id] = gml_out
    
    os.system(f"rm -r {tmp_dir}")
    return out_dict

    
def trim_PTGL_gml(gml_obj: nx.Graph, auth_resi_index: List):
    """Extract information from PTGL gml objects
    """
    out_feats = collections.defaultdict(dict)
    
    # e.g. graph info
    pdb_id = gml_obj.graph['pdbId'].lower()
    chain_auth_id = gml_obj.graph['chainId']
    
    authId_to_seqId = {f'{chain_auth_id}-{auth_tuple_id[1]}-{auth_tuple_id[2]}' : i for i, auth_tuple_id in enumerate(auth_resi_index)}
    # nodes info
    node_list = list(gml_obj.nodes(data=True)) # list of tuple(node_id,node_data)
    edge_list = list(gml_obj.edges(data=True))
    for node_i, node_t in enumerate(node_list):
        node_id, node_data = node_t
        # node_id: '26-H'
        # node_data: {'numInChain': 27, 'numResidues': 12, 'pdbResStart': 'A-404- ', 'pdbResEnd': 'A-415- ', 'dsspResStart': 402, 'dsspResEnd': 413, 'pdbResiduesFull': 'A-404- ,A-405- ,A-406- ,A-407- ,A-408- ,A-409- ,A-410- ,A-411- ,A-412- ,A-413- ,A-414- ,A-415- ', 'aaSequence': 'KCSEFGDAIIEN', 'sseType': 'H', 'fgNotationLabel': 'h'}
        # for multi-chain proteins, dsspResStart/End represent 1-based indices in the concatenated sequecnes
        # for proteins with inserted residues, such residues will be included in all cases
        pdbRes_list = node_data['pdbResiduesFull'].split(',')
        seqRes_ids = [authId_to_seqId[pdb_r] for pdb_r in pdbRes_list if pdb_r in authId_to_seqId.keys()]
        out_feats['sse_nodes'][node_id] = {
            'seqRes_ids': seqRes_ids,
            'sse_type': node_data['sseType'],
            'aa_seq': node_data['aaSequence']}
    for edge_i, edge_t in enumerate(edge_list):
        node_s, node_t, edge_data = edge_t
        # node_s: '24-H'
        # node_t: '26-H'
        # edge_data: {'label': 'a', 'spatial': 'a'}
        # m=mixed, p=parallel, a=antiparallel, l=ligand contact
        out_feats['sse_edges'][edge_i] = {
            'node_s': node_s,
            'node_t': node_t,
            'edge_type': edge_data['label']}
    return out_feats

def voronota_contact_dataframe(
        voronota_dir: str,
        pdb_id: str,
        chain_auth_id: str,
        pdb_file_dir: str,
        min_seq_sep: int = 5,
        min_area: float = 0.0,
        max_dist: float = 10.0,
        node_edge_process: bool = False,):
    """Acquire protein Voronoi diagram of balls with Voronota, where each ball represents an atom of
    some Van Der Waals radius.
    Two working modes:
        Single residue - find atom composition of surrounding contacts
        Pair of residues - find atom composition of interacting contacts
    
    Args:
        voronota_dir: str, dir containing executable files of voronota
        pdb_id: str, 4-letter pdb id
        chain_auth_id: str, authored defined chain id
        pdb_file_dir: str, dir containing structure cif files
        min_seq_sep: int, min sequential positional seperations for contact calculation, default 5
        min_area: float,  min area to be considered as contacts, default 0.0
        max_dist: float, max distance to be considered as contacts, default 10.0
        node_edge_process: bool, generate node/edge feature vectors, default False
    Returns:
        dataframe for atom contacts: keys - 'chain_auth_id_1','res_auth_idx_1','iCode_1','serial_1','altLoc_1','res_name_1','atom_name_1','chain_auth_id_2','res_auth_idx_2','iCode_2','serial_2','altLoc_2','res_name_2','atom_name_2','area','dist','tags','adjuncts'
    """
    # decompress pdb cif.gz
    pdb_file_dir = os.path.abspath(pdb_file_dir)
    parent_dir = os.path.dirname(pdb_file_dir)
    if not os.path.isdir(f"{parent_dir}/tmp"):
        os.mkdir(f"{parent_dir}/tmp")
    os.system(f"gunzip -c {pdb_file_dir}/{pdb_id.lower()}.cif.gz > {parent_dir}/tmp/{pdb_id.lower()}.cif")
    try:
        os.system(f"./{voronota_dir}/voronota-contacts -i {parent_dir}/tmp/{pdb_id.lower()}.cif --contacts-query '--match-first c<{chain_auth_id}> --match-second c<{chain_auth_id}>' --contacts-query-additional '--match-min-seq-sep {min_seq_sep} --match-min-area {min_area} --match-max-dist {max_dist} --no-solvent' --cache-dir ./ > {parent_dir}/tmp/contacts.txt")
        os.system(f"cat {parent_dir}/tmp/contacts.txt | ./{voronota_dir}/voronota expand-descriptors | column -t > {parent_dir}/tmp/table_contacts.txt")
        atom_contacts_df = pd.read_csv(f'{parent_dir}/tmp/table_contacts.txt',sep='\s+',header=None,names=['chain_auth_id_1','res_auth_idx_1','iCode_1','serial_1','altLoc_1','res_name_1','atom_name_1','chain_auth_id_2','res_auth_idx_2','iCode_2','serial_2','altLoc_2','res_name_2','atom_name_2','area','dist','tags','adjuncts'])
    except pd.errors.EmptyDataError:
        atom_contacts_df = None
    except Exception:
        atom_contacts_df = None

    if node_edge_process:
        pass
        # generate node/edge features

    return atom_contacts_df
