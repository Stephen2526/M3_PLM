"""Script for preprocessing mmcif files for faster consumption.

- Parses all mmcif protein files in a directory.
- Filters out low resolution files.
- Performs any additional processing.
- Writes all processed examples out to specified path.
"""

import argparse, glob, random
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
from tqdm import tqdm
import numpy as np
import mdtraj as md
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import ResidueDepth

from src.data.components.structure_utils import mmcif_parsing, residue_constants, errors, parsers
import src.data.components.structure_utils.utils as du
from src.data.components.openfold.data.parsers import parse_hhr


# Define the parser
parser = argparse.ArgumentParser(
    description='mmCIF processing script.')
parser.add_argument(
    '--mmcif_dir',
    help='Path to directory with mmcif files.',
    type=str)
parser.add_argument(
    '--max_file_size',
    help='Max file size.',
    type=int,
    default=3000000)  # Only process files up to 3MB large.
parser.add_argument(
    '--min_file_size',
    help='Min file size.',
    type=int,
    default=1000)  # Files must be at least 1KB.
parser.add_argument(
    '--max_resolution',
    help='Max resolution of files.',
    type=float,
    default=5.0)
parser.add_argument(
    '--max_len',
    help='Max length of protein.',
    type=int,
    default=1024)
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=100)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,
    default='./data/processed_pdb')
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')
parser.add_argument(
    '--tmp_dir',
    help='Temporary dir storing intermediate files.',
    type=str,
    default='./data/tmp_dir')
parser.add_argument(
    '--PTGL_path',
    help='Path to PTGL executable files.',
    type=str)
parser.add_argument(
    '--OpenProtein_path',
    help='Path to OpenProteinSet data files.',
    type=str)

def _retrieve_mmcif_files(
        mmcif_dir: str, max_file_size: int, min_file_size: int, debug: bool):
    """Set up all the mmcif files to read."""
    print('Gathering mmCIF paths')
    total_num_files = 0
    all_mmcif_paths = []
    mmcif_dir = args.mmcif_dir
    for subdir in tqdm(os.listdir(mmcif_dir)):
        mmcif_file_dir = os.path.join(mmcif_dir, subdir)
        if not os.path.isdir(mmcif_file_dir):
            continue
        for mmcif_file in os.listdir(mmcif_file_dir):
            mmcif_path = os.path.join(mmcif_file_dir, mmcif_file)
            total_num_files += 1
            if min_file_size <= os.path.getsize(mmcif_path) <= max_file_size:
                all_mmcif_paths.append(mmcif_path)
        if debug and total_num_files >= 100:
            # Don't process all files for debugging
            break
    print(
        f'Processing {len(all_mmcif_paths)} files our of {total_num_files}')
    return all_mmcif_paths

def _retrieve_mmcif_files_4OpenProteinSet(
        mmcif_dir: str, min_file_size: int=1000, debug: bool=False):
    """Set up all the mmcif files to read."""
    print('Gathering mmCIF paths')
    total_num_files = 0
    all_mmcif_paths = []
    mmcif_list = os.listdir(mmcif_dir)
    random.shuffle(mmcif_list)
    for cif_fl in tqdm(mmcif_list):
        mmcif_path = os.path.join(mmcif_dir,cif_fl)
        total_num_files += 1
        if os.path.getsize(mmcif_path) >= min_file_size:
            all_mmcif_paths.append(mmcif_path)
        if debug and total_num_files >= 100:
            # Don't process all files for debugging
            break
    print(f'Processing {len(all_mmcif_paths)} files out of {total_num_files}')
    return all_mmcif_paths

def process_mmcif_save_pkl_4openProteinSet(
        mmcif_path: str, write_dir: str, tmp_dir: str, PTGL_path: str, OpenProtein_path: str, max_resolution: int=5.0, max_len: int=2014):
    """Processes MMCIF files into usable, smaller pickles.
    Reduces the repetitive calculation of Secondary Structure(& SSE interaction) and residue depth, as these information cannot be read from pdb file directly.

    Args:
        mmcif_path: Path to mmcif file to read.
        max_resolution: Max resolution to allow.
        max_len: Max length to allow.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.
        
    MetaData keys:
        pdb_name_lower (str): 106m
        pkl_file_path (str): /path/to/OpenProteinSet/pdb_pickle/106m.pkl
        mmcif_file_path (str): /path/to/OpenProteinSet/pdb_mmcif/106m.cif
        oligomeric_count (str): 1:...
        oligomeric_detail (str): monomeric:...
        resolution (float): 1.99
        release_date (str): 1998-04-08
        structure_method (str): x-ray diffraction:...
        num_chains (int): 1
        pdb_auth_chain (str): 106m_A
        quaternary_category (str): homomer
        seq_len (int): 154
        modeled_seq_len (int): 150
        coil_percent (float): 0.3
        helix_percent (float): 0.4
        strand_percent (float): 0.3
        radius_gyration (float): 5.2
        msa_path (str): /path/to/OpenProteinSet/pdb/106m_A/a3m/uniref90_hits.a3m or None
        pdb_temp_path (str): /path/to/OpenProteinSet/pdb/106m_A/hhr/pdb70_hits.hhr or None
        auth_chain_in_OPS (bool): True or False
    
    Pickle complex features: refer to protein.ProteinChain

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata_shared = {}
    mmcif_name = os.path.basename(mmcif_path).replace('.cif', '').lower()
    metadata_shared['pdb_name_lower'] = mmcif_name
    mmcif_subdir = os.path.join(write_dir, mmcif_name[1:3].lower()) # middle two letters for pickle file grouping
    if not os.path.isdir(mmcif_subdir):
        os.mkdir(mmcif_subdir)
    processed_mmcif_path = os.path.join(mmcif_subdir, f'{mmcif_name}.pkl')
    processed_mmcif_path = os.path.abspath(processed_mmcif_path)
    metadata_shared['pkl_file_path'] = processed_mmcif_path
    with open(mmcif_path, 'r') as f:
        parsed_mmcif = mmcif_parsing.parse(
            file_id=mmcif_name, mmcif_string=f.read())
    metadata_shared['mmcif_file_path'] = mmcif_path
    # probe parsing error
    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(
            f'Encountered errors {parsed_mmcif.errors}'
        )
    # parsed_mmcif: mmcif_parsing.MmcifObject
    parsed_mmcif = parsed_mmcif.mmcif_object
    raw_mmcif = parsed_mmcif.raw_string
    if '_pdbx_struct_assembly.oligomeric_count' in raw_mmcif:
        raw_olig_count = raw_mmcif['_pdbx_struct_assembly.oligomeric_count']
        oligomeric_count = ':'.join(raw_olig_count).lower()
    else:
        oligomeric_count = None
    if '_pdbx_struct_assembly.oligomeric_details' in raw_mmcif:
        raw_olig_detail = raw_mmcif['_pdbx_struct_assembly.oligomeric_details']
        oligomeric_detail = ':'.join(raw_olig_detail).lower()
    else:
        oligomeric_detail = None
    metadata_shared['oligomeric_count'] = oligomeric_count
    metadata_shared['oligomeric_detail'] = oligomeric_detail

    # Parse mmcif header
    mmcif_header = parsed_mmcif.header
    mmcif_resolution = mmcif_header['resolution']
    metadata_shared['resolution'] = mmcif_resolution
    metadata_shared['release_date'] = mmcif_header['release_date']
    metadata_shared['structure_method'] = mmcif_header['structure_method']
    if mmcif_resolution >= max_resolution:
        raise errors.ResolutionError(
            f'Too high resolution {mmcif_resolution}')
    if mmcif_resolution == 0.0:
        raise errors.ResolutionError(
            f'Invalid resolution {mmcif_resolution}')

    # Extract all pipetide chains {auth_chain_id: Chain}
    auth_cid_to_seq = parsed_mmcif.chain_to_seqres
    mmcif_model = parsed_mmcif.structure
    mmcif_to_auth_cid = parsed_mmcif.mmcif_to_auth_cid
    all_seqs = set()
    struct_chains = {}
    for auth_cid, entitySeq in auth_cid_to_seq.items():
        struct_chains[auth_cid] = mmcif_model[auth_cid]
        all_seqs.add(entitySeq)
    
    if len(all_seqs) == 1:
        metadata_shared['quaternary_category'] = 'homomer'
    else:
        metadata_shared['quaternary_category'] = 'heteromer'
    metadata_shared['num_chains'] = len(struct_chains)
    
    try:
        # Biopython calculation of SS and residueDepth
        dssp = DSSP(mmcif_model, mmcif_path, dssp='mkdssp')
        rd = ResidueDepth(mmcif_model)
        # SSE contacts
        SSE_contact_chains = du.run_PTGLtools(mmcif_name, mmcif_path, PTGL_path, f"{tmp_dir}/{mmcif_name}")
    except Exception as e:
        raise errors.DataError(f'Exit DSSP/ResidueDepth/PTGL with error {e}')
    
    # Extract features
    metadata_chains = []
    struct_feats = {}
    
    for auth_chain_id, chain in struct_chains.items():
        metadata_tmp = {}
        chain_prot = parsers.process_chain(chain, auth_chain_id, dssp_obj=dssp, resiDepth_obj=rd)
        chain_prot = du.parse_chain_feats(chain_prot)
        
        # find modeled indices
        modeled_idx = np.where(chain_prot.aatype != 20)[0]
        if len(modeled_idx) == 0:
            raise errors.LengthError('No modeled residues')
        min_modeled_idx = np.min(modeled_idx)
        max_modeled_idx = np.max(modeled_idx)
        modeled_seq_len = int(max_modeled_idx - min_modeled_idx + 1)
        chain_prot.modeled_idx = modeled_idx
        if chain_prot.aatype.shape[0] > max_len:
            raise errors.LengthError(
                f"Too long {chain_prot.aatype.shape[0]}")

        if auth_chain_id not in SSE_contact_chains.keys():
            continue
        # SSE contacts (sse_nodes:{node_id:{...}, ...}, sse_edges: {0:{...}, ...})
        chain_prot.sse_contacts = du.trim_PTGL_gml(SSE_contact_chains[auth_chain_id], chain_prot.residue_auth_index)
        
        # update chain-specific metadata
        metadata_tmp['pdb_auth_chain'] = f'{mmcif_name}_{auth_chain_id}'
        metadata_tmp['seq_len'] = len(chain_prot.aatype)
        metadata_tmp['modeled_seq_len'] = modeled_seq_len
        
        # SSE percentage
        metadata_tmp['coil_percent'] = np.sum(chain_prot.sse3_type_ids == residue_constants.SS3_char2id['C']) / modeled_seq_len
        metadata_tmp['helix_percent'] = np.sum(chain_prot.sse3_type_ids == residue_constants.SS3_char2id['H']) / modeled_seq_len
        metadata_tmp['strand_percent'] = np.sum(chain_prot.sse3_type_ids == residue_constants.SS3_char2id['E']) / modeled_seq_len 

        OPS_exist = os.path.exists(f"{OpenProtein_path}/pdb/{mmcif_name}_{auth_chain_id}")
        if OPS_exist:
            # MSA
            msa_path = os.path.abspath(f"{OpenProtein_path}/pdb/{mmcif_name}_{auth_chain_id}/a3m/uniref90_hits.a3m")
            pdbTemp_path = os.path.abspath(f"{OpenProtein_path}/pdb/{mmcif_name}_{auth_chain_id}/hhr/pdb70_hits.hhr")
            rawSeq_ids, _ = du.parse_a3m_rawSeq(msa_path)
            chain_prot.homoSeq_aatype = rawSeq_ids
            chain_in_OPS = True
            
            # pdb templates
            with open(pdbTemp_path,'r') as fp:
                hhr_string = fp.read()
            temp_hits = parse_hhr(hhr_string)
            chain_prot.pdb_temp = [h.pdb_chain_id for h in temp_hits]
        else:
            msa_path = None
            pdbTemp_path = None
            chain_in_OPS = False
            chain_prot.homoSeq_aatype = None
            chain_prot.pdb_temp = None
        metadata_tmp['msa_path'] = msa_path
        metadata_tmp['pdb_temp_path'] = pdbTemp_path
        metadata_tmp['auth_chain_in_OPS'] = chain_in_OPS
        metadata_chains.append(metadata_tmp)
        # chain_dict = dataclasses.asdict(chain_prot) # convert to dict
        struct_feats[auth_chain_id] = chain_prot

    #complex_feats = du.concat_np_features(struct_feats, False)

    try:
        # Workaround for MDtraj not supporting mmcif in their latest release.
        # MDtraj source does support mmcif https://github.com/mdtraj/mdtraj/issues/652
        # We temporarily save the mmcif as a pdb and delete it after running mdtraj.
        p = MMCIFParser(QUIET=True)
        struc = p.get_structure("", mmcif_path)
        io = PDBIO()
        io.set_structure(struc)
        pdb_path = mmcif_path.replace('.cif', '.pdb')
        io.save(pdb_path)

        # MDtraj
        traj = md.load(pdb_path)
        # SS calculation
        #pdb_ss = md.compute_dssp(traj, simplified=True)[0]
        # RG calculation
        pdb_rg = md.compute_rg(traj)[0]
        os.remove(pdb_path)
    except Exception as e:
        os.remove(pdb_path)
        raise errors.DataError(f'Mdtraj failed with error {e}')

    # metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    # metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    # metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata_shared['radius_gyration'] = pdb_rg

    # update chain-specfic metadata with shared entries
    metadata_out =[]
    for meta_chain in metadata_chains:
        meta_chain.update(metadata_shared)
        metadata_out.append(meta_chain)


    # Write features to pickles.
    du.write_pkl(processed_mmcif_path, struct_feats, create_dir=True)

    # Return metadata
    return metadata_out


def process_mmcif(
        mmcif_path: str, max_resolution: int, max_len: int, write_dir: str):
    """Processes MMCIF files into usable, smaller pickles.

    Args:
        mmcif_path: Path to mmcif file to read.
        max_resolution: Max resolution to allow.
        max_len: Max length to allow.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    mmcif_name = os.path.basename(mmcif_path).replace('.cif', '')
    metadata['pdb_name'] = mmcif_name
    mmcif_subdir = os.path.join(write_dir, mmcif_name[1:3].lower())
    if not os.path.isdir(mmcif_subdir):
        os.mkdir(mmcif_subdir)
    processed_mmcif_path = os.path.join(mmcif_subdir, f'{mmcif_name}.pkl')
    processed_mmcif_path = os.path.abspath(processed_mmcif_path)
    metadata['processed_path'] = processed_mmcif_path
    with open(mmcif_path, 'r') as f:
        parsed_mmcif = mmcif_parsing.parse(
            file_id=mmcif_name, mmcif_string=f.read())
    metadata['raw_path'] = mmcif_path
    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(
            f'Encountered errors {parsed_mmcif.errors}'
        )
    parsed_mmcif = parsed_mmcif.mmcif_object
    raw_mmcif = parsed_mmcif.raw_string
    if '_pdbx_struct_assembly.oligomeric_count' in raw_mmcif:
        raw_olig_count = raw_mmcif['_pdbx_struct_assembly.oligomeric_count']
        oligomeric_count = ','.join(raw_olig_count).lower()
    else:
        oligomeric_count = None
    if '_pdbx_struct_assembly.oligomeric_details' in raw_mmcif:
        raw_olig_detail = raw_mmcif['_pdbx_struct_assembly.oligomeric_details']
        oligomeric_detail = ','.join(raw_olig_detail).lower()
    else:
        oligomeric_detail = None
    metadata['oligomeric_count'] = oligomeric_count
    metadata['oligomeric_detail'] = oligomeric_detail

    # Parse mmcif header
    mmcif_header = parsed_mmcif.header
    mmcif_resolution = mmcif_header['resolution']
    metadata['resolution'] = mmcif_resolution
    metadata['structure_method'] = mmcif_header['structure_method']
    if mmcif_resolution >= max_resolution:
        raise errors.ResolutionError(
            f'Too high resolution {mmcif_resolution}')
    if mmcif_resolution == 0.0:
        raise errors.ResolutionError(
            f'Invalid resolution {mmcif_resolution}')

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in parsed_mmcif.structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['seq_len'] = len(complex_aatype)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx
    if complex_aatype.shape[0] > max_len:
        raise errors.LengthError(
            f'Too long {complex_aatype.shape[0]}')

    try:
        
        # Workaround for MDtraj not supporting mmcif in their latest release.
        # MDtraj source does support mmcif https://github.com/mdtraj/mdtraj/issues/652
        # We temporarily save the mmcif as a pdb and delete it after running mdtraj.
        p = MMCIFParser()
        struc = p.get_structure("", mmcif_path)
        io = PDBIO()
        io.set_structure(struc)
        pdb_path = mmcif_path.replace('.cif', '.pdb')
        io.save(pdb_path)

        # MDtraj
        traj = md.load(pdb_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        os.remove(pdb_path)
    except Exception as e:
        os.remove(pdb_path)
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]

    # Write features to pickles.
    du.write_pkl(processed_mmcif_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(
        all_mmcif_paths,
        max_resolution,
        max_len,
        write_dir,
        tmp_dir:str=None, 
        PTGL_path:str=None,
        OpenProtein_path:str=None):
    all_metadata = []
    for i, mmcif_path in enumerate(all_mmcif_paths):
        try:
            start_time = time.time()
            # metadata = process_mmcif(
            #     mmcif_path,
            #     max_resolution,
            #     max_len,
            #     write_dir)
            metadata = process_mmcif_save_pkl_4openProteinSet(
                    mmcif_path,
                    write_dir, 
                    tmp_dir,
                    PTGL_path,
                    OpenProtein_path,
                    max_resolution,
                    max_len)
            elapsed_time = time.time() - start_time
            print(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
            all_metadata.extend(metadata)
        except errors.DataError as e:
            print(f'Failed {mmcif_path}: {e}')
    return all_metadata


def process_fn(
        mmcif_path,
        verbose=None,
        max_resolution=None,
        max_len=None,
        write_dir=None,
        tmp_dir:str=None,
        PTGL_path:str=None,
        OpenProtein_path:str=None):
    try:
        start_time = time.time()
        # metadata = process_mmcif(
        #     mmcif_path,
        #     max_resolution,
        #     max_len,
        #     write_dir)
        metadata = process_mmcif_save_pkl_4openProteinSet(
            mmcif_path,
            write_dir, 
            tmp_dir,
            PTGL_path,
            OpenProtein_path,
            max_resolution,
            max_len)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {mmcif_path}: {e}')


def main(args):
    # Get all mmcif files to read.
    # all_mmcif_paths = _retrieve_mmcif_files(
    #     args.mmcif_dir, args.max_file_size, args.min_file_size, args.debug)
    all_mmcif_paths = _retrieve_mmcif_files_4OpenProteinSet(
        args.mmcif_dir, args.min_file_size, args.debug)
    total_num_paths = len(all_mmcif_paths)
    write_dir = args.write_dir
    tmp_dir = args.tmp_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_mmcif_paths,
            args.max_resolution,
            args.max_len,
            write_dir,
            tmp_dir=tmp_dir, 
            PTGL_path=args.PTGL_path,
            OpenProtein_path=args.OpenProtein_path)
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            max_resolution=args.max_resolution,
            max_len=args.max_len,
            write_dir=write_dir,
            tmp_dir=tmp_dir, 
            PTGL_path=args.PTGL_path,
            OpenProtein_path=args.OpenProtein_path)
        # Uses max number of available cores.
        with mp.Pool() as pool:
            pool_metadata = pool.map(_process_fn, all_mmcif_paths)
        all_metadata = []
        for x in pool_metadata:
            if len(x) > 0:
                all_metadata.extend(x)
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, sep='\t', index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)