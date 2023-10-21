"""Library for parsing different data structures."""
from typing import Union, Dict
from Bio.PDB.Chain import Chain
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.Data import SCOPData
import numpy as np

from src.data.components.structure_utils import residue_constants, protein
from src.data.components.structure_utils import utils as du

ProteinChain = protein.ProteinChain
Protein = protein.Protein


def process_chain(chain: Chain, auth_chain_id: Union[str,int], dssp_obj: DSSP=None, resiDepth_obj: ResidueDepth=None) -> ProteinChain:
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
    if isinstance(auth_chain_id, str):
        chain_id_int = du.chain_str_to_int(auth_chain_id)
    elif isinstance(auth_chain_id, int):
        chain_id_int = auth_chain_id
    
    process_dssp_rd = (dssp_obj is not None) and (resiDepth_obj is not None)
    atom_positions = []
    aatype = []
    resi_str = ""
    resi_type3 = []
    atom_mask = []
    residue_auth_index = []
    b_factors = []
    sse3_ids = []
    sse8_ids = []
    resi_depth = []
    ca_atom_depth = []
    for res in chain:
        res_id = res.id #(hetero flag, sequence identifier, insertion code)
        if res_id[0] != ' ':
            continue
        res_name = res.resname
        #res_shortname = residue_constants.restype_3to1.get(res_name, 'X')
        res_shortname = SCOPData.protein_letters_3to1.get(res_name, 'X')
        if res_shortname in 'BZJUO':
            res_shortname = residue_constants.ambiguous_restype_1to1(res_shortname)
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num) # default value: 20
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
        resi_str += res_shortname
        resi_type3.append(res_name)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_auth_index.append(res_id)
        b_factors.append(res_b_factors)

        # process DSSP and ResidueDepth outputs
        if process_dssp_rd:
            sse8_char = dssp_obj[(auth_chain_id, res_id)][2] if (auth_chain_id, res_id) in dssp_obj.keys() else '-'
            sse8 = residue_constants.SS8_char2id[sse8_char]
            sse3 = residue_constants.SS3_char2id[residue_constants.SS8_to_SS3[sse8_char]]
            
            if (auth_chain_id, res_id) in resiDepth_obj.keys(): 
                resi_rd = resiDepth_obj[(auth_chain_id, res_id)][0]
                ca_atom_rd = resiDepth_obj[(auth_chain_id, res_id)][1]
            else:
                resi_rd = np.nan
                ca_atom_rd = np.nan
            sse8_ids.append(sse8)
            sse3_ids.append(sse3)
            resi_depth.append(resi_rd)
            ca_atom_depth.append(ca_atom_rd)

    return ProteinChain(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        seqres=resi_str,
        res_type3=resi_type3,
        residue_auth_index=residue_auth_index,
        auth_chain_id=auth_chain_id,
        b_factors=np.array(b_factors),
        sse3_type_ids=np.array(sse3_ids, dtype=np.int8) if len(sse3_ids) > 0 else None,
        sse8_type_ids=np.array(sse8_ids, dtype=np.int8) if len(sse8_ids) > 0 else None,
        depth_resi=np.array(resi_depth) if len(resi_depth) > 0 else None,
        depth_ca_atom=np.array(ca_atom_depth) if len(ca_atom_depth) > 0 else None,
        )
    