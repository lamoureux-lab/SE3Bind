import os
import re

import pandas as pd

def gather_dataset_ids(data):
    
    data = pd.read_csv(data, sep=",")
    dataset_ids_dict = dict(zip(data["PDB_ID"].str.lower(), data["Antibody_chains"].str.lower()))
    return dataset_ids_dict

def parse_html(html_file_path, concat_all_cdrs=False):
    """"
    parse sbabad html to collect CDR sequences for all pdb_dis
    """
    # print("Parsing HTML file:", html_file_path)

    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.readlines()
        start_line = '<tr><td style="text-align: center;">'
        start_reading = False
        buffer = []
        for idx, line in enumerate(html_content):
            if not start_reading:
                if start_line in line:
                    start_reading = True
                continue
            if line.strip().startswith('<tr><td style="text-align: center;">'):
                # Collect all lines for this entry
                match_lines = [line]
                while not line.strip().endswith('</a><br></td>'):
                    idx += 1
                    line = html_content[idx]
                    match_lines.append(line)
                full_line = ''.join(match_lines)
                # Extract PDB, region, and CDR sequence
                matches = re.findall(
                    r"pdb=([a-zA-Z0-9]+)&(loop=CDR\w+&chain=\w+&).*?'>([^<]+)</a>",
                    full_line
                )
                if matches:
                    buffer.extend(matches)

    # print('buffer', buffer)

    pdb2cdr_dict = {}
    for pdb, region, cdr_sequence in buffer:
        if pdb not in pdb2cdr_dict:
            pdb2cdr_dict[pdb] = []
        # Clean region: extract CDR and chain
        match = re.match(r'loop=(CDR\w+)&chain=(\w+)&', region)
        if match:
            clean_region = f"{match.group(1)},{match.group(2)}"
        else:
            clean_region = region  # fallback if pattern doesn't match

        pdb2cdr_dict[pdb].append([clean_region, cdr_sequence])


    all_concat_clean_pdb2cdr_dict = {}
    separate_cdrs_dict= {} # concat each loop separately, H1-H1, H2-H2,H3-H3 L1-L1, L2-L2,H3-H3

    if concat_all_cdrs:
        ## concat all cdrs for heavy chains together and light chains together.
        
        for k, v in pdb2cdr_dict.items():
            H_cdrs = {}
            L_cdrs = {}
            chain_order = []
            for entry in v:
                region_label = entry[0]
                cdr_dict_chainis = entry[0].split(",")[1]
                
                seq = entry[1]
                if region_label.startswith("CDRH"):
                    if cdr_dict_chainis in H_cdrs:
                        H_cdrs[cdr_dict_chainis] += seq
                    else:
                        H_cdrs[cdr_dict_chainis] = seq
                elif region_label.startswith("CDRL"):
                    if cdr_dict_chainis in L_cdrs:
                        L_cdrs[cdr_dict_chainis] += seq
                    else:
                        L_cdrs[cdr_dict_chainis] = seq

                all_concat_clean_pdb2cdr_dict[k] = [H_cdrs, L_cdrs]


    else:
        for k, v in pdb2cdr_dict.items():
            H1_cdrs = {}
            H2_cdrs = {}
            H3_cdrs = {}
            L1_cdrs = {}
            L2_cdrs = {}
            L3_cdrs = {}

            chain_order = []
            for entry in v:
                region_label = entry[0]
                cdr_dict_chainis = entry[0].split(",")[1]
                seq = entry[1]

                if region_label.startswith("CDRH1"):
                    if cdr_dict_chainis in H1_cdrs:
                        H1_cdrs[cdr_dict_chainis] += seq
                    else:
                        H1_cdrs[cdr_dict_chainis] = seq
                if region_label.startswith("CDRH2"):
                    if cdr_dict_chainis in H2_cdrs:
                        H2_cdrs[cdr_dict_chainis] += seq
                    else:
                        H2_cdrs[cdr_dict_chainis] = seq
                if region_label.startswith("CDRH3"):
                    if cdr_dict_chainis in H3_cdrs:
                        H3_cdrs[cdr_dict_chainis] += seq
                    else:
                        H3_cdrs[cdr_dict_chainis] = seq

                elif region_label.startswith("CDRL1"):
                    if cdr_dict_chainis in L1_cdrs:
                        L1_cdrs[cdr_dict_chainis] += seq
                    else:
                        L1_cdrs[cdr_dict_chainis] = seq
                elif region_label.startswith("CDRL2"):
                    if cdr_dict_chainis in L2_cdrs:
                        L2_cdrs[cdr_dict_chainis] += seq
                    else:
                        L2_cdrs[cdr_dict_chainis] = seq
                elif region_label.startswith("CDRL3"):
                    if cdr_dict_chainis in L3_cdrs:
                        L3_cdrs[cdr_dict_chainis] += seq
                    else:
                        L3_cdrs[cdr_dict_chainis] = seq
                region_label = v[0][0]
                separate_cdrs_dict[k] = [H1_cdrs, H2_cdrs, H3_cdrs, L1_cdrs, L2_cdrs, L3_cdrs]

    return all_concat_clean_pdb2cdr_dict, separate_cdrs_dict


def match_allloops_cdr2fasta(cdr_dict, AB_fasta_sequences ):
    """
    Extract and clean Dict to contain only antibody dataset ids and CDR sequences
    cdr_dict is a dict of pdb_id: [H_cdrs, L_cdrs] where H_cdrs and L_cdrs are dicts of all 3 CDR loops per chain 
    concated together regions
    returns: 3 dicts, Heavy chain dict, light chain dict, combined dict [joined_H, joined_L,'joined_H + joined_L']
    """
    
    heavy_chain_dict = {}
    light_chain_dict = {}
    H_L_combined_dict = {}
    with open(AB_fasta_sequences, 'r') as file:
        for line in file:
            if line.startswith('>'):
                pdb_id = line[1:].strip().split('|')[0].lower()[:-2]  # Extract PDB ID
                chains = line[1:].strip().split('|')[1].split(" ") #[1].lower()  # Extract chain ID
                if len(chains)>1:
                    chain_id = chains[1:]
                    chain_id_str = ' '.join(chain_id)
                    chain_id = ''.join(re.findall(r'\b([A-Za-z])\b', chain_id_str)).lower()

                for k, v in cdr_dict.items():
                    if k != pdb_id or not chain_id:
                        continue
                    H_cdrs_dict, L_cdrs_dict = v
                    for c in chain_id:
                        c_up = c.upper()
                        if c_up in H_cdrs_dict:
                            if pdb_id not in heavy_chain_dict:
                                heavy_chain_dict[pdb_id] = []
                            heavy_chain_dict[pdb_id].append([c, H_cdrs_dict.keys(), H_cdrs_dict[c_up]])
                    
                        if c_up in L_cdrs_dict:
                            if pdb_id not in light_chain_dict:
                                light_chain_dict[pdb_id] = []
                            light_chain_dict[pdb_id].append([c, L_cdrs_dict.keys(), L_cdrs_dict[c_up]])
                        
                        if pdb_id not in H_L_combined_dict:
                            joined_H = ''.join(H_cdrs_dict.values())
                            joined_L = ''.join(L_cdrs_dict.values())
                            H_L_combined_dict[pdb_id] = [[joined_H, joined_L, joined_H + joined_L]]


    return heavy_chain_dict, light_chain_dict, H_L_combined_dict


def convert_dict2fasta(heavy_chain_dict, light_chain_dict, H_L_combined_dict, save_path):

    heavy_save_name = os.path.join(save_path, "heavy_chain_cdrs.fasta")
    light_save_name = os.path.join(save_path, "light_chain_cdrs.fasta")
    heavy_light_combined_save_name = os.path.join(save_path, "heavy_light_combined_chain_cdrs.fasta")

    # Write heavy chain CDRs
    with open(heavy_save_name, "w") as f:
        for k, v in heavy_chain_dict.items():
            chain_id = v[0][0]
            cdr_seq = v[0][-1]
            header = f'>{k}|{chain_id}|'
            f.write(header + '\n')
            f.write(cdr_seq + '\n')

    # Write light chain CDRs
    with open(light_save_name, "w") as f:
        for k, v in light_chain_dict.items():
            chain_id = v[0][0]
            cdr_seq = v[0][-1]
            header = f'>{k}|{chain_id}|'
            f.write(header + '\n')
            f.write(cdr_seq + '\n')

    # Write heavy-light combined CDRs
    with open(heavy_light_combined_save_name, "w") as f:
        for k, v in H_L_combined_dict.items():
            joined_H, joined_L, joined_HL = v[0]
            header = f'>{k}|H+L|'
            f.write(header + '\n')
            f.write(joined_HL + '\n')


def allconcat_cdr_seq_lens(heavy_chain_dict, light_chain_dict, combined_dict, train_dataset_dict,  valid_dataset_dict, save_path):
    """
    Based on similar sequence length, get sequence identity, look at distribution within train and validset, and between
    train and and valid set.

    get all sequences of same length -> run Clustal Omega to get precent identity - then split by dataset
    """
    
    train_setids_list = train_dataset_dict.keys()
    valid_setis_list = valid_dataset_dict.keys()
    print("valid_set_list",valid_setis_list, len(valid_setis_list))

    # print('heavy_chain_dict', heavy_chain_dict)

    ## sort H/L and combined dict based on length of sequence

    len_sorted_heavy_dict = {}
    len_sorted_light_dict = {}
    len_sorted_combined_dict = {}

    # Sort heavy_chain_dict by sequence length
    for k, v in heavy_chain_dict.items():
        seq = v[0][-1]
        seq_len = len(seq)
        if seq_len not in len_sorted_heavy_dict:
            len_sorted_heavy_dict[seq_len] = []
        len_sorted_heavy_dict[seq_len].append((k, seq))

    # Sort light_chain_dict by sequence length
    for k, v in light_chain_dict.items():
        seq = v[0][-1]
        seq_len = len(seq)
        if seq_len not in len_sorted_light_dict:
            len_sorted_light_dict[seq_len] = []
        len_sorted_light_dict[seq_len].append((k, seq))

    # Sort combined_dict by sequence length (joined_HL)
    for k, v in combined_dict.items():
        seq = v[0][2]  # joined_HL
        seq_len = len(seq)
        if seq_len not in len_sorted_combined_dict:
            len_sorted_combined_dict[seq_len] = []
        len_sorted_combined_dict[seq_len].append((k, seq))

    print("Heavy chain sequence lengths:", len_sorted_heavy_dict)

    def process_chain_length_dict(len_sorted_dict, chain_label, setids_list, valid_setis_list, save_path=None, save_fasta=False):
        """
        Process a length-sorted dict for a chain type (heavy, light, combined).
        Returns: (len_train_valid_counts, len_train_valid_df)
        """
        len_train_valid_counts = {}
        len_train_valid_df = pd.DataFrame(columns=['seq_len', 'train_count', 'valid_count', 'train_ids', 'valid_ids'])
        for seq_len, entries in len_sorted_dict.items():
            print(seq_len, len(entries))
            for entry in entries:
                seq = entry[-1]
                header = f'>{entry[0]}|{chain_label}|'
                # Optionally write to fasta if save_path is provided

                if save_fasta:
                    if save_path is not None:
                        fasta_filename = os.path.join(save_path, f'{chain_label.lower()}_chain_len_{seq_len}.fasta')
                        with open(fasta_filename, 'a') as fasta_file:
                            fasta_file.write(header + '\n')
                            fasta_file.write(seq + '\n')
                        
            train_count = 0
            valid_count = 0
            train_ids = []
            valid_ids = []
            for entry in entries:
                pdb_id = entry[0]
                if pdb_id in setids_list:
                    train_count += 1
                    train_ids.append(pdb_id)
                elif pdb_id in valid_setis_list:
                    valid_count += 1
                    valid_ids.append(pdb_id)
            len_train_valid_counts[seq_len] = {
                'train_count': train_count,
                'valid_count': valid_count,
                'train_ids': train_ids,
                'valid_ids': valid_ids
            }
            len_train_valid_df = pd.concat([
                len_train_valid_df,
                pd.DataFrame([{
                    'seq_len': seq_len,
                    'train_count': train_count,
                    'valid_count': valid_count,
                    'train_ids': train_ids,
                    'valid_ids': valid_ids
                }])
            ], ignore_index=True)
        # print(f"{chain_label} chain length train/valid counts:", len_train_valid_counts)
        # print(f'{chain_label.lower()}_len_train_valid_df\n', len_train_valid_df)
        return len_train_valid_counts, len_train_valid_df

    # Process each chain type using the modular function
    heavy_len_train_valid_counts, heavy_len_train_valid_df = process_chain_length_dict(
        len_sorted_heavy_dict, "Heavy", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    )
    light_len_train_valid_counts, light_len_train_valid_df = process_chain_length_dict(
        len_sorted_light_dict, "Light", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    )
    combined_len_train_valid_counts, combined_len_train_valid_df = process_chain_length_dict(
        len_sorted_combined_dict, "Combined", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    )

    # print("Combined chain length train/valid counts:", combined_len_train_valid_counts)
    # print('heavy_len_train_valid_df\n', heavy_len_train_valid_df)
    
    # print('light_len_train_valid_df\n', light_len_train_valid_df)

    
    # print('combined_len_train_valid_df\n', combined_len_train_valid_df)


def match_perloop_cdr2fasta(cdr_dict, AB_fasta_sequences ):
    """
    match each cdr loop to the antibody datasets
    
    """
    h1_chain_dict = {}
    h2_chain_dict = {}
    h3_chain_dict = {}
    l1_chain_dict = {}
    l2_chain_dict = {}
    l3_chain_dict = {}

    with open(AB_fasta_sequences, 'r') as file:
        for line in file:
            if line.startswith('>'):
                pdb_id = line[1:].strip().split('|')[0].lower()[:-2]  # Extract PDB ID
                chains = line[1:].strip().split('|')[1].split(" ") #[1].lower()  # Extract chain ID
                if len(chains)>1:
                    chain_id = chains[1:]
                    chain_id_str = ' '.join(chain_id)
                    chain_id = ''.join(re.findall(r'\b([A-Za-z])\b', chain_id_str)).lower()

                for k, v in cdr_dict.items():
                    if k != pdb_id or not chain_id:
                        continue
                    # if k == '5mev' or k == '6mlk':
                    h1_cdrs = v[0]
                    h2_cdrs = v[1]
                    h3_cdrs = v[2]
                    l1_cdrs = v[3]
                    l2_cdrs = v[4]
                    l3_cdrs = v[5]

                    for c in chain_id:
                        c_up = c.upper()
                        if c_up in h1_cdrs:
                            if pdb_id not in h1_chain_dict:
                                h1_chain_dict[pdb_id] = []
                            h1_chain_dict[pdb_id] = (h1_cdrs[c_up])
                        if c_up in h2_cdrs:
                            if pdb_id not in h2_chain_dict:
                                h2_chain_dict[pdb_id] = []
                            h2_chain_dict[pdb_id]=(h2_cdrs[c_up])
                        if c_up in h3_cdrs:
                            if pdb_id not in h3_chain_dict:
                                h3_chain_dict[pdb_id] = []
                            h3_chain_dict[pdb_id]=(h3_cdrs[c_up])
                        if c_up in l1_cdrs:
                            if pdb_id not in l1_chain_dict:
                                l1_chain_dict[pdb_id] = []
                            l1_chain_dict[pdb_id]=(l1_cdrs[c_up])
                        if c_up in l2_cdrs:
                            if pdb_id not in l2_chain_dict:
                                l2_chain_dict[pdb_id] = []
                            l2_chain_dict[pdb_id]=(l2_cdrs[c_up])
                        if c_up in l3_cdrs:
                            if pdb_id not in l3_chain_dict:
                                l3_chain_dict[pdb_id] = []
                            l3_chain_dict[pdb_id]=(l3_cdrs[c_up])


    for k, v in h1_chain_dict.items():
        if k == '5mev' or k== '3nh7':
            print("h1_chain_dict", k, v)
                    
    return h1_chain_dict, h2_chain_dict, h3_chain_dict, l1_chain_dict, l2_chain_dict, l3_chain_dict

def loops_cdr_seqby_lens(h1_chain_dict, h2_chain_dict, h3_chain_dict, 
                         l1_chain_dict, l2_chain_dict, l3_chain_dict,
                        train_dataset_dict,  valid_dataset_dict, save_path):
    
    train_setids_list = train_dataset_dict.keys()
    valid_setis_list = valid_dataset_dict.keys()
    print("valid_set_list",valid_setis_list, len(valid_setis_list))

    # print('heavy_chain_dict', heavy_chain_dict)

    ## sort H/L and combined dict based on length of sequence

    h1_len_sorted_dict = {}
    h2_len_sorted_dict = {}
    h3_len_sorted_dict = {}
    l1_len_sorted_dict = {}
    l2_len_sorted_dict = {}
    l3_len_sorted_dict = {}

    # print("h1_chain_dict",h1_chain_dict)
    # Sort heavy_chain_dict by sequence length
    # Sort h1_chain_dict by sequence length
    for k, v in h1_chain_dict.items():
        seq = v
        seq_len = len(seq)
        if seq_len not in h1_len_sorted_dict:
            h1_len_sorted_dict[seq_len] = []
        h1_len_sorted_dict[seq_len].append((k, seq))

    # Sort h2_chain_dict by sequence length
    for k, v in h2_chain_dict.items():
        seq = v
        seq_len = len(seq)
        if seq_len not in h2_len_sorted_dict:
            h2_len_sorted_dict[seq_len] = []
        h2_len_sorted_dict[seq_len].append((k, seq))

    # Sort h3_chain_dict by sequence length
    for k, v in h3_chain_dict.items():
        seq = v
        seq_len = len(seq)
        if seq_len not in h3_len_sorted_dict:
            h3_len_sorted_dict[seq_len] = []
        h3_len_sorted_dict[seq_len].append((k, seq))

    # Sort l1_chain_dict by sequence length
    for k, v in l1_chain_dict.items():
        seq = v
        seq_len = len(seq)
        if seq_len not in l1_len_sorted_dict:
            l1_len_sorted_dict[seq_len] = []
        l1_len_sorted_dict[seq_len].append((k, seq))

    # Sort l2_chain_dict by sequence length
    for k, v in l2_chain_dict.items():
        seq = v
        seq_len = len(seq)
        if seq_len not in l2_len_sorted_dict:
            l2_len_sorted_dict[seq_len] = []
        l2_len_sorted_dict[seq_len].append((k, seq))

    # Sort l3_chain_dict by sequence length
    for k, v in l3_chain_dict.items():
        seq = v
        seq_len = len(seq)
        if seq_len not in l3_len_sorted_dict:
            l3_len_sorted_dict[seq_len] = []
        l3_len_sorted_dict[seq_len].append((k, seq))
    
    print('h1_len_sorted_dict',h1_len_sorted_dict)

    def process_chain_length_dict(len_sorted_dict, chain_label, setids_list, valid_setis_list, save_path=None, save_fasta=False):
        """
        Process a length-sorted dict for a chain type (heavy, light, combined).
        Returns: (len_train_valid_counts, len_train_valid_df)
        """
        len_train_valid_counts = {}
        len_train_valid_df = pd.DataFrame(columns=['seq_len', 'train_count', 'valid_count', 'train_ids', 'valid_ids'])
        for seq_len, entries in len_sorted_dict.items():
            print(seq_len, len(entries))
            for entry in entries:
                seq = entry[-1]
                header = f'>{entry[0]}|{chain_label}|'
                # Optionally write to fasta if save_path is provided

                if save_fasta:
                    if save_path is not None:
                        fasta_filename = os.path.join(save_path, f'{chain_label.lower()}_chain_len_{seq_len}.fasta')
                        with open(fasta_filename, 'a') as fasta_file:
                            fasta_file.write(header + '\n')
                            fasta_file.write(seq + '\n')
                        
            train_count = 0
            valid_count = 0
            train_ids = []
            valid_ids = []
            for entry in entries:
                pdb_id = entry[0]
                if pdb_id in setids_list:
                    train_count += 1
                    train_ids.append(pdb_id)
                elif pdb_id in valid_setis_list:
                    valid_count += 1
                    valid_ids.append(pdb_id)
            len_train_valid_counts[seq_len] = {
                'train_count': train_count,
                'valid_count': valid_count,
                'train_ids': train_ids,
                'valid_ids': valid_ids
            }
            len_train_valid_df = pd.concat([
                len_train_valid_df,
                pd.DataFrame([{
                    'seq_len': seq_len,
                    'train_count': train_count,
                    'valid_count': valid_count,
                    'train_ids': train_ids,
                    'valid_ids': valid_ids
                }])
            ], ignore_index=True)
        # print(f"{chain_label} chain length train/valid counts:", len_train_valid_counts)
        # print(f'{chain_label.lower()}_len_train_valid_df\n', len_train_valid_df)
        return len_train_valid_counts, len_train_valid_df


    h1_len_train_valid_counts, h1_len_train_valid_df = process_chain_length_dict(
        h1_len_sorted_dict, "H1", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    )
    h2_len_train_valid_counts, h2_len_train_valid_df = process_chain_length_dict(
        h2_len_sorted_dict, "H2", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    )
    h3_len_train_valid_counts, h3_len_train_valid_df = process_chain_length_dict(
        h3_len_sorted_dict, "H3", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    )
    l1_len_train_valid_counts, l1_len_train_valid_df = process_chain_length_dict(
        l1_len_sorted_dict, "L1", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    )
    l2_len_train_valid_counts, l2_len_train_valid_df = process_chain_length_dict(
        l2_len_sorted_dict, "L2", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    )
    l3_len_train_valid_counts, l3_len_train_valid_df = process_chain_length_dict(
        l3_len_sorted_dict, "L3", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    )
    # light_len_train_valid_counts, light_len_train_valid_df = process_chain_length_dict(
    #     len_sorted_light_dict, "Light", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    # )
    # combined_len_train_valid_counts, combined_len_train_valid_df = process_chain_length_dict(
    #     len_sorted_combined_dict, "Combined", train_setids_list, valid_setis_list, save_path=save_path, save_fasta=False
    # )


    # print("h1_len_train_valid_counts\n",h1_len_train_valid_counts)
    print("h1_len_train_valid_df\n",h1_len_train_valid_df)
    print("h2_len_train_valid_df\n",h2_len_train_valid_df)
    print("h3_len_train_valid_df\n",h3_len_train_valid_df)

    print('')
    print("l1_len_train_valid_df\n",l1_len_train_valid_df)
    print("l2_len_train_valid_df\n",l2_len_train_valid_df)
    print("l3_len_train_valid_df\n",l3_len_train_valid_df)

if __name__=="__main__":

    base_dir = os.path.join(os.path.expanduser('~'), 'Documents', )
    
    trainset = os.path.join(base_dir, 'trainset_HM0_only_examples_dataframe.csv')
    validset = os.path.join(base_dir, 'validset_detail.csv')

    html_file_path = os.path.join(base_dir, 'AntibodyDocking', 'data', 'dataset_sequences', 'sabdab_cdr_search', 'SAbDab_The_Structural_Antibody_Database.html')
    antibody_fasta_seq = os.path.join(base_dir, 'AntibodyDocking', 'data', 'dataset_sequences', 'dataset_fastas', 'train_valid_antibody_only.fasta')
    
    save_path = os.path.join(base_dir, 'AntibodyDocking', 'data', 'dataset_sequences', 'dataset_fastas')

    train_dataset_dict = gather_dataset_ids(trainset)
    valid_dataset_dict = gather_dataset_ids(validset)

    """
    parse_html function: Parses SAbDab HTML to collect CDR sequences for all PDB IDs.

    Returns:
        1. Dict of all CDR loops concatenated together for each chain type (e.g., H1-H3 concatenated, L1-L3 concatenated).
        2. Dict of separate CDR loops for each chain (e.g., H1, H2, H3, L1, L2, L3 as separate entries).
    """
    
    pdb2cdr_dict, separate_cdrs_dict = parse_html(html_file_path, concat_all_cdrs=False)

    # heavy_chain_dict, light_chain_dict, H_L_combined_dict = match_allloops_cdr2fasta(pdb2cdr_dict, antibody_fasta_seq)

    h1_chain_dict, h2_chain_dict, h3_chain_dict, l1_chain_dict, l2_chain_dict, l3_chain_dict = match_perloop_cdr2fasta(separate_cdrs_dict, antibody_fasta_seq)

    loops_cdr_seqby_lens(h1_chain_dict, h2_chain_dict, h3_chain_dict, l1_chain_dict, l2_chain_dict, l3_chain_dict, train_dataset_dict,  valid_dataset_dict, save_path)

    # convert_dict2fasta(heavy_chain_dict, light_chain_dict, H_L_combined_dict, save_path)

    # allconcat_cdr_seq_lens(heavy_chain_dict, light_chain_dict, H_L_combined_dict, train_dataset_dict,  valid_dataset_dict, save_path)