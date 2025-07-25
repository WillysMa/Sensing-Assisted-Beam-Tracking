import numpy as np
import pandas as pd
import os
current_dir = os.path.dirname(__file__) #get the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

data_root = parent_dir + '/dataset/scenario9/' 
csv_name = 'scenario9.csv'
csv_path = data_root + csv_name

in_len = 8
out_len = 8


all_data = pd.read_csv(csv_path)
# Modify paths for relevant columns
# for col in ['unit1_rgb', 'unit1_radar', 'unit1_pwr_60ghz']:
#     all_data[col] = all_data[col].apply(lambda x: x.replace('./unit1/', './dataset/scenario9/unit1/'))

all_seq_idx = all_data['seq_index'].unique()

all_seq_split = []

for i in all_seq_idx:
    tmp = all_data[all_data['seq_index'] == i] # belong to the same time stamp
    tmp = tmp[['unit1_rgb', 'unit1_pwr_60ghz', 'seq_index']] 
    # tmp = tmp[['unit1_rgb', 'unit1_radar', 'unit1_pwr_60ghz', 'seq_index']]   
    all_seq_split.append(tmp)

all_seqs = []
for seq in all_seq_split: # iterate over each sequence, i.e., a DataFrame
    start = 0
    while start+in_len+out_len < seq.shape[0]:
        image = seq['unit1_rgb'][start:start+in_len].tolist()
        # radar = seq['unit1_radar'][start:start + in_len].tolist()
        in_beam = seq['unit1_pwr_60ghz'][start:start+in_len].tolist()
        out_beam = seq['unit1_pwr_60ghz'][start+in_len:start+in_len+out_len].tolist()
        seq_idx = seq['seq_index'][0:1].tolist() # slices the first row of the seq_index column.
        all_seqs.append(image+in_beam+out_beam+seq_idx)
        # all_seqs.append(image+radar+in_beam+out_beam+seq_idx)
        start += 1

col_names = [
                f'camera{i}' for i in range(1, in_len + 1)] + \
            [f'beam{i}' for i in range(1, in_len + 1)] + \
            [f'future_beam{i}' for i in range(1, out_len + 1)] + \
            ['seq_index']

# col_names = [
#                 f'camera{i}' for i in range(1, in_len + 1)] + \
#             [f'radar{i}' for i in range(1, in_len + 1)] + \
#             [f'beam{i}' for i in range(1, in_len + 1)] + \
#             [f'future_beam{i}' for i in range(1, out_len + 1)] + \
#             ['seq_index']

all_seqs = pd.DataFrame(all_seqs, columns = col_names)


training_set_pct = 0.8
ind_select = int(training_set_pct*all_seq_idx.shape[0])
train_seq_idx = np.sort(all_seq_idx[:ind_select])
test_seq_idx = np.sort(all_seq_idx[ind_select:])

train_seqs = all_seqs[all_seqs['seq_index'].isin(train_seq_idx)]
test_seqs = all_seqs[all_seqs['seq_index'].isin(test_seq_idx)]

train_csv_name = 'train_seqs_'+str(out_len)+'.csv'  
test_csv_name = 'test_seqs_'+str(out_len)+'.csv'

train_seqs.to_csv(data_root + train_csv_name, index=False)
test_seqs.to_csv(data_root + test_csv_name, index=False)

print('done')
