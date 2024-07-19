import os
import torch

import kaldi_io
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def read_xvec(file):
    return kaldi_io.read_vec_flt(file)

def change_base_paths(files, new_base_path='./'):
    filenames = [os.path.basename(file) for file in files]
    new_filenames = [os.path.join(new_base_path, file) for file in filenames]
    return np.array(new_filenames)

def load_n_col(file, numpy=False):
    data = []
    with open(file) as fp:
        for line in fp:
            data.append(line.strip().split())
    columns = list(zip(*data))
    if numpy:
        columns = [np.array(list(i)) for i in columns]
    else:
        columns = [list(i) for i in columns]
    return columns

def calc_overlap(segment, ref_segment):
    '''
    calculates magnitude of overlap

    segment format: [start, end]
    '''
    return max(0.0, min(segment[1], ref_segment[1]) - max(segment[0], ref_segment[0]))

def assign_overlaps(events0, events1, events1_labels):
    events0_labels = []
    for event in events0:
        ols = []
        ols_t = []
        for i, evcheck in enumerate(events1):
            overlap = calc_overlap(event, evcheck)
            if overlap > 0.0:
                ols.append(events1_labels[i])
                ols_t.append(overlap)
        if len(ols) == 1:
            events0_labels.append(ols[0])
        if len(ols) == 0:
            events0_labels.append(None)
        if len(ols) > 1:
            events0_labels.append(ols[np.argmax(ols_t)])
    assert len(events0) == len(events0_labels)
    return events0_labels

def segment_labels(segments, rttm, xvectorscp, xvecbase_path=None):
    segment_cols = load_n_col(segments, numpy=True)
    segment_rows = np.array(list(zip(*segment_cols)))
    rttm_cols = load_n_col(rttm, numpy=True)
    vec_utts, vec_paths = load_n_col(xvectorscp, numpy=True)
    if not xvecbase_path:
        xvecbase_path = os.path.dirname(xvectorscp)
    assert sum(vec_utts == segment_cols[0]) == len(segment_cols[0])
    vec_paths = change_base_paths(vec_paths, new_base_path=xvecbase_path)

    rttm_cols.append(rttm_cols[3].astype(float) + rttm_cols[4].astype(float))
    recording_ids = sorted(set(segment_cols[1]))
    events0 = np.array(segment_cols[2:4]).astype(float).transpose()
    events1 = np.vstack([rttm_cols[3].astype(float), rttm_cols[-1]]).transpose()

    rec_batches = []

    for rec_id in tqdm(recording_ids):
        seg_indexes = segment_cols[1] == rec_id
        rttm_indexes = rttm_cols[1] == rec_id
        ev0 = events0[seg_indexes]
        ev1 = events1[rttm_indexes]
        ev1_labels = rttm_cols[7][rttm_indexes]
        ev0_labels = assign_overlaps(ev0, ev1, ev1_labels)
        ev0_labels = ['{}_{}'.format(rec_id, l) for l in ev0_labels]
        batch = (segment_cols[0][seg_indexes], ev0_labels, vec_paths[seg_indexes], segment_rows[seg_indexes])
        rec_batches.append(batch)

    return recording_ids, rec_batches


def pairwise_cat_matrix(xvecs, labels):
    '''
    xvecs: (seq_len, d_xvec)
    labels: (seq_len)
    '''
    xvecs = np.array(xvecs)
    seq_len, d_xvec = xvecs.shape
    xproject = np.tile(xvecs, seq_len).reshape(seq_len, seq_len, d_xvec)
    yproject = np.swapaxes(xproject, 0, 1)
    matrix = np.concatenate([xproject, yproject], axis=-1)
    label_matrix = sim_matrix_target(labels)
    return np.array(matrix), label_matrix

def sim_matrix_target(labels):
    le = LabelEncoder()
    dist = 1.0 - pairwise_distances(le.fit_transform(labels)[:,np.newaxis], metric='hamming')
    return dist

def make_k_fold_dataset(rec_ids, rec_batches, base_path, k=5):
    p = np.random.choice(np.arange(len(rec_ids)), len(rec_ids), replace=False)
    rec_ids = np.array(rec_ids)
    # issue : /mntnfs/lee_data1/maduo/codebase/FunASR/egs/alimeeting/diarization/sond/local/nn-similarity-diarization/data_io.py:112:
    # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences
    # (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated.
    # If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
    # rec_batches = np.array(rec_batches)
    #rec_batches = np.array(rec_batches)

    # information: >>> import numpy as np
    # >>> np.__version__
    #'1.23.5'
    rec_batches = np.array(rec_batches,dtype=object)
    splits = np.array_split(p, k)
    print('Making splits...')
    for i, te in enumerate(tqdm(splits)):
        fold_path = os.path.join(base_path, 'ch{}'.format(i))
        train_path = os.path.join(fold_path, 'train')
        test_path = os.path.join(fold_path, 'test')
        os.makedirs(fold_path, exist_ok=True)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        tr = [i for i in p if i not in te]

        train_ids = rec_ids[tr]
        train_batches = rec_batches[tr]

        test_ids = rec_ids[te]
        test_batches = rec_batches[te]

        utts, paths, spkrs, seglines = get_subset_files(test_ids, test_batches)
        make_files(test_path, utts, paths, spkrs, seglines)

        utts, paths, spkrs, seglines = get_subset_files(train_ids, train_batches)
        make_files(train_path, utts, paths, spkrs, seglines)


def get_subset_files(rec_ids, rec_batches):
    xvec_utts = []
    xvec_paths = []
    xvec_spk = []
    seglines = []
    for rec_id, batch in zip(rec_ids, rec_batches):
        xvec_paths.append(batch[2])
        xvec_utts.append(batch[0])
        xvec_spk.append(batch[1])
        seglines.append(batch[3])
    return np.concatenate(xvec_utts), np.concatenate(xvec_paths), np.concatenate(xvec_spk), np.concatenate(seglines)


def make_files(data_path, utts, paths, spkrs, seglines):
    os.makedirs(data_path, exist_ok=True)
    utt2spk = os.path.join(data_path, 'utt2spk')
    xvecscp = os.path.join(data_path, 'xvector.scp')
    segments = os.path.join(data_path, 'segments')
    with open(segments, 'w+') as fp:
        for l in seglines:
            line = ' '.join(l) + '\n'
            fp.write(line)
    with open(utt2spk, "w+") as fp:
        for utt, spk in zip(utts, spkrs):
            line = '{} {}\n'.format(utt, spk)
            fp.write(line)
    with open(xvecscp, "w+") as fp:
        for utt, path in zip(utts, paths):
            line = '{} {}\n'.format(utt, path)
            fp.write(line)

def make_subset_rttm(fullref_rttm, segments, rttm_outfile):
    segment_cols = load_n_col(segments, numpy=True)
    rttm_cols = load_n_col(fullref_rttm, numpy=True)
    recording_ids = list(set(segment_cols[1]))
    with open(fullref_rttm) as fp:
        with open(rttm_outfile, 'w+') as wp:
            for line in fp:
                su = line.strip().split()
                if su[1] in recording_ids:
                    wp.write(line)

def recombine_matrix(submatrices):
    dim = int(np.sqrt(len(submatrices)))
    rows = []
    for j in range(dim):
        start = j * dim
        row = np.concatenate(submatrices[start:start+dim], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)

def collate_sim_matrices(out_list, rec_ids):
    '''
    expect input list
    '''
    comb_matrices = []
    comb_ids = []
    matrix_buffer = []
    last_rec_id = rec_ids[0]
    for rid, vec in zip(rec_ids, out_list):
        if last_rec_id == rid:
            matrix_buffer.append(vec)
        else:
            if len(matrix_buffer) > 1:
                comb_matrices.append(recombine_matrix(matrix_buffer))
            else:
                comb_matrices.append(matrix_buffer[0])
            comb_ids.append(last_rec_id)
            matrix_buffer = [vec]
        last_rec_id = rid
    if len(matrix_buffer) > 1:
        comb_matrices.append(recombine_matrix(matrix_buffer))
    else:
        comb_matrices.append(matrix_buffer[0])
    comb_ids.append(last_rec_id)
    return comb_matrices, comb_ids

def batch_matrix(xvecpairs, labels, factor=2):
    baselen = len(labels)//factor
    split_batch = []
    split_batch_labs = []
    for j in range(factor):
        for i in range(factor):
            start_j = j * baselen
            end_j = (j+1) * baselen if j != factor - 1 else None
            start_i = i * baselen
            end_i = (i+1) * baselen if i != factor - 1 else None

            mini_pairs = xvecpairs[start_j:end_j, start_i:end_i, :]
            mini_labels = labels[start_j:end_j, start_i:end_i]

            split_batch.append(mini_pairs)
            split_batch_labs.append(mini_labels)
    return split_batch, split_batch_labs


def group_recs(utt2spk, segments, xvecscp):
    utts, labels = load_n_col(utt2spk, numpy=True)
    uspkdict = {k:v for k,v in zip(utts, labels)}
    xutts, xpaths = load_n_col(xvecscp, numpy=True)
    xdict = {k:v for k,v in zip(xutts, xpaths)}
    sutts, srecs, _, _ = load_n_col(segments, numpy=True)
    rec_ids = sorted(list(set(srecs)))

    rec_batches = []
    for i in rec_ids:
        rutts = sutts[srecs == i]
        rlabs = [uspkdict[u] for u in rutts]
        rpaths = [xdict[u] for u in rutts]
        rec_batches.append([rlabs, rpaths])
    return rec_ids, rec_batches

class dloader:

    def __init__(self, data_path, max_len=400, xvecbase_path=None, shuffle=True):
        utt2spk = os.path.join(data_path, 'utt2spk')
        segments = os.path.join(data_path, 'segments')
        xvecscp = os.path.join(data_path, 'xvector.scp')
        self.ids, self.rec_batches = group_recs(utt2spk, segments, xvecscp)
        self.lengths = np.array([len(batch[0]) for batch in self.rec_batches])
        self.factors = np.ceil(self.lengths/max_len).astype(int)
        self.first_rec = np.argmax(self.lengths)
        self.max_len = max_len
        self.shuffle = shuffle

    def __len__(self):
        return np.sum(self.factors**2)

    def get_batches(self):
        rec_order = np.arange(len(self.rec_batches))
        if self.shuffle:
            np.random.shuffle(rec_order)
            first_rec = np.argwhere(rec_order == self.first_rec).flatten()
            rec_order[0], rec_order[first_rec] = rec_order[first_rec], rec_order[0]

        for i in rec_order:
            rec_id = self.ids[i]
            labels, paths = self.rec_batches[i]
            xvecs = np.array([read_xvec(file) for file in paths])
            pmatrix, plabels = pairwise_cat_matrix(xvecs, labels)
            if len(labels) <= self.max_len:
                yield pmatrix, plabels, rec_id
            else:
                factor = np.ceil(len(labels)/self.max_len).astype(int)
                batched_feats, batched_labels = batch_matrix(pmatrix, plabels, factor=factor)
                for feats, labels in zip(batched_feats, batched_labels):
                    yield feats, labels, rec_id


    def get_batches_seq(self):
        rec_order = np.arange(len(self.rec_batches))
        if self.shuffle:
            np.random.shuffle(rec_order)
            first_rec = np.argwhere(rec_order == self.first_rec).flatten()
            rec_order[0], rec_order[first_rec] = rec_order[first_rec], rec_order[0]
        for i in rec_order:
            rec_id = self.ids[i]
            labels, paths = self.rec_batches[i]
            xvecs = np.array([read_xvec(file) for file in paths])
            pwise_labels = sim_matrix_target(labels)
            yield xvecs, pwise_labels, rec_id


