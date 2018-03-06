import numpy as np
from theano import tensor as T
from src.utils import Vocab

__all__ = [
    'mask_probs',
    'tensor_gather_helper',
    'process_beam_results'
]

_FLOAT32_INF = np.float32(np.finfo('float32').max / 10)

def mask_probs(scores, beam_mask):
    """

    :param beam_mask: [batch_size ,beam_size]
        Beam mask which indicates the state of beam. 1.0 means opened beam and vice versa.

    :param scores: [batch_size , beam_size, vocab_size]
    """
    vocab_size = scores.shape[-1]

    finished_row = T.alloc(np.float32(1.0),  vocab_size) * _FLOAT32_INF

    finished_row = T.set_subtensor(finished_row[Vocab.EOS], 0.0) # [vocab_size, ]

    return scores + T.dot((1.0 - beam_mask)[:,:, None], finished_row[None,:])

def tensor_gather_helper(gather_indices, gather_from, batch_size, range_size, gather_shape):
    """
    :param gather_from: [batch_size, range_size, ...]

    :param gather_indices: [batch_size, beam_size]
    """

    range_ = (T.arange(batch_size) * range_size)[:, None] # [batch_size, 1]
    gather_indices_ = (gather_indices + range_).flatten() # [batch_size * range_size]

    output = T.take(gather_from.reshape(gather_shape), gather_indices_, axis=0)

    final_shape = gather_from.shape[:1 + len(gather_shape)]
    output = output.reshape(final_shape)

    return output

def process_beam_results(final_word_ids, final_beam_ids, final_scores):

    """
    :param final_word_ids: [max_seq_len, beam_size]

    :param final_beam_ids: [beam_size, ]
    """

    gathered_word_ids = np.zeros_like(final_beam_ids)

    for idx in range(final_beam_ids.shape[0]):
        gathered_word_ids = gathered_word_ids[:, final_beam_ids[idx]]
        gathered_word_ids[idx, :] = final_word_ids[idx]

    gathered_word_ids = gathered_word_ids.transpose() # [beam_size, seq_len]

    seq_lengths = np.zeros_like(gathered_word_ids).astype('float32')
    seq_lengths[gathered_word_ids != Vocab.EOS] = 1.0
    seq_lengths = np.sum(seq_lengths, axis=1) # [beam_size, ]

    normalized_scores = final_scores / seq_lengths

    reranked_word_ids = np.zeros_like(gathered_word_ids)

    for ii in np.argsort(normalized_scores):
        reranked_word_ids[ii] = gathered_word_ids[ii]

    return reranked_word_ids
