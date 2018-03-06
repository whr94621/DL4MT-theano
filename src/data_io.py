import numpy as np
from itertools import islice
import random
import tempfile
import os

from .bpe import Bpe
from .utils import Vocab, batch_open, INFO

__all__ = [
    'TextDataset',
    'ZipDatasets',
    'DataIterator'
]

class Dataset(object):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        raise NotImplementedError

    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def _data_iter(self):
        raise NotImplementedError

    def data_iter(self):
        return self._data_iter()

class TextDataset(Dataset):

    def __init__(self,
                 data_path,
                 vocab,
                 bpe_codes=None,
                 use_char=False,
                 max_len=-1
                 ):

        super(TextDataset, self).__init__()

        if bpe_codes is not None and use_char is True:
            raise ValueError("BPE and character tokenizer could not use simultaneously!")

        if not isinstance(vocab, Vocab):
            raise ValueError("vocab must be an instance of Vocab.")

        self._data_path = data_path
        self._vocab = vocab
        self._use_char = use_char
        self._max_len = max_len

        if bpe_codes is not None and len(bpe_codes) > 0:
            self._bpe = Bpe(codes=bpe_codes) # type: Bpe
        else:
            self._bpe = None

    def __len__(self):
        return 1

    def _data_iter(self):
        with open(self._data_path) as f:
            for line in f:
                yield self.apply(line)

    def apply(self, line):
        """
        Process one line

        :type line: str
        """
        line = line.strip().split()

        if self._bpe is not None:
            line = sum([self._bpe.segment_word(w) for w in line], [])

        if self._use_char is True:
            line = [list(w) for w in line]

        line = [self._vocab.token2id(w) for w in line]

        if self._max_len > 0 and len(line) > self._max_len:
            return None
        return line

class ZipDatasets(Dataset):

    def __init__(self, *datasets, shuffle=False):
        """
        """
        super(ZipDatasets, self).__init__()
        self.shuffle = shuffle
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets)


    def _data_iter(self):

        with batch_open([d._data_path for d in self.datasets]) as fs:
            for lines in zip(*fs):

                outs = [ds.apply(line) for ds, line in zip(self.datasets, lines)]

                yield outs

    def _shuffled_data_iter(self):

        f_handles = [open(ds._data_path) for ds in self.datasets]

        # Read all the data
        lines = []
        for l in f_handles[0]:
            line = [l.strip()] + [ff.readline().strip() for ff in f_handles[1:]]
            lines.append(line)

        # close file handles
        [f.close() for f in f_handles]

        # random shuffle the data
        INFO('Shuffling data...')
        random.shuffle(lines)
        INFO('Done.')

        # Set up temp files
        f_handles = []
        for ds in self.datasets:
            dirname, filename = os.path.split(ds._data_path)
            f_handles.append(tempfile.TemporaryFile(prefix=filename + '.shuf', dir=dirname, mode="a+"))

        for line in lines:
            for ii, f in enumerate(f_handles):
                print(line[ii], file=f)

        # release memory
        lines = []

        # Reset file handles
        [f.seek(0) for f in f_handles]

        for line in f_handles[0]:
            outs = [line] + [ff.readline() for ff in f_handles[1:]]
            outs = [ds.apply(line) for line, ds in zip(outs, self.datasets)]
            yield outs

    def data_iter(self):

        if self.shuffle is False:
            return self._data_iter()
        else:
            return self._shuffled_data_iter()


    def apply(self, *args, **kwargs):
        return


class DataIterator(object):

    def __init__(self, dataset, batch_size,
                 buffer_size=None, sort_buffer=True, sort_fn=None):
        """:
        :type dataset: Dataset
        """
        self.dataset = dataset
        self.batch_size = batch_size

        self.buffer_size = buffer_size if buffer_size is not None else 100 * batch_size


        if sort_buffer is True:
            assert sort_fn is not None

        self.sort_buffer=sort_buffer
        self.sort_fn = sort_fn

        self.reset()

    @property
    def n_datasets(self):
        return len(self.dataset)

    def _not_null_sample(self, samples):
        if self.n_datasets == 1:
            if samples is not None:
                return True
            else:
                return False
        else:
            if len([s for s in samples if s is None]) == 0:
                return True
            else:
                return False


    def _fill_buffer(self):
        batch = list(islice(self.data_iter, self.buffer_size))

        if len(batch) <= 0:
            # data_iter reach the end of the dataset
            self._end = True
            return

        batch = [sample for sample in batch if self._not_null_sample(sample) is True]

        if self.sort_buffer is True:
            # Customize buffer sorting algorithm
            scores = np.array([self.sort_fn(sample) for sample in batch])
            sorted_indices = np.argsort(scores).tolist()

            batch = [batch[i] for i in sorted_indices]
        else:
            # FIFO, so reverse the buffer
            batch.reverse()

        self.buffer = batch + self.buffer

    @property
    def is_end(self):
        return self._end

    def reset(self):
        self.buffer = []
        self.data_iter = self.dataset.data_iter()
        self._end = False


    def build_generator(self, batch_size=None):

        n_datasets = len(self.dataset)

        while True:

            batch = [[] for _ in range(n_datasets)]

            if len(self.buffer) == 0:
                self._fill_buffer()

            if len(self.buffer) == 0:
                """Reach the end of the dataset, exit.
                """
                self.reset()
                break

            while True:
                try:
                    samples = self.buffer.pop()
                except IndexError:
                    break

                if n_datasets == 1:

                    batch[0].append(samples)
                else:
                    for i in range(n_datasets):
                        batch[i].append(samples[i])

                if len(batch[0]) >= (batch_size if batch_size is not None else self.batch_size):
                    break

            if len(batch[0]) == 0:
                self.reset()
                break

            if n_datasets == 1:
                yield batch[0]
            else:
                yield batch