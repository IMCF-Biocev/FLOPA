import numpy as np
from .file import Header

class TTTRReader:
    def __init__(self, path):
        self.path = path
        self._header = Header(path)  # parse ONCE
        self.offset = self._header.data_offset
        self.bits = self._header.bits_per_record
        self.dtype = np.uint32 if self.bits == 32 else np.uint64

    @property
    def header(self):
        return self._header

    def read(self, count=None):
        with open(self.path, 'rb') as f:
            f.seek(self.offset)
            return np.fromfile(f, dtype=self.dtype, count=count)

    def iter_chunks(self, chunk_size=1000000):
        with open(self.path, 'rb') as f:
            f.seek(self.offset)
            while True:
                chunk = np.fromfile(f, dtype=self.dtype, count=chunk_size)
                if len(chunk) == 0:
                    break
                yield chunk
