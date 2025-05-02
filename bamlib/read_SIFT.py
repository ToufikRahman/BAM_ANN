import sys
import numpy as np
# def ivecs_read(fname):
#     x = np.memmap(fname, dtype='uint8', mode='r')
#     d = x[:4].view('int32')[0]
#     data = x.reshape(-1, d + 1)[:, 1:]
#     vectors = data.tolist()
#     return vectors

# def fvecs_read(fname):
#     x = np.memmap(fname, dtype='uint8', mode='r')
#     d = x[:4].view('int32')[0]
#     data = x.reshape(-1, d + 1)[:, 1:]
#     vectors = data.tolist()
#     vectors = np.asarray(vectors, dtype=np.float32)
#     return vectors

import numpy as np

def fvecs_read(filename, bounds=None):
    # Open the file and count the number of descriptors
    with open(filename, 'rb') as fid:
        if fid is None:
            raise IOError(f'I/O error: Unable to open the file {filename}')

        # Read the vector size
        d = np.fromfile(fid, dtype=np.int32, count=1)[0]
        vecsizeof = 1 * 4 + d * 4

        # Get the number of vectors
        fid.seek(0, 2)
        a, bmax = 1, fid.tell() // vecsizeof
        b = bmax

        if bounds is not None:
            if len(bounds) == 1:
                b = bounds[0]
            elif len(bounds) == 2:
                a, b = bounds[0], bounds[1]

        assert a >= 1
        if b > bmax:
            b = bmax

        if b == 0 or b < a:
            return np.array([])

        # Compute the number of vectors that are really read and go to starting positions
        n = b - a + 1
        fid.seek((a - 1) * vecsizeof, 0)

        # Read n vectors
        v = np.fromfile(fid, dtype=np.float32, count=(d + 1) * n)
        v = v.reshape((d + 1, n), order='F')

        # Check if the first column (dimension of the vectors) is correct
        assert np.sum(v[0, 1:] == v[0, 0]) == n - 1
        v = v[1:, :]

    return v
