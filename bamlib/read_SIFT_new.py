import numpy as np

def fvecs_read(filename, bounds=None):
    """
    Reads a set of vectors stored in the fvec format (int + n * float).

    Parameters:
    filename (str): Path to the .fvec file.
    bounds (int or list): Optional. Specify the number of vectors to read 
                          or a range [a, b] where indices start from 1.

    Returns:
    np.ndarray: A 2D NumPy array where each column is a vector.
    """
    # Open the file in binary read mode
    try:
        with open(filename, 'rb') as f:
            # Read the size of a vector (dimension)
            d = np.fromfile(f, dtype=np.int32, count=1)[0]
            vec_size = 4 + d * 4  # int (4 bytes) + d floats (4 bytes each)

            # Calculate the total number of vectors in the file
            f.seek(0, 2)  # Move to the end of the file
            total_vectors = f.tell() // vec_size

            # Handle bounds (reading all, some, or a range)
            a, b = 1, total_vectors  # Default: read all vectors
            if bounds is not None:
                if isinstance(bounds, int):
                    b = min(bounds, total_vectors)
                elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    a, b = bounds
                    a = max(1, a)
                    b = min(b, total_vectors)

            # Ensure valid range
            if b < a:
                return np.array([])

            # Move to the start position of the first vector to read
            f.seek((a - 1) * vec_size, 0)

            # Read the vectors
            n = b - a + 1
            data = np.fromfile(f, dtype=np.float32, count=(d + 1) * n)
            data = data.reshape(n, d + 1).T  # Transpose to match MATLAB column layout

            # Check if all vector dimensions are consistent
            assert np.all(data[0, :] == data[0, 0]), "Inconsistent vector dimensions"

            # Return the vectors (excluding the first row with dimensions)
            return data[1:, :]

    except FileNotFoundError:
        raise IOError(f"I/O error: Unable to open the file {filename}")

# Example usage:
# v = fvecs_read('vectors.fvec')
# v = fvecs_read('vectors.fvec', 10)  # Read first 10 vectors
# v = fvecs_read('vectors.fvec', [5, 20])  # Read vectors from index 5 to 20
