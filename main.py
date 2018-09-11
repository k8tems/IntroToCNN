import numpy as np


def flatten(x):
    # `The input and kernel is flattened in a "column first` manner
    return x.ravel(order='F')


def get_patch_crds(input_shape, patch_size):
    # Highly important for outer loop to iterate over rows and inner loop to iterate over cols
    return [(col, row)
            for col in range(input_shape[1] - patch_size[1] + 1)
            for row in range(input_shape[0] - patch_size[0] + 1)]


def im2col(x, patch_size):
    # Apply MATLAB style `im2col` expansion to `A`
    patch_crds = get_patch_crds(x.shape, patch_size)
    B = np.ndarray(shape=(patch_size[0] * patch_size[1], len(patch_crds)))

    for i, (col, row) in enumerate(patch_crds):
        patch = x[row:row + patch_size[0], col:col + patch_size[1]]
        B[:, i] = flatten(patch)

    return B


if __name__ == '__main__':
    A = np.array(
        [[1, 2, 3, 1],
         [4, 5, 6, 1],
         [7, 8, 9, 1]])

    f = np.array(
        [[1, 1],
         [1, 1]]
    )

    expected = [
        [1, 4, 2, 5, 3, 6],
        [4, 7, 5, 8, 6, 9],
        [2, 5, 3, 6, 1, 1],
        [5, 8, 6, 9, 1, 1],
    ]

    B = im2col(A, [2, 2])

    print(B)
    print(B.transpose())

    print(np.matmul(B.transpose(), flatten(f)))
