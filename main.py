import numpy


def im2col(A, patch_size):
    num_patches = 6  # how do I compute this
    B = numpy.ndarray(shape=(num_patches, patch_size[0]*patch_size[1]))

    i = 0
    for col in range(A.shape[1] - patch_size[1] + 1):
        for row in range(A.shape[0] - patch_size[0] + 1):
            B[i] = A[row:row+patch_size[0], col:col+patch_size[1]].ravel(order='F')
            i += 1

    return B.transpose()


if __name__ == '__main__':
    A = numpy.array(
        [[1, 2, 3, 1],
         [4, 5, 6, 1],
         [7, 8, 9, 1]])

    f = numpy.array(
        [[1, 1],
         [1, 1]]
    )

    expected = [
        [1, 4, 2, 5, 3, 6],
        [4, 7, 5, 8, 6, 9],
        [2, 5, 3, 6, 1, 1],
        [5, 8, 6, 9, 1, 1],
    ]

    assert(numpy.array_equal(expected, im2col(A, [2, 2])))
