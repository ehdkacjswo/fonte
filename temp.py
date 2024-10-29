from scipy import sparse

if __name__ == "__main__":
    arr = sparse.load_npz('/root/workspace/data/Defects4J/diff/Cli/0a8de54ff89093fc8c5a2b00f7c0c856c5cbe57d/encode/simple_encode_res.npz')
    row, col = arr.nonzero()
    for _row, _col in zip(row, col):
        print(_row, _col)

    print(arr.data)
    print(arr)