def to_chunks(array, chunk_size):
    for index in range(0, len(array), chunk_size):
        yield array[index:index + chunk_size]


def column(array, index): return [item[index] for item in array]
