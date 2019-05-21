def to_chunks(array, chunk_size):
    for index in range(0, len(array), chunk_size):
        yield array[index:index + chunk_size]


def column(array, index): return [item[index] for item in array]


def args_max(array, top=1): return (-array).argsort()[:top]


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))