def dist_values_append(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)


def values_of(dictionary, keys):
    return [dictionary[k] for k in keys if k in dictionary]
