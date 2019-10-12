import string


def remove_pre_post_fix(text, prefix='$', postfix='#'):
    return text.replace(prefix, '').replace(postfix, '').strip()


def clean_punctuation(phase):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    # tokenize
    words = phase.split()
    # convert to lower case

    words = [word.lower() for word in words]
    # remove punctuation from each token
    words = [word.translate(table) for word in words]
    # remove hanging 's' and 'a'
    words = [word for word in words if len(word) > 1]
    # remove tokens with numbers in them
    words = [word for word in words if word.isalpha()]
    # store as string
    return ' '.join(words)


def word_to_index_and_index_to_word(vocabulary):
    index_to_word = {}
    word_to_index = {}

    for index, word in enumerate(vocabulary):
        word_to_index[word] = index
        index_to_word[index] = word

    return word_to_index, index_to_word
