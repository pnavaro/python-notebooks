def words(textfile):
    """ Parse a file and returns a sorted list of words """
    with open(textfile) as file:
        read_text = file.read()

    return sorted(read_text.lower().replace('.', '').split())


def reduce(word_sorted_list):
    """ Count the number of occurences of a word in list
    and return a dictionary """
    current_word = None
    result = {}
    for word in word_sorted_list:
        if current_word is None:
            current_word = word
            result[word] = 0  # Add the first word in result

        # this if only works because words output is sorted
        if current_word == word:
            result[word] += 1
        else:
            current_word = word
            result[word] = 1

    return result


if __name__ == '__main__':
    from lorem import text

    with open('sample.txt', 'w') as f:
        f.write(text())

    words('sample.txt')
    print(reduce(words('sample.txt')))
