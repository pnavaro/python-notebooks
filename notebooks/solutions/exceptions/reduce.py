def reduce(word_sorted_list):
    """ Count the number of occurences of a word in list
    and return a sorted dictionary """
    result = {}
    for word in word_sorted_list:
	  try:
            result[word] += 1
        except KeyError:
            result[word] = 1

    return sorted(result.items(), key=lambda v:v[1], reverse=True)
