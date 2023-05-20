"""
    Auxillary class

"""

import csv

def load_data(filename):
    """TODO"""

    word1_list = []
    word2_list = []

    with open(filename, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if len(row) != 2:
                continue
            word1, word2 = row
            word1_list.append(word1.strip())
            word2_list.append(word2.strip())

    return word1_list, word2_list
