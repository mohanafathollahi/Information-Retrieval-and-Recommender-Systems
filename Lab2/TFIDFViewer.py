import argparse

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.client import CatClient
from elasticsearch.exceptions import NotFoundError
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q


def search_file_by_path(client, index, path):
    """
    Search for a file using its path

    :param path:
    :return:
    """
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()
    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id


def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())


def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = []
    for (t, fd), (_, df) in zip(file_tv, file_df):
        ### Own implementation
        # 1) compute tf(d,i). Use f(d,i) = file_td[t] = fd and max = max_freq
        tf = fd / max_freq

        # 2) compute idf(i). log2(D/df(i)). D=dcount. df(i) = file_df[t]=df)
        idf = np.log2(dcount / df)
        w = tf * idf
        tfidfw.append((t, w))
    return normalize(tfidfw)


def print_term_weight_vector(twv):
    """
    Prints the term vector and the correspondig weights
    :param twv:
    :return:
    """
    ### Own implementation
    for t, v in twv:
        print(f'({t}, {v:3.5f})')


def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    ### Own implementation
    # Allocate terms in a tuple, and weights in another
    # vt: vector of terms
    # vw: vector of weights
    vt, vw = zip(*tw)

    # Convert the zip iterator in a numpy array
    vw = np.array(vw)
    # Normalize the vector of weights
    # nvw: normalized vector of weights
    nvw = vw / np.sqrt(np.sum(vw ** 2))

    # Return the tuples back together in the form of the original vector
    return list(zip(vt, nvw))


def cosine_similarity(tw1, tw2):
    """
    Computes the cosine similarity between two weight vectors, terms are alphabetically ordered
    :param tw1:
    :param tw2:
    :return:
    """
    ### Own implementation
    # We initialize the lists containing the indexes of the vectors
    l, l1, l2 = 0, 0, 0
    # Iterate over each tuple in the list
    while l1 < len(tw1) and l2 < len(tw2):
        # If the term 1 is on a lower alphabetical order than increase index of vector 1
        if tw1[l1][0] < tw2[l2][0]:
            l1 += 1
        # If the term 1 is on a higher alphabetical order than increase index of vector 2
        elif tw1[l1][0] > tw2[l2][0]:
            l2 += 1
        # When the terms are equal, perform the scalar product and then increase both indices
        else:
            l += tw1[l1][1] * tw2[l2][1]
            l1 += 1
            l2 += 1

    return l


def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, required=True, help='Index to search')
    parser.add_argument('--files', default=None, required=True, nargs=2, help='Paths of the files to compare')
    parser.add_argument('--print', default=False, action='store_true', help='Print TFIDF vectors')

    args = parser.parse_args()
    index = args.index
    file1 = args.files[0]
    file2 = args.files[1]

    # client = Elasticsearch(timeout=1000,use_ssl=True,ssl_context=context, scheme = "https",http_auth = ("elastic", "h_9RfOqHmS-v5-ZV9Lk7"))
    client = Elasticsearch(timeout=1000)
    try:
        # Get the files ids
        file1_id = search_file_by_path(client, index, file1)
        file2_id = search_file_by_path(client, index, file2)

        # Compute the TF-IDF vectors
        file1_tw = toTFIDF(client, index, file1_id)
        file2_tw = toTFIDF(client, index, file2_id)

        if args.print:
            print(f'TFIDF FILE {file1}')
            print_term_weight_vector(file1_tw)
            print('---------------------')
            print(f'TFIDF FILE {file2}')
            print_term_weight_vector(file2_tw)
            print('---------------------')

        print(f"Similarity = {cosine_similarity(file1_tw, file2_tw):3.5f}")


    except NotFoundError:
        print(f'Index {index} does not exists')
