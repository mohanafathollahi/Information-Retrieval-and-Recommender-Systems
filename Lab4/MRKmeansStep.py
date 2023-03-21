"""
.. module:: MRKmeansDef

MRKmeansDef
*************

:Description: MRKmeansDef



:Authors: bejar


:Version:

:Created on: 17/07/2017 7:42

"""

from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np

__author__ = 'bejar'


class MRKmeansStep(MRJob):
    prototypes = {}

    def jaccard(self, prot, doc):
        """
        Compute here the Jaccard similarity between  a prototype and a document
        prot should be a list of pairs (word, probability)
        doc should be a list of words
        Words must be alphabeticaly ordered

        The result should be always a value in the range [0,1]
        """

        # As words are alphabeticaly sorted we can sequentially
        # scan the lists only once

        document_list = []
        prototype_list = []

        # Indexes of the scan
        d = 0
        p = 0

        while (d < len(doc) and p < len(prot)):
            # Word in both document and prototype
            if doc[d] == prot[p][0]:
                document_list.append(1)
                prototype_list.append(prot[p][1])
                d += 1
                p += 1
            # Word in the document but not in the prototype
            elif doc[d] < prot[p][0]:
                document_list.append(1)
                prototype_list.append(0)
                d += 1
            # Word in the prototype but not in the document
            elif doc[d] > prot[p][0]:
                document_list.append(0)
                prototype_list.append(prot[p][1])
                p += 1

        # Input the missing part of the scan
        if (p < len(prot)):
            document_list.extend([0 for _ in prot[p:]])
            prototype_list.extend([probability for word, probability in prot[p:]])
        elif (d < len(doc)):
            document_list.extend([1 for _ in doc[d:]])
            prototype_list.extend([0 for _ in doc[d:]])

        document_list = np.array(document_list)
        prototype_list = np.array(prototype_list)

        # Computing the jaccard similarity coefficient
        jaccard_similarity = np.dot(document_list, prototype_list) / (sum(np.square(document_list)) + sum(np.square(prototype_list)) - np.dot(document_list, prototype_list))

        return jaccard_similarity

    def configure_args(self):
        """
        Additional configuration flag to get the prototypes files

        :return:
        """
        super(MRKmeansStep, self).configure_args()
        self.add_file_arg('--prot')

    def load_data(self):
        """
        Loads the current cluster prototypes

        :return:
        """
        f = open(self.options.prot, 'r')
        for line in f:
            cluster, words = line.split(':')
            cp = []
            for word in words.split():
                cp.append((word.split('+')[0], float(word.split('+')[1])))
            self.prototypes[cluster] = cp

    def assign_prototype(self, _, line):
        """
        This is the mapper it should compute the closest prototype to a document

        Words should be sorted alphabetically in the prototypes and the documents

        This function has to return at list of pairs (prototype_id, document words)

        You can add also more elements to the value element, for example the document_id
        """

        # Each line is a string docid:wor1 word2 ... wordn
        doc, words = line.split(':')
        lwords = words.split()

        # Sorting words alphabeticaly
        lwords.sort()

        best_similarity_id = None
        best_similarity = None

        for p, prototype in self.prototypes.items():
            jaccard_similarity = self.jaccard(prototype, lwords)
            if (best_similarity is None) or (jaccard_similarity > best_similarity):
                best_similarity = jaccard_similarity
                best_similarity_id = p
            

        # Return pair key, value
        yield best_similarity_id, (doc, lwords)

    def aggregate_prototype(self, key, values):
        """
        input is cluster and all the documents it has assigned
        Outputs should be at least a pair (cluster, new prototype)

        It should receive a list with all the words of the documents assigned for a cluster

        The value for each word has to be the frequency of the word divided by the number
        of documents assigned to the cluster

        Words are ordered alphabetically but you will have to use an efficient structure to
        compute the frequency of each word

        :param key:
        :param values:
        :return:
        """
        # Checking the new prototype
        counts = dict()
        docids = []

        for docid, words in values:
            docids.append(docid)
            for word in words:
                counts[word] = counts.get(word, 0) + 1

        # Average word counts for the kmeans algorithm
        new_prototype = [(word, count/len(docids)) for word, count in counts.items()]

        yield key, (new_prototype, docids)

    def steps(self):
        return [MRStep(mapper_init=self.load_data, mapper=self.assign_prototype,
                       reducer=self.aggregate_prototype)
            ]


if __name__ == '__main__':
    MRKmeansStep.run()
