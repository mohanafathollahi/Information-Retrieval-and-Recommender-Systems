import time
import sys
import argparse
import numpy as np
import pandas as pd
import random

class Edge:
    def __init__(self, origin=None, index=None):
        __slots__ = 'origin', 'index'
        self.origin = origin
        self.weight = 1.0
        self.index = index

        def __repr__(self):
            return f"edge: {self.origin}\t{self.weight}"

class Airport:
    def __init__(self, iden=None, name=None, index=None):
        __slots__ = 'iden', 'name', 'index'
        self.code = iden
        self.name = name
        self.routes = []
        self.route_hash = {}
        self.outweight = 0.0
        self.index = index

    def add_edge(self, orig_code):
        if orig_code in self.route_hash:
            index = self.route_hash[orig_code]
            edge = self.routes[index]
            edge.weight += 1.0
        else:
            pos = airport_hash[orig_code].index
            edge = Edge(orig_code, pos)

            self.routes.append(edge)
            index = len(self.routes) - 1
            self.route_hash[orig_code] = index

    def __repr__(self):
        return f"{self.code}\t{self.index}\t{self.name}"


airport_list = []  # list of Airport
airport_hash = {}  # hash key IATA code -> Airport


def read_airports(fd):
    print("Reading Airport file from {0}".format(fd))
    with open(fd, "r") as f:
        cont = 0
        for line in f.readlines():
            a = Airport()
            try:
                temp = line.split(',')
                if len(temp[4]) != 5:
                    raise Exception('not an IATA code')
                a.name = temp[1][1:-1] + ", " + temp[3][1:-1]
                a.code = temp[4][1:-1]
                a.index = cont
            except Exception as inst:
                pass
            else:
                cont += 1
                airport_list.append(a)
                airport_hash[a.code] = a
                #return[airport_list, airport_hash]

    print("Airports with IATA code: {0}".format(cont))


def match_airport(code):
    if not (code in airport_hash):
        raise Exception("Airport not found.")
    idx = airport_hash[code].index
    ap_code = airport_list[idx]
    return ap_code


def read_routes(fd):
    print("Reading Routes file from {0}".format(fd))
    with open(fd, "r") as f:
        cont = 0
        for line in f.readlines():
            try:
                temp = line.split(',')
                if len(temp[2]) != 3 or len(temp[4]) != 3:
                    raise Exception('not an IATA code')

                c_origin = temp[2]
                c_destin = temp[4]

                origin_ap = match_airport(c_origin)
                destin_ap = match_airport(c_destin)

                destin_ap.add_edge(c_origin)
                origin_ap.outweight += 1.0

            except Exception as inst:
                pass
            else:
                cont += 1

    print("Routes of Airports with IATA code: {0}".format(cont))


def compute_pageranks(lam=0.85, threshold=1e-6):
    nodes = len(airport_list)
    L = lam
    aux = 1.0 / nodes
    P = [aux] * nodes
    TL = (1.0 - L) / nodes
    sink_nodes = len([a for a in airport_list if a.outweight == 0])     #2455 airports do not have outweight
    sink_value = (L / float(nodes)) * sink_nodes      #constant value for sink nodes
    convergence = False
    iters = 0

    while not convergence:
        Q = [0.0] * nodes
        for i in range(nodes):
            airport = airport_list[i]
            sigma = 0
            for edge in airport.routes:
                w = edge.weight
                out = airport_list[edge.index].outweight
                sigma += P[edge.index] * w / out
            Q[i] = L * sigma + TL + aux * sink_value
        aux = TL + aux * sink_value
        vals = [abs(a - b) for a, b in zip(P, Q)]
        convergence = all([v < threshold for v in vals])
        P = Q
        iters += 1
    return [P, iters, sum(P)]


def outputPageRanks(pr):
    pagerank_list = {key: p for key, p in zip(range(len(pr)), pr)}
    sortedpr = sorted(pagerank_list.items(), key=lambda item: item[1], reverse=True)
    index = 1
    for  airport, pagerank in dict(sortedpr).items():
        print(f'{index}\t{pagerank:.10f}\t{airport_list[airport].name}')
        index += 1


def main():
    Lambda = 0.85
    threshold = 1e-7
    read_airports("airports.txt")
    read_routes("routes.txt")
    time1 = time.time()
    pageranks, iterations, sumpagerank = compute_pageranks(Lambda, threshold)
    time2 = time.time()
    outputPageRanks(pageranks)
    print(f'\n#Iterations: {iterations}\tLambda: {Lambda}\tThreshold: {threshold}')
    print(f'Time of compute_pageranks(): {time2-time1:.3f}s')
    print(f'sum of weights: {sumpagerank}')


if __name__ == "__main__":
    sys.exit(main())