import networkx as nx
import numpy as np
import csv
import json
import os
import matplotlib.pyplot as plt
import pandas as pds
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns

from ..utilities import findFileInDirectory, computeProbabilityDistribution, plotProbabilityDistribution


class InteractionNetwork():
    _interactionsMatrix: np.matrix
    _thresholdPositiveValues: float
    _thresholdNegativeValues: float
    _nodeNames: list

    def __init__(self,
                 interactionsMatrix: np.matrix,
                 nodeNames: list,
                 thresholdValues=None,
                 thresholdMethod: str = "auto") -> None:
        self._interactionsMatrix = interactionsMatrix
        self._nodeNames = nodeNames
        self.updateTrasholdValue(thresholdValues, thresholdMethod)

    @property
    def nodeNames(self):
        return self._nodeNames

    @property
    def numberComponents(self):
        return len(self._nodeNames)

    @property
    def allInteractionsMatrix(self):
        return np.bitwise_or(self.positiveInteractionsMatrix,
                             self.negativeInteractionsMatrix)

    @property
    def positiveInteractionsMatrix(self):
        n = self.numberComponents
        s = np.concatenate([
            self._interactionsMatrix[:, j] > self._thresholdPositiveValues[j]
            for j in range(n)
        ],
                           axis=1)
        return s

    @property
    def negativeInteractionsMatrix(self):
        n = self.numberComponents
        s = np.concatenate([
            self._interactionsMatrix[:, j] < self._thresholdNegativeValues[j]
            for j in range(n)
        ],
                           axis=1)
        return s

    @property
    def interactionsMatrix(self):
        return self._interactionsMatrix

    @property
    def thresholdValues(self):
        return [self._thresholdNegativeValues, self._thresholdPositiveValues]

    @staticmethod
    def getTrasholdMethods():
        return [
            "global-half", "global-median", "global-mean", "single-half",
            "single-mean", "single-median", "single-half-2", "single-mean-2",
            "single-median-2"
        ]

    def copy(self):
        return InteractionNetwork(
            interactionsMatrix=self._interactionsMatrix.copy(),
            nodeNames=self._nodeNames.copy(),
            thresholdMethod="manual",
            thresholdValue=[
                self._thresholdNegativeValues, self._thresholdPositiveValues
            ])

    def updateTrasholdValue(self, value: list = None, method: str = "manual"):
        n = self.numberComponents

        if method == "auto":
            method = "global-half"

        if method == "manual":
            assert value is not None
            if isinstance(value, float):
                value = [[-value for _ in range(n)], [value for _ in range(n)]]
            elif isinstance(value, list):
                if isinstance(value[0], float) and isinstance(value[1], float):
                    value = [[value[0] for _ in range(n)],
                             [value[1] for _ in range(n)]]
            assert isinstance(value, list) and isinstance(
                value[0], list) and isinstance(value[1], list) and len(
                    value[0]) == len(value[1]) and len(value[0]) == n

            self._thresholdNegativeValues = value[0]
            self._thresholdPositiveValues = value[1]
            pass
        elif method == "global-half":
            vmax = self._interactionsMatrix.max()
            vmin = self._interactionsMatrix.min()

            self._thresholdNegativeValues = [vmin / 2 for _ in range(n)]
            self._thresholdPositiveValues = [vmax / 2 for _ in range(n)]
            pass
        elif method == "global-mean":
            l = np.reshape(np.array(self._interactionsMatrix), (n * n, ))
            psTh = np.mean(l[l > 0])
            ngTh = np.mean(l[l < 0])

            self._thresholdNegativeValues = [ngTh for _ in range(n)]
            self._thresholdPositiveValues = [psTh for _ in range(n)]
            pass
        elif method == "global-median":
            l = np.reshape(np.array(self._interactionsMatrix), (n * n, ))
            psTh = np.median(l[l > 0])
            ngTh = np.median(l[l < 0])

            self._thresholdNegativeValues = [ngTh for _ in range(n)]
            self._thresholdPositiveValues = [psTh for _ in range(n)]
            pass
        elif method == "single-half":
            l = np.array(self._interactionsMatrix)
            vmax = np.max(l, axis=1)
            vmin = np.min(l, axis=1)

            self._thresholdNegativeValues = (vmin / 2).tolist()
            self._thresholdPositiveValues = (vmax / 2).tolist()
            pass
        elif method == "single-mean":
            l = np.array(self._interactionsMatrix)
            psTh = [
                np.mean(l[l[:, i] > 0, i]) if np.any(l[:, i] > 0) else 0
                for i in range(n)
            ]
            ngTh = [
                np.mean(l[l[:, i] < 0, i]) if np.any(l[:, i] < 0) else 0
                for i in range(n)
            ]

            self._thresholdNegativeValues = ngTh
            self._thresholdPositiveValues = psTh
            pass
        elif method == "single-median":
            l = np.array(self._interactionsMatrix)
            psTh = [
                np.median(l[l[:, i] > 0, i]) if np.any(l[:, i] > 0) else 0
                for i in range(n)
            ]
            ngTh = [
                np.median(l[l[:, i] < 0, i]) if np.any(l[:, i] < 0) else 0
                for i in range(n)
            ]

            self._thresholdNegativeValues = ngTh
            self._thresholdPositiveValues = psTh
            pass
        elif method == "single-half-2":
            l = np.array(self._interactionsMatrix)
            vmax = np.max(l, axis=0)
            vmin = np.min(l, axis=0)

            self._thresholdNegativeValues = (vmin / 2).tolist()
            self._thresholdPositiveValues = (vmax / 2).tolist()
            pass
        elif method == "single-mean-2":
            l = np.array(self._interactionsMatrix)
            psTh = [
                np.mean(l[i, l[i, :] > 0]) if np.any(l[i, :] > 0) else 0
                for i in range(n)
            ]
            ngTh = [
                np.mean(l[i, l[i, :] < 0]) if np.any(l[i, :] < 0) else 0
                for i in range(n)
            ]

            self._thresholdNegativeValues = ngTh
            self._thresholdPositiveValues = psTh
            pass
        elif method == "single-median-2":
            l = np.array(self._interactionsMatrix)
            psTh = [
                np.median(l[i, l[i, :] > 0]) if np.any(l[i, :] > 0) else 0
                for i in range(n)
            ]
            ngTh = [
                np.median(l[i, l[i, :] < 0]) if np.any(l[i, :] < 0) else 0
                for i in range(n)
            ]

            self._thresholdNegativeValues = ngTh
            self._thresholdPositiveValues = psTh
            pass

    def getProbabilsticGraph(self):
        g = nx.Graph()
        n = self.numberComponents

        probsMatrix = self.getProbInteractionsMatrix()
        for r in range(n):
            for c in range(n):
                rn = self._nodeNames[r]
                cn = self._nodeNames[c]
                prob = probsMatrix[r, c]
                if g.has_edge(cn, rn):
                    g[cn][rn]["weight"] = max(g[cn][rn]["weight"], prob)
                else:
                    g.add_edge(rn, cn, weight=prob)
        return g

    def getGraph(self):
        g = nx.DiGraph()
        n = self.numberComponents

        positiveMatrix = self.positiveInteractionsMatrix
        for r in range(n):
            for c in range(n):
                if positiveMatrix[r, c]:
                    g.add_edge(self._nodeNames[r],
                               self._nodeNames[c],
                               type="pos",
                               color="r",
                               weight=self._interactionsMatrix[r, c])

        negativeMatrix = self.negativeInteractionsMatrix
        for r in range(n):
            for c in range(n):
                if negativeMatrix[r, c]:
                    g.add_edge(self._nodeNames[r],
                               self._nodeNames[c],
                               type="neg",
                               color="b",
                               weight=self._interactionsMatrix[r, c])
        return g

    def getIndexFromName(self, name):
        if isinstance(name, str):
            return self._nodeNames.index(name)
        elif isinstance(name, list):
            return [self._nodeNames.index(nn) for nn in name]
        return None

    def hasInteraction(self,
                       entities: list,
                       interationType: bool = False,
                       ignoreDirection: bool = False):
        if isinstance(entities, tuple):
            entities = [entities]

        if ignoreDirection:
            v1 = self.hasInteraction(entities=entities,
                                     interationType=interationType,
                                     ignoreDirection=False)
            v2 = self.hasInteraction(entities=[(y, x) for (x, y) in entities],
                                     interationType=interationType,
                                     ignoreDirection=False)

            if interationType:
                return [(item1 + item2 if abs(item1 + item2) != 2 else np.nan)
                        for (item1, item2) in zip(v1, v2)]
            else:
                return np.bitwise_and(v1, v2)

        else:
            if interationType:
                return [
                    1 if self._interactionsMatrix[x, y] >
                    self._thresholdPositiveValues[y] else
                    -1 if self._interactionsMatrix[x, y] <
                    self._thresholdNegativeValues[y] else 0
                    for (x, y) in entities
                ]
            else:
                return [
                    self._interactionsMatrix[x, y] <
                    self._thresholdNegativeValues[y]
                    or self._interactionsMatrix[x, y] >
                    self._thresholdPositiveValues[y] for (x, y) in entities
                ]

    def getNormInteractionsMatrix(self):
        vmatrix = self._interactionsMatrix
        n = vmatrix.shape[0]
        vmax = vmatrix.max()
        vmin = abs(vmatrix.min())

        resultMatrix = np.zeros(vmatrix.shape)
        for r in range(n):
            for c in range(n):
                v = vmatrix[r, c]
            resultMatrix[r, c] = (v / vmax) if v > 0 else (v / vmin)
        return resultMatrix

    def getProbInteractionsMatrix(self):

        smatrix = np.sign(self._interactionsMatrix)
        vmatrix = np.abs(self._interactionsMatrix)
        n = vmatrix.shape[0]
        th = np.median(np.abs(
            [self._thresholdPositiveValues, self._thresholdNegativeValues]),
                       axis=0)

        vmax = vmatrix.max()
        vmin = vmatrix.min()

        resultMatrix = np.zeros(vmatrix.shape)
        for c in range(n):
            thp = th[c]
            thn = th[c]
            #assert thp <= vmax and thn >= vmin and thn <= thp
            for r in range(n):
                v = vmatrix[r, c]

                if v > thp:
                    newv = 0.5 + 0.5 * (v - thp) / (vmax - thp)
                elif v <= thn:
                    newv = 0.5 + 0.5 * (v - thn) / (vmin - thn)
                else:
                    newv = v * (thn + thp - v) / (2 * thn * thp)
                resultMatrix[r, c] = 0 if np.isnan(newv) or newv < 0 else newv
        return resultMatrix

    def getProbInteractionsMatrix1(self):

        vmatrix = self._interactionsMatrix
        n = vmatrix.shape[0]
        thps = self._thresholdPositiveValues
        thns = self._thresholdNegativeValues
        vmax = vmatrix.max()
        vmin = vmatrix.min()

        resultMatrix = np.zeros(vmatrix.shape)
        for c in range(n):
            thp = thps[c]
            thn = thns[c]
            #assert thp <= vmax and thn >= vmin and thn <= thp
            for r in range(n):
                v = vmatrix[r, c]

                if v >= thp:
                    newv = 0.5 + 0.5 * (v - thp) / (vmax - thp)
                elif v <= thn:
                    newv = 0.5 + 0.5 * (v - thn) / (vmin - thn)
                else:
                    newv = v * (thn + thp - v) / (2 * thn * thp)
                resultMatrix[r, c] = 0 if np.isnan(newv) or newv < 0 else newv
        return resultMatrix

    def getProbabilities(self, entities: list):
        if isinstance(entities, tuple):
            entities = [entities]

        probMatrix = self.getProbInteractionsMatrix()
        return [probMatrix[x, y] for (x, y) in entities]

    def getInteractionScore(self, entities: list):
        if isinstance(entities, tuple):
            entities = [entities]

        return [self._interactionsMatrix[x, y] for (x, y) in entities]

    def getDegreeMatrix(self):
        return [self._interactionsMatrix * 180 / np.pi]


class CompareInteractionNetworkResult():
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float


class InteractionNetworkHelper():
    _network: InteractionNetwork

    def __init__(self, net: InteractionNetwork) -> None:
        self._network = net

    @property
    def network(self):
        return self._network

    def save(self, filename: str):
        self.saveNetwork(filename=filename, net=self._network)

    @staticmethod
    def getMatrixTypes():
        return ["interactionsMatrix","probabilistic","normalize"]

    def exportInteractionMatrix(self, filename: str, matrixType: str = None):
        net = self._network

        if matrixType is None or matrixType == "default" or matrixType == "interactionsMatrix":
            matrix = net.interactionsMatrix
        elif matrixType == "probabilistic":
            matrix = net.getProbInteractionsMatrix()
        elif matrixType == "normalize":
            matrix = net.getNormInteractionsMatrix()
        else:
            raise Exception("Not supported")

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([''] + net.nodeNames)
            for r in range(net.numberComponents):
                l = [net.nodeNames[r]]
                for c in range(net.numberComponents):
                    l.append(matrix[r, c])
                csvwriter.writerow(l)

    def export(self, filename: str, format: str = None):
        pass

    def compareWith(self,
                    network: InteractionNetwork,
                    nameMapping: dict = None,
                    resultFoldername: str = None):
        res, matrix = self.compareNetworks(source=self._network,
                                           target=network,
                                           nameMapping=nameMapping,
                                           resultFoldername=resultFoldername)
        return res, matrix

    def computeAUC(self, targetNet: InteractionNetwork):
        pred, target = self.compareSingleEdge(source=self._network,
                                              target=targetNet,
                                              useProbabilities=True)
        ls_auc = roc_auc_score(target, pred)
        return ls_auc

    @staticmethod
    def getPlotTypes():
        return ["interactions","generic-interactions", "probabilistic"]

    def saveAsPlot(self, filename: str, plotType: str = None):
        plt.close()
        f = plt.figure()

        def nudge(pos, x_shift, y_shift):
            return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}

        if plotType is None or plotType == "interactions":
            G = self._network.getGraph()
            pos = nx.circular_layout(G)
            pos_nodes = nudge(pos, 0, 0.1)
            edges = G.edges()
            colors = [G[u][v]['color'] for u, v in edges]

            pos_attrs = {}
            for node, coords in pos.items():
                pos_attrs[node] = (coords[0], coords[1] + 0.08)
            nx.draw(G, pos, edge_color=colors, with_labels=False)
            nx.draw_networkx_labels(G, pos=pos_nodes)
        elif plotType=="generic-interactions":
            G = self._network.getGraph()
            pos = nx.circular_layout(G)
            pos_nodes = nudge(pos, 0, 0.1)
            edges = G.edges()

            pos_attrs = {}
            for node, coords in pos.items():
                pos_attrs[node] = (coords[0], coords[1] + 0.08)
            nx.draw(G, pos, with_labels=False)
            nx.draw_networkx_labels(G, pos=pos_nodes)
        elif plotType == "probabilistic":
            G = self._network.getProbabilsticGraph()
            pos = nx.circular_layout(G)
            pos_nodes = nudge(pos, 0, 0.1)
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]

            #edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

            nx.draw(G,
                    pos,
                    edgelist=edges,
                    edge_color=weights,
                    width=3,
                    with_labels=False,
                    edge_cmap=plt.cm.Blues)
            nx.draw_networkx_labels(G, pos=pos_nodes)
        else:
            raise Exception("Not supported")

        f.savefig(filename)
        plt.close()

    @staticmethod
    def importNetwork(filename: str, sep='\t') -> InteractionNetwork:
        if os.path.isdir(filename):
            filename = findFileInDirectory(filename,
                                           ["gold", "network", "target"], 1)
        assert filename is not None

        datatxt = pds.read_csv(filename, sep=sep, header=None)
        c = datatxt.values.shape[0]
        nodes1: list = datatxt.values[:, 0].tolist()
        nodes2: list = datatxt.values[:, 1].tolist()
        arcs: list = datatxt.values[:, 2].tolist()

        labels = list(set((nodes1 + nodes2)))
        n = len(labels)
        matrix = np.zeros((n, n))
        for i in range(c):
            v = arcs[i]
            if v != 0:
                if v in ('+', 1):
                    v = np.pi / 2
                elif v in ('-', -1):
                    v = -np.pi / 2
                else:
                    raise Exception("File not supported")

                idx1 = labels.index(nodes1[i])
                idx2 = labels.index(nodes2[i])
                matrix[idx1, idx2] = v

        net = InteractionNetwork(interactionsMatrix=np.matrix(matrix),
                                 thresholdValues=0.0,
                                 thresholdMethod="manual",
                                 nodeNames=labels)

        return net

    @staticmethod
    def compareSingleEdge(source: InteractionNetwork,
                          target: InteractionNetwork,
                          nameMapping: dict = None,
                          useProbabilities=False):

        if nameMapping is None:
            nameMapping = {}

        sourceNodeArcs = []
        targetNodeArcs = []
        nodeNames = target.nodeNames
        for name in nodeNames:
            if not name in nameMapping.keys():
                nameMapping[name] = name

        for name1 in nodeNames:
            for name2 in nodeNames:
                targetNodeArcs.append((target.getIndexFromName(name1),
                                       target.getIndexFromName(name2)))
                sourceNodeArcs.append(
                    (source.getIndexFromName(nameMapping[name1]),
                     source.getIndexFromName(nameMapping[name2])))

        if not useProbabilities:
            sourceInteractions = source.hasInteraction(sourceNodeArcs)
        else:
            sourceInteractions = source.getProbabilities(sourceNodeArcs)
        targetInteractions = target.hasInteraction(targetNodeArcs)
        return sourceInteractions, targetInteractions

    @staticmethod
    def compareNetworks(source: InteractionNetwork,
                        target: InteractionNetwork,
                        nameMapping: dict = None,
                        resultFoldername: str = None):

        sourceInteractions, targetInteractions = InteractionNetworkHelper.compareSingleEdge(
            source=source, target=target, nameMapping=nameMapping)

        confusionMatrix = confusion_matrix(y_true=targetInteractions,
                                           y_pred=sourceInteractions)
        tn, fp, fn, tp = confusionMatrix.ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

        result = CompareInteractionNetworkResult()
        result.tp = tp
        result.tn = tn
        result.fp = fp
        result.fn = fn
        result.accuracy = (tp + tn) / (tp + tn + fp + fn)
        result.sensitivity = tp / (tp + fn)
        result.specificity = tn / (tn + fp)
        result.precision = tp / (tp + fp)

        if resultFoldername is not None:
            if not os.path.exists(resultFoldername):
                os.makedirs(resultFoldername)

            confusionMatrix.tofile(os.path.join(resultFoldername,
                                                "confusion_matrix.txt"),
                                   format="text")

            with open(
                    os.path.join(resultFoldername, "comparision_results.json"),
                    "w") as filep:
                json.dump(result.__dict__, filep)

            f = plt.figure()
            ax = sns.heatmap(
                confusionMatrix,
                annot=True,
                cmap='Blues',
            )
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Actual Values')
            ax.xaxis.set_ticklabels(["No Interaction", "Interaction"])
            ax.yaxis.set_ticklabels(["No Interaction", "Interaction"])
            f.savefig(os.path.join(resultFoldername, "confusion_matrix.svg"))

        return result, confusionMatrix

    @staticmethod
    def saveNetwork(filename: str, net: InteractionNetwork):
        obj = {
            "interactionsMatrix": net._interactionsMatrix.tolist(),
            "thresholdNegativeValues": net._thresholdNegativeValues,
            "thresholdPositiveValues": net._thresholdPositiveValues,
            "nodeNames": net._nodeNames
        }

        with open(filename, "w") as fp:
            json.dump(obj, fp)

    @staticmethod
    def loadNetwork(filename: str) -> InteractionNetwork:
        with open(filename, "r") as fp:
            obj = json.load(fp)

        return InteractionNetwork(
            interactionsMatrix=np.matrix(obj["interactionsMatrix"]),
            thresholdValues=[
                obj["thresholdNegativeValues"], obj["thresholdPositiveValues"]
            ],
            thresholdMethod="manual",
            nodeNames=obj["nodeNames"])

    @staticmethod
    def saveProbabilityDistribution(foldername: str, net: InteractionNetwork):
        distributions, disparams, scores, disTypes = computeProbabilityDistribution(
            net.interactionsMatrix, "auto")
        plotProbabilityDistribution(foldername, distributions)
