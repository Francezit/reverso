import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path as pth
import os
import numpy as np
import imageio


class EnvironmentAnalysis():
    _data: np.ndarray
    _nameComponents: list

    def __init__(self, data: np.ndarray, nameComponents: list) -> None:
        self._data = data
        self._nameComponents = nameComponents

    def _getData(self, ranges: list = None):
        if ranges is not None:
            data = self._data[ranges[0]:ranges[1], :]
            obs = data.shape[0]
            xdata = np.arange(ranges[0], ranges[1], dtype=np.int32)
        else:
            data = self._data
            obs = data.shape[0]
            xdata = np.arange(0, obs, dtype=np.int32)
        ncomps = data.shape[1]
        return data, xdata, obs, ncomps

    def generateGIF(self,
                    filename: str,
                    ranges: list = None,
                    fps=1,
                    window=None):
        y, x, obs, ncomps = self._getData(ranges)

        base_folder = pth.dirname(filename)
        frames = []
        for t in range(obs):
            plt.figure()
            if window is not None and t > window:
                st = t - window
                plt.plot(x[st:(t + 1)], y[st:(t + 1), :])
            else:
                plt.plot(x[:(t + 1)], y[:(t + 1), :])
            plt.legend(self._nameComponents,
                       loc='center right',
                       bbox_to_anchor=(1.25, 0.5))
            plt.xlabel("Time Step")
            plt.ylabel("Values")

            f = pth.join(base_folder, 'img_' + str(t) + '.png')
            plt.savefig(f, transparent=False)
            plt.close()

            frame = imageio.v2.imread(f)
            frames.append(frame)
            os.remove(f)

        plt.close()
        imageio.mimsave(filename, frames, fps=fps)

    def saveSnapshotImage(self, filename: str, ranges: list = None, useYLimit: bool = False, enable_bounds: bool = False):
        data, xdata, obs, ncomps = self._getData(ranges)
        if enable_bounds:
            data = data.clip(0, 1)

        plt.clf()
        with mpl.rc_context({'lines.linewidth': 1}):
            plt.plot(xdata, data, linewidth=0.7)
            plt.legend(self._nameComponents)
            plt.xlabel("Time step")
            plt.ylabel("Values")
            if useYLimit:
                my = data.max()
                plt.ylim(0, my)
            elif enable_bounds:
                plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(filename)
            plt.clf()

    def saveData(self, filename, ranges: list = None):
        data, xdata, obs, ncomps = self._getData(ranges)
        np.savetxt(filename, data, fmt='%.10f', delimiter=',')

    def export(self, foldername: str, ranges: list = None):
        self.saveData(pth.join(foldername, "output.csv"), ranges)
        self.saveSnapshotImage(pth.join(foldername, "output.svg"), ranges)
