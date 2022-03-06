import seaborn as sns
import pandas
import numpy as np
from pmgan.particle_mover import ParticleMoverDistances

class MatchingParticleMovementDensityPlot(object):
    pmd : ParticleMoverDistances

    def __init__(self, pmd : ParticleMoverDistances):
        self.pmd = pmd
        self.x1 = []
        self.y1 = []
        self.weights1 = []
        self.c1 = []

        self.x2 = []
        self.y2 = []
        self.weights2 = []
        self.c2 = []

    def collect_marriage_xy_samples(self):
        mm = self.pmd.marriages_xy
        for i,j in mm:
            self.x1.append(float(self.pmd.samples_x.data[i]))
            self.y1.append(float(self.pmd.samples_y.data[j]))
            self.c1.append(float(self.pmd.sources[i,0]))
        try:

            self.weights1.extend(self.pmd.marriage_weights_xy)
        except:
            pass

    def collect_marriage_yx_samples(self):
        mm = self.pmd.marriages_yx
        for i,j in mm:
            self.y2.append(float(self.pmd.samples_y.data[i]))
            self.x2.append(float(self.pmd.samples_x.data[j]))
            self.c2.append(float(self.pmd.sources[j,0]))
        try:
            self.weights2.extend(self.pmd.marriage_weights_yx)
        except:
            pass

    def jointplot(self, kind='kde', show_1=True, show_2=True, **kwargs):
        x = []
        y = []
        if show_1:
            x.extend(self.x1)
            y.extend(self.y1)
        if show_2:
            x.extend(self.x2)
            y.extend(self.y2)
        sns.jointplot(x=np.array(x), y=np.array(y), stat_func=None, kind=kind, **kwargs)

    def jointplot2(self, show_1=True, show_2=True, bw='scott', **kwargs):
        x = []
        y = []
        c = []

        if show_1:
            x.extend(self.x1)
            y.extend(self.y1)
            c.extend(self.c1)
        if show_2:
            x.extend(self.x2)
            y.extend(self.y2)
            c.extend(self.c2)
        x = np.array(x)
        y = np.array(y)
        c = np.array(c)
        g = sns.JointGrid(x=x, y=y)
        #sns.kdeplot(x, ax=g.ax_marg_x, bw=bw, legend=False)
        #sns.kdeplot(y, ax=g.ax_marg_y, bw=bw, vertical=True, legend=False)
        sns.distplot(x, hist=True, kde=True, ax=g.ax_marg_x, bins=50 )
        sns.distplot(y, hist=True, kde=True, ax=g.ax_marg_y, bins=50, vertical=True )
        g.ax_joint.scatter(x[c==0], y[c==0],c='r', alpha=0.2)
        g.ax_joint.scatter(x[c!=0], y[c!=0],c='b', alpha=0.2)


    def reset(self):
        self.x1 = []
        self.y1 = []
        self.c1 = []
        self.weights1 = []
        self.x2 = []
        self.y2 = []
        self.c2 = []
        self.weights2 = []



