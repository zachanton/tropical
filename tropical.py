import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import torch
from torch import nn
import string
from scipy.spatial import ConvexHull, Delaunay
import JuPyMake
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from math import gcd
from fractions import Fraction
import itertools
from itertools import chain, combinations
from collections import defaultdict

JuPyMake.InitializePolymake()
JuPyMake.ExecuteCommand("application 'tropical';")

class Tropical:

    def __init__(self, val):
        self.val = val

    # Relations
    def __lt__(self, other):
        if isinstance(other, Tropical):
            return self.val - other.val < 0
        return self.val - other < 0

    def __gt__(self, other):
        if isinstance(other, Tropical):
            return self.val - other.val > 0
        return self.val - other > 0

    def __le__(self, other):
        if isinstance(other, Tropical):
            return self.val - other.val <= 0
        return self.val - other <= 0

    def __ge__(self, other):
        if isinstance(other, Tropical):
            return self.val - other.val >= 0
        return self.val - other >= 0

    def __eq__(self, other):
        if isinstance(other, Tropical):
            return self.val == other.val
        return self.val == other

    # Simple operations
    def __add__(self, other):
        if isinstance(other, Tropical):
            return Tropical(max(self.val, other.val))
        return Tropical(max(self.val, other))

    def __radd__(self, other):
        if isinstance(other, Tropical):
            return Tropical(max(self.val, other.val))
        return Tropical(max(self.val, other))

    def __mul__(self, other):
        if isinstance(other, Tropical):
            return Tropical(self.val + other.val)
        return Tropical(self.val + other)

    def __rmul__(self, other):
        if isinstance(other, Tropical):
            return Tropical(self.val + other.val)
        return Tropical(self.val + other)

    def __pow__(self, other):
        assert float(other) == int(float(other)), 'pow should be natural'
        assert float(other) >= 0, 'pow should be natural'
        if isinstance(other, Tropical):
            return Tropical(self.val * other.val)
        return Tropical(self.val * other)

    # Other
    def __abs__(self):
        return Tropical(abs(self.val))

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)

    def __float__(self):
        return float(self.val)

    def sym(self):
        return Tropical(-self.val)

    def __truediv__(self, b):
        return self * b.sym()

    def __floordiv__(self, b):
        return self * b.sym()


class TropicalMonomial:

    def __init__(self, coef):
        if isinstance(coef, TropicalMonomial):
            self.coef = coef.coef
        else:
            self.coef = [Tropical(x) if not isinstance(x, Tropical) else x for x in coef]

    def __getitem__(self, n):
        return self.coef[n]

    def __len__(self):
        return len(self.coef)

    def __eq__(self, mono):
        if len(self) == len(mono):
            if all([x == y for x, y in zip(self.coef, mono.coef)]):
                return True
        return False

    def __add__(self, other):
        assert type(other) == TropicalMonomial
        assert len(self) == len(other)

        return TropicalPolynomial([self.coef, other.coef])

    def __radd__(self, other):
        assert type(other) == TropicalMonomial
        assert len(self) == len(other)

        return TropicalPolynomial([self.coef, other.coef])

    def __mul__(self, other):
        if type(other) == TropicalMonomial:
            assert len(self) == len(other)

            return TropicalMonomial([x * y for x, y in zip(self.coef, other.coef)])

        elif type(other) == Tropical:
            return TropicalMonomial([x * other if i == 0 else x for i, x in enumerate(self.coef)])

    def __rmul__(self, other):
        if type(other) == TropicalMonomial:
            assert len(self) == len(other)

            return TropicalMonomial([x * y for x, y in zip(self.coef, other.coef)])
        elif type(other) == Tropical:
            return TropicalMonomial([x * other if i == 0 else x for i, x in enumerate(self.coef)])

    def __div__(self, other):
        assert type(other) == TropicalMonomial
        assert len(self) == len(other)

        return TropicalMonomial([x / y for x, y in zip(self.coef, other.coef)])

    def __rdiv__(self, other):
        assert type(other) == TropicalMonomial
        assert len(self) == len(other)

        return TropicalMonomial([x / y for x, y in zip(self.coef, other.coef)])

    def __pow__(self, other):
        return TropicalMonomial([x ** other for x in self.coef])

    def __repr__(self):
        return str(self)

    def __str__(self):
        d = len(self)

        var = string.ascii_lowercase

        if d == -1:
            return '0'

        if d == 0:
            return '{}'.format(self[0])

        out = []

        for pwr in range(d):

            coe = self[pwr]

            v = var[pwr - 1]

            if coe.val == 0:
                continue

            if pwr == 0:
                s = '{}'.format(coe)

            else:
                if coe == 1:
                    s = '{}'.format(v)

                else:
                    s = '{}^{}'.format(v, int(coe.val))

            out.append(s)
        
        if len(out)==0:
            out = '0'
            
        out = '⨀'.join(out)

        return out
    
    def to_latex(self):
        d = len(self)

        var = string.ascii_lowercase

        if d == -1:
            return '0'

        if d == 0:
            return '{}'.format(self[0])

        out = []

        for pwr in range(d):

            coe = self[pwr]

            v = var[pwr - 1]

            if coe.val == 0:
                continue

            if pwr == 0:
                s = '{}'.format(coe)

            else:
                if coe == 1:
                    s = '{}'.format(v)

                else:
                    s = '{}^{}'.format(v, '{'+str(int(coe.val))+'}')

            out.append(s)
        
        if len(out)==0:
            out = '0'
            
        out = '⨀'.join(out)

        return out

    def evaluate(self, point):
        """Evaluate the monomial at a given point or points"""
        point = [Tropical(x) for x in point]
        assert len(point) == len(self) - 1

        out = self.coef[0]

        for pwr, coef in enumerate(self.coef[1:]):
            out *= point[pwr] ** coef

        return out


class TropicalPolynomial:

    def __init__(self, monoms):

        if isinstance(monoms, list):
            self.monoms = {}

            for x in monoms:
                if any(isinstance(i, Tropical) for i in x):
                    x = [i.val for i in x]

                if tuple(x[1:]) in self.monoms.keys():
                    prev_mon = self.monoms[tuple(x[1:])]
                    new_x = [prev_mon[0] + x[0]] + list(x[1:])
                    self.monoms[tuple(new_x[1:])] = TropicalMonomial(new_x)
                else:
                    self.monoms[tuple(x[1:])] = TropicalMonomial(x)
        elif isinstance(monoms, dict):
            self.monoms = monoms
            
        self.pts = None
        self.new_simp = None

    def __getitem__(self, n):
        return list(self.monoms.values())[n]

    def __len__(self):
        return len(self.monoms)

    def __eq__(self, other):
        if len(set(self.monoms.keys()).symmetric_difference(other.monoms.keys())) == 0:
            return True
        return False

    def __add__(self, other):
        new_monom = self.monoms.copy()
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                if x in new_monom.keys():
                    new_x = [other.monoms[x][0] + new_monom[x][0]] + list(new_monom[x][1:])
                    new_monom[x] = TropicalMonomial(new_x)
                else:
                    new_monom[x] = other.monoms[x]
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return other + self
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val] + [0 for _ in range(len(self[0].coef) - 1)]])
            return other + self
        return TropicalPolynomial(new_monom)

    def __radd__(self, other):
        new_monom = self.monoms.copy()
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                if x in new_monom.keys():
                    new_x = [other.monoms[x][0] + new_monom[x][0]] + list(new_monom[x][1:])
                    new_monom[x] = TropicalMonomial(new_x)
                else:
                    new_monom[x] = other.monoms[x]
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return other + self
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val] + [0 for _ in range(len(self[0].coef) - 1)]])
            return other + self

        return TropicalPolynomial(new_monom)

    def __mul__(self, other):
        new_monom = {}
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                for y in self.monoms:
                    z = other.monoms[x] * self.monoms[y]
                    key_z = tuple([i.val for i in z.coef[1:]])
                    if key_z not in new_monom:
                        new_monom[key_z] = z
                    else:
                        new_monom[key_z] += z
                        new_monom[key_z] = new_monom[key_z].monoms[key_z]
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return other * self
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val] + [0 for _ in range(len(self[0].coef) - 1)]])
            return other * self

        return TropicalPolynomial(new_monom)

    def __rmul__(self, other):
        new_monom = {}
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                for y in self.monoms:
                    print(x,y)
                    z = other.monoms[x] * self.monoms[y]
                    key_z = tuple([i.val for i in z.coef[1:]])
                    new_monom[key_z] = z
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return other * self
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val] + [0 for _ in range(len(self[0].coef) - 1)]])
            return other * self

        return TropicalPolynomial(new_monom)

    def __pow__(self, other):
        new_monom = {}
        for y in self.monoms:
            z = self.monoms[y] ** other
            key_z = tuple([i.val for i in z.coef[1:]])
            new_monom[key_z] = z
        return TropicalPolynomial(new_monom)

    def __repr__(self):
        return str(self)

    def __str__(self):

        if len(self) == 0:
            return "0"

        out = []

        for pwr in self.monoms.keys():
            s = str(self.monoms[pwr])

            out.append(s)
            
        out = sorted(out, key=lambda x: x[::-1])

        out = ' ⨁ '.join(out)

        return out
    
    def to_latex(self):

        if len(self) == 0:
            return "0"

        out = []

        for pwr in self.monoms.keys():
            s = self.monoms[pwr].to_latex()

            out.append(s)
            
        out = sorted(out, key=lambda x: x[::-1])

        out = ' ⨁ '.join(out)

        return out

    def evaluate(self, point):
        """Evaluate the polynomial at a given point or points"""
        out = []

        for x in self.monoms:
            out.append(self.monoms[x].evaluate(point))

        out = max(out)

        return out
    
    def minimize_depr(self, tolerance=1e-12):
        def in_hull(p, del_hull):
            return del_hull.find_simplex(p)>=0
        
        def points_in_hull(points, hull):
            eq=hull.equations.T
            V,b=eq[:-1].T,eq[-1]
            flag = np.prod(np.dot(V,points.T)+b[:,None]<= tolerance,axis=0)
            return flag.astype(bool)

        def hit(U, hull):
            U0 = np.ones(U.shape)
            U0[:,1:] *= 0
            eq=hull.equations.T
            V,b=eq[:-1].T,eq[-1]
            num = -(b[:,None] + np.dot(V,U.T))
            num[np.isclose(num,0)] *= 0
            den = np.dot(V,U0.T)
            alpha = np.divide(num,den)
            a = np.min(alpha,axis=0,initial=np.inf,where=(~np.isnan(alpha))&(~np.isinf(alpha))&(alpha>0))
            U0[:,0] = a
            pa = U + U0
            return pa
        
        def filter_hull(points,hull):
            ch = points[hull.vertices]
            hit_p = hit(ch,hull)
            in_hull = points_in_hull(hit_p, hull)
            new_ch = ch[~in_hull]
            return sorted(new_ch.tolist())
        
        pts = np.array([[i.val for i in mon.coef] for mon in self.monoms.values()])
        if len(pts)<len(pts[0]):
            return self
        hull = ConvexHull(pts,qhull_options='Qa')
        
        new_monom = filter_hull(pts,hull)
        return TropicalPolynomial(new_monom)
    
    def minimize_depr_v2(self):
        name = 'test'
        JuPyMake.ExecuteCommand(f'${name} = toTropicalPolynomial("{self.poly_to_str()}");')

        JuPyMake.ExecuteCommand(f'$V = new Hypersurface<Max>(POLYNOMIAL=>${name});')
        JuPyMake.ExecuteCommand('$ds = $V->dual_subdivision();')
        
        pts = JuPyMake.ExecuteCommand('print $ds->POINTS;')[1]
        pts = np.array([[int(j) for j in i.split()[1:]] for i in pts.split('\n')[:-1]])
        
        simp = JuPyMake.ExecuteCommand('print $ds->MAXIMAL_CELLS;')[1]
        simp = np.array([[int(j) for j in i[1:-1].split()] for i in simp.split('\n')[:-1]])
        
        adj = JuPyMake.ExecuteCommand('for (my $i=0; $i<$ds->N_MAXIMAL_CELLS; ++$i)\
                                    {print $ds->cell($i)->GRAPH->ADJACENCY, "\t" }')[1]
        new_adj = []
        for i in adj.split('\t')[:-1]:
            new_a = []
            for j in i.split('\n')[:-1]:
                kek = j[1:-1].split()
                if len(kek)>0:
                    new_a.append([int(kek[0]),int(kek[-1])])
                else:
                    new_a.append([])
            new_adj.append(new_a)

        new_simp = []
        for i, vv in enumerate(new_adj):
            new_s = []

            for j, v in enumerate(vv):
                if len(v)>0:
                    new_s.append([simp[i][j],simp[i][v[0]]])
                    new_s.append([simp[i][j],simp[i][v[1]]])
#             new_s = merge_intervals(new_s)
            new_simp.append(new_s)
            
#         print(simp)
#         print(new_simp)
        used_points = np.unique([i for j in new_simp for i in j])
        
        return TropicalPolynomial([self.monoms[tuple(v)] for v in pts[used_points]])
    
    
    def minimize(self):
        if self.pts is not None:
            new_poly = TropicalPolynomial([self.monoms[tuple(v)] for v in self.pts])
            new_poly.pts = self.pts
            return new_poly
            
        name = 'test'
        JuPyMake.ExecuteCommand(f'${name} = toTropicalPolynomial("{self.poly_to_str()}");')
        JuPyMake.ExecuteCommand(f'$V = new Hypersurface<Max>(POLYNOMIAL=>${name});')
        JuPyMake.ExecuteCommand('$ds = $V->dual_subdivision();')
        JuPyMake.ExecuteCommand('$pc = $ds->POLYHEDRAL_COMPLEX;')
        self.pts = JuPyMake.ExecuteCommand('print $pc->VERTICES;')[1]
        self.pts = np.array([[int(j) for j in i.split()[1:]] for i in self.pts.split('\n')[:-1]])                 
        new_poly = TropicalPolynomial([self.monoms[tuple(v)] for v in self.pts])
        new_poly.pts = self.pts
        return new_poly
    

    def poly_to_str(self):
        def mon_to_str(monom):
            return str(monom.coef[0].val) + '+' +''.join([str(int(v.val))+'*'+'x{}+'.format(i) for i,v in enumerate(monom.coef[1:])])[:-1]
        
        s = 'max('+','.join([mon_to_str(mon) for mon in self.monoms.values()]) + ')'
        return s
    
    def poly_to_latex(self):
        return self.to_latex().replace('⨁', '\oplus').replace('⨀', ' \odot ')

    def plot_dual_sub(self, color='blue', label=None):
        if self.pts is not None and self.new_simp is not None:
            plt.plot(self.pts[:,0], self.pts[:,1], 'o', color='black')
            for simplex in self.new_simp:
                plt.plot(self.pts[simplex, 0], self.pts[simplex, 1], 'k-', color=color, label=label)
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            plt.legend(handles, labels, loc='best')
            return
        
        JuPyMake.ExecuteCommand(f'$a = toTropicalPolynomial("{self.poly_to_str()}");')

        JuPyMake.ExecuteCommand(f'$V = new Hypersurface<Max>(POLYNOMIAL=>$a);')
        JuPyMake.ExecuteCommand('$ds = $V->dual_subdivision();')
        JuPyMake.ExecuteCommand('$pc = $ds->POLYHEDRAL_COMPLEX;')
        
        self.pts = JuPyMake.ExecuteCommand('print $pc->VERTICES;')[1]
        self.pts = np.array([[int(j) for j in i.split()[1:]] for i in self.pts.split('\n')[:-1]])

        adj = JuPyMake.ExecuteCommand('print $pc->GRAPH->ADJACENCY;')[1]
        adj = np.array([[int(j) for j in i[1:-1].split()] for i in adj.split('\n')[:-1]])

        self.new_simp = []
        for i, vv in enumerate(adj):
            for j, v in enumerate(vv):
                if [v,i] not in self.new_simp:
                    self.new_simp.append([i,v])  
        self.new_simp = np.array(self.new_simp)
        
#         pts = JuPyMake.ExecuteCommand('print $ds->POINTS;')[1]
#         pts = np.array([[int(j) for j in i.split()[1:]] for i in pts.split('\n')[:-1]])
#         plt.plot(pts[:,0], pts[:,1], 'o', color='black')
        
        
            
        plt.plot(self.pts[:,0], self.pts[:,1], 'o', color='black')
        for simplex in self.new_simp:
            plt.plot(self.pts[simplex, 0], self.pts[simplex, 1], 'k-', color=color, label=label)
            
        if label is not None:
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            plt.legend(handles, labels, loc='best')
        return
    
    def create_curve(self):
        JuPyMake.ExecuteCommand(f'$a = toTropicalPolynomial("{self.poly_to_str()}");')

        JuPyMake.ExecuteCommand(f'$V = new Hypersurface<Max>(POLYNOMIAL=>$a);')
        JuPyMake.ExecuteCommand('$ds = $V->dual_subdivision();')
        JuPyMake.ExecuteCommand('$pc = $ds->POLYHEDRAL_COMPLEX;')

        old_pts = JuPyMake.ExecuteCommand('print $ds->POINTS;')[1]
        old_pts = np.array([[int(j) for j in i.split()[1:]] for i in old_pts.split('\n')[:-1]])

        pts = JuPyMake.ExecuteCommand('print $pc->VERTICES;')[1]
        pts = np.array([[int(j) for j in i.split()[1:]] for i in pts.split('\n')[:-1]])

        w = JuPyMake.ExecuteCommand('print $ds->WEIGHTS;')[1].split(' ')
        weights = np.array([-1*int(j) for i,j in enumerate(w) if list(old_pts[i]) in pts.tolist()])
        ext_pts = np.hstack((pts,weights[:,None]))

        adj = JuPyMake.ExecuteCommand('print $pc->MAXIMAL_POLYTOPES;')[1]
        max_cells = np.array([[int(j) for j in i[1:-1].split()] for i in adj.split('\n')[:-1]])

        if self.new_simp is None:
            self.pts = JuPyMake.ExecuteCommand('print $pc->VERTICES;')[1]
            self.pts = np.array([[int(j) for j in i.split()[1:]] for i in self.pts.split('\n')[:-1]])
        
            adj = JuPyMake.ExecuteCommand('print $pc->GRAPH->ADJACENCY;')[1]
            adj = np.array([[int(j) for j in i[1:-1].split()] for i in adj.split('\n')[:-1]])

            self.new_simp = []
            for i, vv in enumerate(adj):
                for j, v in enumerate(vv):
                    if [v,i] not in self.new_simp:
                        self.new_simp.append([i,v])  
            self.new_simp = np.array(self.new_simp)
            
        if len(self.new_simp)==1:
            self.simp_weights = {tuple(sorted(self.new_simp[0])): gcd(*np.abs(np.diff(pts[self.new_simp],axis=1)).squeeze())}
        else:
            self.simp_weights = {tuple(sorted(self.new_simp[j])): gcd(*i) for j, i in enumerate(np.abs(np.diff(pts[self.new_simp],axis=1)).squeeze())}

        self.locus_points = {}
        self.simp_d = {}
        self.region_d = {}
        
        
        for cell in max_cells:
            if len(cell)==2:
                realC, ssimp, direct = line_from_cell(cell, ext_pts, self.new_simp.tolist())
                self.simp_d[tuple(sorted(ssimp))] = {'line_center': realC, 'direct': direct, 'weight': self.simp_weights[tuple(sorted(ssimp))]}
                continue
            
            realC, ssimp, direct = rays_from_cell(cell, ext_pts, self.new_simp.tolist())
            self.locus_points[len(self.locus_points.keys())] = realC
            realC = len(self.locus_points.keys())-1
            for i in range(len(ssimp)):
                if tuple(sorted(ssimp[i])) not in self.simp_d:
#                     if (sorted(ssimp[i]) != ssimp[i]).any():
#                         direct[i] *= -1
                    self.simp_d[tuple(sorted(ssimp[i]))] = {'center': realC, 'direct': direct[i], 'weight': self.simp_weights[tuple(sorted(ssimp[i]))]}
                else:
                    self.simp_d[tuple(sorted(ssimp[i]))] = {'segment': np.array([realC, self.simp_d[tuple(sorted(ssimp[i]))]['center']]), 'weight': self.simp_weights[tuple(sorted(ssimp[i]))], 'direct': [direct[i], self.simp_d[tuple(sorted(ssimp[i]))]['direct']]}
        
        self.point_d = {}
        for k in self.simp_d.keys():
            if 'center' in self.simp_d[k]:
                new_k = self.simp_d[k]['center']
                if new_k in self.point_d:
                    self.point_d[new_k]['direct'].append(self.simp_d[k]['direct'])
                    self.point_d[new_k]['weights'].append(self.simp_d[k]['weight'])
                    self.point_d[new_k]['simp'].append(k)
                else:
                    self.point_d[new_k] = {}
                    self.point_d[new_k]['direct'] = [self.simp_d[k]['direct']]
                    self.point_d[new_k]['weights'] = [self.simp_d[k]['weight']]
                    self.point_d[new_k]['simp'] = [k]
            elif 'segment' in self.simp_d[k]:
                for i in [0,1]:
                    new_k = self.simp_d[k]['segment'][i]
                    if new_k in self.point_d:
                        self.point_d[new_k]['direct'].append(self.simp_d[k]['direct'][i])
                        self.point_d[new_k]['weights'].append(self.simp_d[k]['weight'])
                        self.point_d[new_k]['simp'].append(k)
                    else:
                        self.point_d[new_k] = {}
                        self.point_d[new_k]['direct'] = [self.simp_d[k]['direct'][i]]
                        self.point_d[new_k]['weights'] = [self.simp_d[k]['weight']]
                        self.point_d[new_k]['simp'] = [k]

    def plot_curve(self, color='blue', length=10, label=None):
    
        self.create_curve()
        
        for v in self.simp_d.values():
            if 'segment' in v:
                plot_segment(v['segment'][0], v['segment'][1], self.locus_points, color, label)
            elif 'center' in v:
                plot_ray(v['center'], v['direct'], self.locus_points, length, color, label)
            else:
                plot_line(v['line_center'], v['direct'], length, color, label)
                
        if label is not None:
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            plt.legend(handles, labels, loc='best')
            
    def trivial_factorize(self):
        new_h = []
        for v in self.monoms:
            new_h.append(list(np.array([i.val for i in self.monoms[v].coef])))
        hmin = np.min(new_h,axis=0)
        new_h = (np.array(new_h) - hmin).tolist()
        new_h = TropicalPolynomial(new_h)

        return new_h, [TropicalPolynomial([(j*np.eye(len(hmin))[i]).tolist()]) for i,j in enumerate(hmin)]

            
    def factorize(self):
        h_min, emin = self.trivial_factorize()
        h_min.create_curve()

        decomp_point_d = {}
        for i in h_min.point_d:
            p = h_min.point_d[i]
            direct, weights, simp = np.array(p['direct']),np.array(p['weights']), np.array(p['simp'])
            decomp_point_d[i] = decompose_point(direct, weights, simp) 


        for k in h_min.simp_d.keys():
            if 'segment' in h_min.simp_d[k]:
                break
        it_buf = [(0,0)]        
        out = [[k]]

        history = set()


        for i, S in enumerate(out):
            if set(S) == set(h_min.simp_d):
                continue
            while True:
                New_S = S.copy()
                for s in S:
                    if 'segment' not in h_min.simp_d[s]:
                        continue
                    n1,n2 = h_min.simp_d[s]['segment']
                    possible_decomp1 = possible_decomp(decomp_point_d, n1, s)
                    possible_decomp2 = possible_decomp(decomp_point_d, n2, s)
                    New_S0 = list(set(New_S).union(set(possible_decomp1[0])).union(set(possible_decomp2[0])))
                    history.update([frozenset(New_S0)])
                    for ii in range(len(possible_decomp1)):
                        pd1 = possible_decomp1[ii]
                        for jj in range(len(possible_decomp2)):
                            pd2 = possible_decomp2[jj]
                            if ii==jj==0:
                                continue
                            New_S_ = list(set(New_S).union(set(pd1)).union(set(pd2)))
                            if set(New_S_) not in history:
                                history.update([frozenset(New_S_)])
                                out.append(New_S_)
                    New_S = New_S0
                if set(New_S0) == set(S):
                    out[i] = set(S)
                    break
                else:
                    S = New_S0
        out = list(map(list, set(map(frozenset, out))))
        out += [list(set(h_min.simp_d.keys()) - set(j)) for j in out if len(j)!=len(h_min.simp_d.keys())]
        
        smallest_factors = []
        for fact1 in out:
            f = True
            for fact2 in out:
                if set(fact2) < set(fact1):
                    f = False
                    break
            if f:
                smallest_factors.append(fact1)
                
        self.out = [[]]
        for fact in smallest_factors:
            for dec in self.out:
                if len(set(sum(dec,[])).intersection(set(fact)))==0:
                    dec.append(fact)
                    break

        graph = create_graph(h_min.new_simp.tolist().copy())

        factors = []

        for decompose in self.out:
            fact = []
            for i in range(len(decompose)):
                sub_curve = decompose[i].copy()
                min_edg = list(set(mst2(sub_curve.copy(), graph)))
                min_graph = create_graph(min_edg)
                start = sorted(min_graph.keys())[0]
                bias_monoms,edgs,coef1 = dfs(h_min, sub_curve, min_graph, start, [], [np.array([0.,0.])], 0, [], [0])
                bias_monoms,coef1 = np.array(bias_monoms),np.array(coef1)

                monoms1 = bias_monoms-np.clip(np.min(bias_monoms,axis=0), None,0)
                coef1 = (coef1 - coef1.min())[:,None].astype(int)
                factor1 = TropicalPolynomial(np.hstack((coef1, monoms1)).tolist())
                factor1, fmin = factorize_monomials(factor1)
                fact.append(factor1.minimize())

            factors.append(fact)
        
        return factors
    
def line_from_cell(cell,pts,simp):
    ssimp = np.array(cell)

    a,b,c = np.diff(pts[ssimp], axis=0)[0]

    real_center = np.array([0, Fraction(-c, b)])
    
    direct = []
    centerX = Fraction(sum(pts[cell,0]), len(cell))
    centerY = Fraction(sum(pts[cell,1]), len(cell))
    cell_center = np.array([centerX, centerY])
    
    a, b = -b, a
    
#     k = ssimp
#     a = pts[k[0]][:-1]
#     b = pts[k[1]][:-1]
#     print(a,b)

#     n = b - a
#     v = cell_center - a

#     z_ = a + n*Fraction(np.dot(v, n),np.dot(n, n))
#     print(z_)
#     a,b = z_ - cell_center
#     print(a,b)
    a,b = (a*a.denominator*b.denominator).numerator, (b*a.denominator*b.denominator).numerator
    a,b = a/gcd(a,b), b/gcd(a,b)
    direct.append((a,b))

    return real_center, ssimp, direct


def rays_from_cell(cell,pts,simp):
    ssimp = []
    for s in simp:
        if s[0] in cell and s[1] in cell:
            ssimp.append(s)
    ssimp = np.array(ssimp)

    coef_diff = np.diff(pts[ssimp], axis=1)
    
    L1, L2 = coef_diff[:2,0,:]
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    real_center = np.array([Fraction(-Dx, D), Fraction(-Dy, D)])
    
    direct = []
    centerX = Fraction(sum(pts[cell,0]), len(cell))
    centerY = Fraction(sum(pts[cell,1]), len(cell))
    cell_center = np.array([centerX, centerY])
    for k in ssimp:

        a = pts[k[0]][:-1]
        b = pts[k[1]][:-1]

        n = b - a
        v = cell_center - a
        
        z_ = a + n*Fraction(np.dot(v, n),np.dot(n, n))
        a,b = z_ - cell_center
        a,b = (a*a.denominator*b.denominator).numerator, (b*a.denominator*b.denominator).numerator
        a,b = a/gcd(a,b), b/gcd(a,b)
        direct.append((a,b))

    return real_center, ssimp, direct


def plot_line(point, direct, length, color, label):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will plot the line on a 10 x 10 plot.
    '''

    x, y = point
    direct = direct[0]
    
    if direct[0]==0:
        
        endy = y + np.sign(direct[1]) * length
        endx = x
        
        y = y - np.sign(direct[1]) * length
        
    else:
        angle = direct[1]/direct[0]
        det = np.sign(direct[0])
        
        l = np.sqrt(length**2/(1+angle**2))
        if det<0:
            endy = y - l * angle
            endx = x - l
            
            y = y + l * angle
            x = x + l 
        else:
            endy = y + l * angle
            endx = x + l 
            
            y = y - l * angle
            x = x - l
    plt.plot([x, endx], [y, endy], color=color, label=label)
    
    
def plot_ray(point, direct, locus_points, length, color, label):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will plot the line on a 10 x 10 plot.
    '''
    point = locus_points[point]
    
    plt.plot([point[0]],[point[1]], 'o', color='black')
    x, y = point
    
    if direct[0]==0:
        
        endy = y + np.sign(direct[1]) * length
        endx = x
        
    else:
        angle = direct[1]/direct[0]
        det = np.sign(direct[0])
        
        l = np.sqrt(length**2/(1+angle**2))
        if det<0:
            endy = y - l * angle
            endx = x - l
        else:
            endy = y + l * angle
            endx = x + l 
    plt.plot([x, endx], [y, endy], color=color, label=label)
    
def plot_segment(point1, point2, locus_points, color,label):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will plot the line on a 10 x 10 plot.
    '''
    
    point1 = locus_points[point1]
    point2 = locus_points[point2]
    
    plt.plot([point1[0],point2[0]], [point1[1],point2[1]], 'o', color='black')
    plt.plot([point1[0],point2[0]], [point1[1],point2[1]], 'k-', color=color, label=label)
            
                
def factorize_monomials(h):
    new_h = []
    for v in h.monoms:
        new_h.append(list(np.array([i.val for i in h.monoms[v].coef])))
    hmin = np.min(new_h,axis=0)
    new_h = (np.array(new_h) - hmin).tolist()
    new_h = TropicalPolynomial(new_h)

    return new_h, hmin
    
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2,len(s)+1))

def decompose_point(direct, weights, simp):
    s = []
    for i in powerset(range(len(direct))):
        mask = np.zeros(len(direct),dtype=bool)
        mask[list(i)] = True
        
        we = [list(range(1,i+1)) for i in weights[mask]]
        we = np.array(list(itertools.product(*we)))[:,:,None]
        for wi in we:
            
            if ((direct[mask]*wi).sum(0) == [0.,0.]).all():
                s.append(list(tuple(x) for x in simp[mask]))
    return s

def possible_decomp(d, n, s):
    buf = []
    for e in d[n]:
        if s in e:
            buf.append(e)
    return buf


def create_graph(edgelist):
    graph = {}
    for e1, e2 in edgelist:
        graph.setdefault(e1, []).append(e2)
        graph.setdefault(e2, []).append(e1)
    return graph

def mst(start, graph):
    closed = set()
    edges = []
    q = [(start, start)]
    while q:
        v1, v2 = q.pop()
        if v2 in closed:
            continue
        closed.add(v2)
        edges.append((v1, v2))
        for v in graph[v2]:
            if v in graph:
                q.append((v2, v))
    del edges[0]
    assert len(edges) == len(graph)-1
    return edges

def mst2(start_edges, graph):
    closed = set()
    edges = []#start_edges.copy()
    q = start_edges
    while q:
        v1, v2 = q.pop()
        if v2 in closed:
            continue
        closed.add(v2)
        edges.append((v1, v2))
        for v in graph[v2]:
            if v in graph:
                q.append((v2, v))
    del edges[0]
#     assert len(edges) == len(graph)-1
    return edges

def dfs(poly, curve, graph, node, visited, monoms, k, edgs, lol):
    if node not in visited:
        visited.append(node)
#         if set(visited) == set(graph.keys()):
#             return monoms
        for n in graph[node]:
            if n not in visited:
                edg = tuple(sorted([n,node]))
                simp_edg = poly.simp_d[edg]

                a,b = poly.pts[node] - poly.pts[n]
                kek = np.array([a,b])
                cur_monom = monoms[k] - kek
                
                if 'segment' in simp_edg:
                    loc_point = poly.locus_points[simp_edg['segment'][0]]
                else:
                    loc_point = poly.locus_points[simp_edg['center']]
                cur_lol = lol[k]+np.dot(loc_point,poly.pts[node])-np.dot(loc_point,poly.pts[n])
                if edg in curve:
                    
                    lol.insert(k+1,cur_lol)
                    
#                     print(edg)
                    if 'center' in simp_edg:
                        center = simp_edg['center']
                        w = simp_edg['weight']
#                         print(cur_monom, h_min.locus_points[center], w)
                    else:
                        a,b = simp_edg['segment']
                        w = simp_edg['weight']
#                         print(cur_monom, h_min.locus_points[a], h_min.locus_points[b], w)
#                     print(simp_edg)
                    monoms.insert(k+1,cur_monom)
                    edgs.append(edg)
                    dfs(poly, curve, graph,n, visited, monoms, k+1,edgs,lol) 
                else:
#                     monoms.insert(k+1,monoms[k])
                    dfs(poly, curve, graph,n, visited, monoms, k,edgs,lol) 
    return monoms, edgs, lol


class PolyNet(nn.Module):
    def __init__(self, poly):
        super().__init__()

        self.linears = convert_polynomial_to_net(poly)

    def forward(self, output):
        for i, l in enumerate(self.linears):
            output = l.forward(output)
            if i<len(self.linears)-1:
                output = torch.relu(output)
        return output


def convert_monomial_to_net(monom):
    bias = monom.coef[0].val
    weight = [x.val for x in monom.coef[1:]]

    layer = nn.Linear(len(weight), 1)

    layer.weight.data.copy_(to_tensor(weight))
    layer.bias.data.copy_(to_tensor(bias))

    return layer


def to_tensor(x):
    return torch.tensor(x).float()


def create_w1(n):
    basis = np.array([[1., -1.], [0., 1.], [0., -1.]])
    if n % 2 == 0:
        affine = np.kron(np.eye(n // 2, dtype=int), basis)
    else:
        affine = np.kron(np.eye((n - 1) // 2, dtype=int), basis)
        affine = np.vstack([affine, np.zeros((2,affine.shape[1]))])
        affine = np.hstack([affine, np.zeros((affine.shape[0], 1))])
        affine[-2, -1] = 1
        affine[-1, -1] = -1
    return to_tensor(affine)


def create_w2T(n):
    if n % 3 == 0:
        n = n // 3
        basis = np.array([[1., 1., -1.]])
        w = np.kron(np.eye(n, dtype=int), basis)
    else:
        basis = np.array([[1., 1., -1.]])
        n = n // 3
        w = np.kron(np.eye(n, dtype=int), basis)
        w = np.vstack([w, np.zeros(w.shape[1])])
        w = np.hstack([w, np.zeros((w.shape[0], 1))])
        w = np.hstack([w, np.zeros((w.shape[0], 1))])
        w[-1, -2] = 1
        w[-1, -1] = -1
    
    w = to_tensor(w)
    b = torch.zeros(len(w))
    return w, b

def init_wb_from_poly(poly):
    weights = []
    biases = []
    for c in poly.monoms.keys():
        ab = poly.monoms[c].coef[0].val
        aw = [x.val for x in poly.monoms[c].coef[1:]]
        weights.append(to_tensor(aw))
        biases.append(to_tensor(ab))
    return torch.stack(weights), torch.stack(biases)
    

def linear_from_wb(w,b):
    l = nn.Linear(*w.shape[::-1])
    l.weight.data.copy_(w)
    l.bias.data.copy_(b)
    return l


def convert_polynomial_to_net(poly):
    layers = []

    w2, b = init_wb_from_poly(poly)
    
    if len(poly) == 1:
        l = linear_from_wb(w,b)
        net = nn.ModuleList([l])
        return net
    
    w1 = create_w1(w2.shape[0])
    w = w1 @ w2
    b = w1 @ b
    l = linear_from_wb(w,b)
    layers.append(l)
    
    while w.shape[0]>3:
        l_of = layers[-1].out_features
        w2, b = create_w2T(l_of)
        w1 = create_w1(w2.shape[0])
        w = w1 @ w2
        b = w1 @ b
        l = linear_from_wb(w,b)
        layers.append(l)

    w, b = create_w2T(3)
    l = linear_from_wb(w,b)
    layers.append(l)
    net = nn.ModuleList(layers)
    return net


class DiffPolyNet(nn.Module):
    def __init__(self, poly1, poly2):
        super().__init__()

        self.net = convert_polynomial_diff_to_net(poly1, poly2)

    def forward(self, x):
        x = torch.cat((x, x), dim=0)
        o = self.net.forward(x)
        return o


def convert_polynomial_diff_to_net(poly1, poly2):
    net1 = convert_polynomial_to_net(poly1)
    net2 = convert_polynomial_to_net(poly2)

    layers = []
    flag = False

    try:
        len_net1 = len(net1)
    except:
        len_net1 = 1
        net1 = [net2]

    try:
        len_net2 = len(net2)
    except:
        len_net2 = 1
        net2 = [net2]

    n = max(len_net1, len_net2)
    for i in range(0, n, 2):
        if (i < len_net1) & (i < len_net2):

            w1 = net1[i].weight.data
            b1 = net1[i].bias.data

            w2 = net2[i].weight.data
            b2 = net2[i].bias.data

            r_1 = torch.cat((w1, torch.zeros((w1.shape[0], w2.shape[1]))), dim=1)
            r_2 = torch.cat((torch.zeros((w2.shape[0], w1.shape[1])), w2), dim=1)

            w = torch.cat((r_1, r_2), dim=0)

            b = torch.cat((b1, b2), dim=0)

            l = nn.Linear(*w.shape[::-1])

            l.weight.data.copy_(w)
            l.bias.data.copy_(b)
            layers.append(l)

        elif i < len_net1:

            w = net1[i].weight.data
            b = net1[i].bias.data

            if not flag:
                affine = torch.zeros((w.shape[1], w.shape[1] + 1))
                for j in range(affine.shape[0] - 1):
                    affine[j, j] = 1
                    if j % 3 == 0:
                        affine[j, -1] = -1

                w = w @ affine

            l = nn.Linear(*w.shape[::-1])

            l.weight.data.copy_(w)
            l.bias.data.copy_(b)
            layers.append(l)

            flag = True

        elif i < len_net2:
            w = net2[i].weight.data
            b = net2[i].bias.data

            if not flag:
                affine = torch.zeros((w.shape[1], w.shape[1] + 1))
                for j in range(affine.shape[0]):
                    affine[j, j + 1] = 1
                    if j % 3 == 0:
                        affine[j, 0] = -1
                w = w @ affine

            l = nn.Linear(*w.shape[::-1])
            if l.out_features == 1:
                w *= -1

            l.weight.data.copy_(w)
            l.bias.data.copy_(b)
            layers.append(l)
            flag = True

    net = [layers[0]]

    for j in range(1, len(layers)):
        net.append(nn.ReLU())
        net.append(layers[j])

    if net[-1].out_features == 2:
        la = nn.Linear(2, 1)
        la.weight.data.copy_(torch.tensor([1., -1.]))
        la.bias.data.copy_(torch.tensor([0.]))
        net.append(la)

    net = nn.Sequential(*net)

    return net
        
def convert_net_to_tropical(net, t = Tropical(0)):
    d = net.linears[0].in_features
    
    f_coef = np.eye(d,d+1,k=1,dtype=int)
    f = [TropicalPolynomial([coef]) for coef in f_coef]
    
    g_coef = np.zeros((d,d+1),dtype=int)
    g = [TropicalPolynomial([coef]) for coef in g_coef]


    for l in net.linears:
        
        n = l.in_features
        m = l.out_features
        a = l.weight.data.detach().cpu().numpy()
        a_plus = np.maximum(a,0)
        a_minus = np.maximum(-a,0)
        
        if l.bias is not None:
            b_ = l.bias.data.detach().cpu().numpy()
        else:
            b_ = np.zeros(m)
        
        new_g = []
        new_h = []
        new_f = []
        
        for i in range(m):
            g_i = Tropical(0)
            h_i = Tropical(b_[i])
            
            for j in range(n):
                g_i = f[j]**a_minus[i][j] * g[j]**a_plus[i][j] * g_i
                h_i = f[j]**a_plus[i][j] * g[j]**a_minus[i][j] * h_i 
                
            new_g.append(g_i)
            new_h.append(h_i)
            new_f.append(h_i+g_i*t)
            
        f = new_f
        g = new_g
        h = new_h

    return h, g