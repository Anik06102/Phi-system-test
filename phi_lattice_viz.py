%%writefile phi_lattice_viz.py
"""
phi_lattice_viz.py  —  Plotly visualiser for trained φ-Lattice
---------------------------------------------------------------
Call `visualise_lattice(centroids, U, …)` after training.
"""
import numpy as np, plotly.graph_objects as go
from plotly.colors import label_rgb

_SHELL_COL = ['#FF4136','#2ECC40','#0074D9','#B10DC9','#FF851B',
              '#39CCCC','#F012BE','#01FF70','#85144b']

def _mesh(r):
    u,v = np.mgrid[0:2*np.pi:40j,0:np.pi:20j]
    return r*np.cos(u)*np.sin(v), r*np.sin(u)*np.sin(v), r*np.cos(v)

def _bezier(p1,p2,steps=30,bend=0.2):
    mid = (p1+p2)/2
    n   = np.cross(p2-p1,[0,0,1]); n = n/np.linalg.norm(n) if np.linalg.norm(n)>1e-8 else np.array([0,1,0])
    ctrl= mid + bend*np.linalg.norm(p2-p1)*n
    t   = np.linspace(0,1,steps)[:,None]
    return (1-t)**2*p1 + 2*(1-t)*t*ctrl + t**2*p2

def visualise_lattice(centroids, U, *, beta=0.65, weights=None, samples=None, show=True):
    fig = go.Figure()
    # shells & centroids
    for k, layer in enumerate(centroids):
        r = beta**k
        if k>0:
            x,y,z = _mesh(r)
            col   = _SHELL_COL[(k-1)%len(_SHELL_COL)]
            fig.add_trace(go.Surface(x=x,y=y,z=z,opacity=0.07,showscale=False,
                                     colorscale=[[0,col],[1,col]],hoverinfo='skip'))
        xyz = np.array([c @ U[:,:3] for c in layer])
        wts = [(weights or {}).get(id(c),0.5) for c in layer]
        base = np.array([int(col.lstrip('#')[i:i+2],16) for col in [_SHELL_COL[(k-1)%len(_SHELL_COL)] for _ in wts] for i in (0,2,4)]).reshape(len(xyz),3)
        cols= [label_rgb((1-w)*base[i]) for i,w in enumerate(wts)]
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],
                                   mode='markers',
                                   marker=dict(size=5+4*np.array(wts),color=cols,line=dict(color='white',width=1.2)),
                                   hovertext=[f'Shell {k}, w={w:.2f}' for w in wts],
                                   hoverinfo='text'))
        # parent-child curves
        if k>0:
            parents = centroids[k-1]
            for c in layer:
                p = min(parents, key=lambda v: np.linalg.norm(c-v))
                curve = _bezier(p@U[:,:3], c@U[:,:3], bend=0.12)
                fig.add_trace(go.Scatter3d(x=curve[:,0],y=curve[:,1],z=curve[:,2],
                                           mode='lines',line=dict(width=1.2,color=_SHELL_COL[(k-1)%len(_SHELL_COL)]),
                                           hoverinfo='skip'))
    # optional sample trajectories
    if samples is not None:
        for s in samples:
            path = [np.zeros(3)]
            for layer in centroids[1:]:
                path.append(min(layer, key=lambda c: np.linalg.norm(s-c)) @ U[:,:3])
            path.append(s @ U[:,:3])
            for i in range(len(path)-1):
                seg=_bezier(path[i],path[i+1],bend=0.18)
                fig.add_trace(go.Scatter3d(x=seg[:,0],y=seg[:,1],z=seg[:,2],
                                           mode='lines',line=dict(width=2,color='gold'),hoverinfo='skip'))
            fig.add_trace(go.Scatter3d(x=[path[-1][0]],y=[path[-1][1]],z=[path[-1][2]],
                                       mode='markers',marker=dict(size=5,color='gold',symbol='cross')))
    fig.update_layout(scene=dict(bgcolor='black',xaxis_visible=False,
                                 yaxis_visible=False,zaxis_visible=False,aspectmode='data'),
                      paper_bgcolor='black',font=dict(color='white'),
                      title='φ-Lattice 3-D view',showlegend=False)
    if show: fig.show()
    return fig
