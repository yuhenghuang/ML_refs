#Python version of SimpleEdge function
#for postprocess of HED
#the original function is in Matlab and can be found on
#https://github.com/phillipi/pix2pix/blob/master/scripts/edges/PostprocessHED.m
#there might be some problem with the 2D triangular filter (not tested)
#defining the filter explicitly in main environment should not be of any problem
#running time in the test for image of size (384,384) is around 40ms


import cv2
from numba import njit
from scipy.signal import triang, convolve2d
from skimage.morphology import remove_small_objects, skeletonize

f1=triang(tri_size*2+1)/(tri_size+1)
filter_2D=np.matmul(f1[:,np.newaxis],f1[None]) #2D filter in postprocessing

def SimpleEdge(E1, filter_2D=filter_2D, threshold=25., small_edge=5):
    threshold/=255.
    E2=EdgeNMS(E1, filter_2D)
    E3=(E2>threshold)
    E4=skeletonize(E3)
    #connectivity=1, 4-direction; 2, 8-direction
    remove_small_objects(E4, min_size=small_edge, connectivity=2, in_place=True)
    return 1-E4

def EdgeNMS(E,filter_2D):
    convTri = convolve2d(E, filter_2D, mode='same', boundary='symm')
    Ox, Oy = np.gradient(convTri, axis=(1,0), edge_order=2)
    Oxx, _ = np.gradient(Ox, axis=(1,0), edge_order=2)
    Oxy, Oyy = np.gradient(Oy, axis=(1,0), edge_order=2)
    O = np.arctan(np.sign(-Oxy)*Oyy/(Oxx+1e-5)) % np.pi
    
    suppress=5 if max(E.shape)==384 else 3
    E_nms = edgesNmsMex(E, O, 1, suppress, 1.01)
    return E_nms

@njit
def interp(E, h, w, y, x):
    y=min(max(0,y),h-1.001)
    x=min(max(0,x),w-1.001)
    y0,x0=int(y),int(x)
    y1,x1=y0+1,x0+1
    dy0,dx0=y-y0,x-x0
    dy1,dx1=1-dy0,1-dx0
    return E[y0,x0]*dy1*dx1+E[y0,x1]*dy1*dx0+E[y1,x0]*dy0*dx1+E[y1,x1]*dy0*dx0

@njit
def edgesNmsMex(E, O, r, s, m):
    buffer=0.4
    EO=E.copy()
    h,w=E.shape
    for x in range(w):
        for y in range(h):
            e=E[y,x]
            if not e:
                continue
            e*=m
            cos0,sin0=np.cos(O[y,x]),np.sin(O[y,x])
            for d in range(-r,r+1):
                if d:
                    e0=interp(E,h,w,y+d*sin0,x+d*cos0)
                    if e<e0:
                        EO[y,x]=0
                        break
    for x in range(s):     #fix margins
        for y in range(h):
            EO[y,x]*=(x+buffer)/(s+buffer)
            EO[y,w-1-x]*=(x+buffer)/(s+buffer)
    for x in range(w):
        for y in range(s):
            EO[y,x]*=(y+buffer)/(s+buffer)
            EO[h-1-y,x]*=(y+buffer)/(s+buffer)
    return EO
