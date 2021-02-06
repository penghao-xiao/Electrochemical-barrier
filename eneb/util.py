"""
Various utility functions for the neb.
"""


import numpy
from math import sqrt, sin, cos, log, pi, ceil

def sPBC(vdir):
    return (vdir % 1.0 + 1.5) % 1.0 - 0.5
            

def DBC(r, box = None, ibox = 0):
    """
    Applies periodic boundary conditions.
    Parameters:
        r:      the vector the boundary conditions are applied to
        box:    the box that defines the boundary conditions
        ibox:   the inverse of the box
    """
    if box is None:
        printf("No box given", ERR)
        return r
    if ibox is 0:
        ibox = numpy.linalg.inv(box)
    vdir = numpy.dot(r, ibox)
    vdir = (vdir % 1.0 + 1.5) % 1.0 - 0.5
    return numpy.dot(vdir, box)


def vproj(v1, v2):
    """
    Returns the projection of v1 onto v2
    Parameters:
        v1, v2: numpy vectors
    """
    mag2 = vmag2(v2)
    if mag2 == 0:
        printf("Can't project onto a zero vector", ERR)
        return v1
    return v2 * (vdot(v1, v2) / mag2)


def vunit(v):
    """
    Returns the unit vector corresponding to v
    Parameters:
        v:  the vector to normalize
    """
    mag = vmag(v)
    if mag == 0:
        printf("Can't normalize a zero vector", ERR)
        return v
    return v / mag


def vmag(v):
    """
    Returns the magnitude of v
    """
    return numpy.sqrt(numpy.vdot(v,v))
    #return sqrt((v * v).sum())


def vmag2(v):
    """
    Returns the square of the magnitude of v
    """
    return (v * v).sum()


def vdot(v1, v2):
    """
    Returns the dot product of v1 and v2
    """
    return (v1 * v2).sum()

def v2rotate(V1, V2, tTh):
    """
    Rotate V1 and V2 in their plane by the angle tTh
    """
    cTh = cos(tTh)
    sTh = sin(tTh)
    V1tmp = V1
    V1 = V1 * cTh + V2 * sTh
    V2 = V2 * cTh - V1tmp * sTh


