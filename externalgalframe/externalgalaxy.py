"""Module for astropy coordinate frames representing external galaxies.
Specific examples for M31, M33, and the LMC are provided, with reasonable
(but not necessarily up-to-date) parameter defaults. Also provided are
routines for transforming between sky coordinates and galaxy disk plane,
and between SkyCoord and tangent-plane (gnomonic) coordinates."""

# This probably makes poor use of the astropy coordinate framework's flexibility, but it appears to work.
# Currently each explicitly named frame requires its own pair of icrs_to_gal, gal_to_icrs transformations with appropriate decorator.


import numpy as np

from astropy import units as u
from astropy.coordinates import representation as r
from astropy.coordinates import ICRS, SkyCoord, SkyOffsetFrame
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
from astropy.coordinates.baseframe import BaseCoordinateFrame, frame_transform_graph
from astropy.coordinates.attributes import CoordinateAttribute, QuantityAttribute, DifferentialAttribute
from astropy.coordinates.transformations import AffineTransform
from astropy.coordinates.errors import ConvertError


class ExternalGalaxyFrame(BaseCoordinateFrame):
    """Astropy coordinate frame centered on an external galaxy.

    The frame is defined by sky coordinate, heliocentric distance and velocity, and inclination and position angle.
    Modeled on astropy's Galactocentric class, but without that class's defaults-setting mechanism.
    x, y, and z are defined such that with inclination and position angle both 0, x points along RA,
    y along dec, and z into sky. With nonzero inclination y will still point within the plane of the sky,
    while x and z rotate about the y axis.

    Keyword Arguments:
        gal_coord: Scalar SkyCoord or Frame object
            Contains sky position of galaxy
        gal_distance: Quantity (units of length, e.g. u.kpc)
            Distance to galaxy
        galvel_heliocentric: CartesianDifferential, units of velocity
            3-d heliocentric velocity of galaxy, in the sky-aligned cartesian frame pointing to galaxy
            (x=East, y=North, z=distance)
        inclination: Quantity (units of angle)
        PA: Quantity (units of angle)
    """
    default_representation = r.CartesianRepresentation
    default_differential = r.CartesianDifferential

    # frame attributes
    # notes: apparently you can't call something "distance"?
    # also, ipython reload insufficient to reset attributes - have to quit out and restart python?
    gal_coord = CoordinateAttribute(frame=ICRS)   # PA defined relative to ICRS ra/dec
    gal_distance = QuantityAttribute(unit=u.kpc)
    galvel_heliocentric = DifferentialAttribute(
        allowed_classes=[r.CartesianDifferential])
    inclination = QuantityAttribute(unit=u.deg)
    PA = QuantityAttribute(unit=u.deg)

    def __init__(self, *args, skyalign=False, **kwargs):
        # left some example code for subclasses in here
        default_params = {
            'gal_coord': ICRS(ra=10.68470833 * u.degree, dec=41.26875 * u.degree),
            "gal_distance": 780.0 * u.kpc,
            "galvel_heliocentric": r.CartesianDifferential([125.2, -73.8, -300.] * (u.km / u.s)),
            "inclination": (-77.) * u.degree,
            "PA": 37. * u.degree,
        }
        kwds = dict()
        kwds.update(default_params)
        if skyalign:
            kwds['inclination'] = 0. * u.deg
            kwds['PA'] = 0. * u.deg
        kwds.update(kwargs)
        super().__init__(*args, **kwds)


def _get_matrix_vectors(gal_frame, inverse=False):
    """
    Utility function: use the ``inverse`` argument to get the inverse transformation, matrix and
    offsets to go from external galaxy frame to ICRS.
    """
    # this is based on the astropy Galactocentric class code
    # shorthand
    gcf = gal_frame

    # rotation matrix to align x(ICRS) with the vector to the galaxy center
    mat0 = np.array([[0., 1., 0.],
                     [0., 0., 1.],
                     [1., 0., 0.]])   # relabel axes so x points in RA, y in dec, z away from us
    mat1 = rotation_matrix(-gcf.gal_coord.dec, 'y')
    mat2 = rotation_matrix(gcf.gal_coord.ra, 'z')

    # construct transformation matrix and use it
    R = matrix_product(mat0, mat1, mat2)

    # Now define translation by distance to galaxy center - best defined in current sky coord system defined by R
    translation = r.CartesianRepresentation(gcf.gal_distance * [0., 0., 1.])

    # Now rotate galaxy to right orientation
    pa_rot = rotation_matrix(-gcf.PA, 'z')
    inc_rot = rotation_matrix(gcf.inclination, 'y')
    H = matrix_product(inc_rot, pa_rot)

    # compute total matrices
    A = matrix_product(H, R)

    # Now we transform the translation vector between sky-aligned and galaxy-aligned frames
    offset = -translation.transform(H)

    if inverse:
        # the inverse of a rotation matrix is a transpose, which is much faster
        #   and more stable to compute
        A = matrix_transpose(A)
        offset = (-offset).transform(A)
        # galvel is given in sky-aligned coords, need to transform to ICRS, so use R transpose
        R = matrix_transpose(R)
        offset_v = r.CartesianDifferential.from_cartesian(
            (gcf.galvel_heliocentric).to_cartesian().transform(R))
        offset = offset.with_differentials(offset_v)

    else:
        # galvel is given in sky-aligned coords, need to transform to gal frame, so use H
        offset_v = r.CartesianDifferential.from_cartesian(
            (-gcf.galvel_heliocentric).to_cartesian().transform(H))
        offset = offset.with_differentials(offset_v)

    return A, offset


def _check_coord_repr_diff_types(c):
    if isinstance(c.data, r.UnitSphericalRepresentation):
        raise ConvertError("Transforming to/from a Galactocentric frame "
                           "requires a 3D coordinate, e.g. (angle, angle, "
                           "distance) or (x, y, z).")

    if ('s' in c.data.differentials and
            isinstance(c.data.differentials['s'],
                       (r.UnitSphericalDifferential,
                        r.UnitSphericalCosLatDifferential,
                        r.RadialDifferential))):
        raise ConvertError("Transforming to/from a Galactocentric frame "
                           "requires a 3D velocity, e.g., proper motion "
                           "components and radial velocity.")


# next two functions are shorthands to remove a few lines later
def from_icrs(icrs_coord, gal_frame):
    _check_coord_repr_diff_types(icrs_coord)
    return _get_matrix_vectors(gal_frame)


def to_icrs(gal_coord, icrs_frame):
    _check_coord_repr_diff_types(gal_coord)
    return _get_matrix_vectors(gal_coord, inverse=True)


@frame_transform_graph.transform(AffineTransform, ICRS, ExternalGalaxyFrame)
def icrs_to_gal(*args):
    return from_icrs(*args)


@frame_transform_graph.transform(AffineTransform, ExternalGalaxyFrame, ICRS)
def gal_to_icrs(*args):
    return to_icrs(*args)


class M31Frame(ExternalGalaxyFrame):
    """Coordinate frame representing M31.\n"""
    __doc__ = __doc__ + ExternalGalaxyFrame.__doc__+"""
    (Careful here - galaxy x and y coordinates as defined in various older papers are
    inconsistent with the convention here, with x and y swapped/reversed.
    Data sources:
        coordinates: 00 42 44.330  41 16 07.50, Skrutskie+ 2006 via Simbad
        distance: 770 kpc, van der Marel+ 2012   note error ~ 25 kpc
        radial velocity: -300 km/s, McConnachie 2012 via Simbad
        PM: pmra (with cosdec factor), pmdec = 49, -38 microarcsec/yr, van der Marel+ 2019, HST+Gaia
           note error ~ 11 muas/yr leading to ~40 km/s velocity errors
        inclination: 77 deg, Simien+ 1978 (NW side closer, our view gives "underside")
        PA: 37 deg, de Vaucouleurs 1958
    """
    def __init__(self, *args, **kwargs):
        # these values for comparison with vdm12 transformation - not necessarily current
        default_params = {
            'gal_coord': ICRS(ra=10.68470833 * u.degree, dec=41.26875 * u.degree),
            "gal_distance": 770.0 * u.kpc,
            "galvel_heliocentric": r.CartesianDifferential([178.9, -138.8, -301.] * (u.km / u.s)),
            "inclination": (-77.) * u.degree,
            "PA": 37. * u.degree,
        }
        kwds = dict()
        kwds.update(default_params)
        kwds.update(kwargs)
        super().__init__(*args, **kwds)


@frame_transform_graph.transform(AffineTransform, ICRS, M31Frame)
def icrs_to_gal(*args):
    return from_icrs(*args)


@frame_transform_graph.transform(AffineTransform, M31Frame, ICRS)
def gal_to_icrs(*args):
    return to_icrs(*args)


class M33Frame(ExternalGalaxyFrame):
    """Frame representing M33.\n"""
    __doc__ = __doc__ + ExternalGalaxyFrame.__doc__+"""
    (Careful here - galaxy x and y coordinates as defined in various older papers are
    inconsistent with the convention here, with x and y swapped/reversed."""
    def __init__(self, *args, **kwargs):
        # these values for comparison with vdm12 transformation - not necessarily current
        default_params = {
            'gal_coord': ICRS(ra=23.4621 * u.degree, dec=30.65994167 * u.degree),
            "gal_distance": 794.0 * u.kpc,
            # "gal_distance": 810.0 * u.kpc,
            "galvel_heliocentric": r.CartesianDifferential([87.3, 28.2, -180.] * (u.km / u.s)),
            "inclination": (-55.) * u.degree,
            "PA": 23. * u.degree,
        }
        kwds = dict()
        kwds.update(default_params)
        kwds.update(kwargs)
        super().__init__(*args, **kwds)


@frame_transform_graph.transform(AffineTransform, ICRS, M33Frame)
def icrs_to_gal(*args):
    return from_icrs(*args)


@frame_transform_graph.transform(AffineTransform, M33Frame, ICRS)
def gal_to_icrs(*args):
    return to_icrs(*args)


class LMCFrame(ExternalGalaxyFrame):
    """Frame representing the LMC.\n"""
    __doc__ = __doc__ + ExternalGalaxyFrame.__doc__+"""
    Caution: x and y are not necessarly specified in accordance with any convention to the literature,
    but instead follow the general ExternalGalaxyFrame convention."""
    def __init__(self, *args, **kwargs):
        # based on Luri et al 2020 including "main" line in table 5
        default_params = {
            'gal_coord': ICRS(ra=81.28 * u.degree, dec=-69.78 * u.degree),
            "gal_distance": 49.5 * u.kpc,
            "galvel_heliocentric": r.CartesianDifferential([439.0, 91.7, 262.2] * (u.km / u.s)),
            "inclination": 34.08 * u.degree,
            "PA": 309.9 * u.degree,
        }
        kwds = dict()
        kwds.update(default_params)
        kwds.update(kwargs)
        super().__init__(*args, **kwds)


@frame_transform_graph.transform(AffineTransform, ICRS, LMCFrame)
def icrs_to_gal(*args):
    return from_icrs(*args)


@frame_transform_graph.transform(AffineTransform, LMCFrame, ICRS)
def gal_to_icrs(*args):
    return to_icrs(*args)


def coords2xieta(c, c0):
    """Convert an astropy skycoord to tangent-plane coordinates in degrees.

    Uses scalar skycoord c0 to set center and orientation of tangent plane.

    Parameters:
        c: Scalar or vector SkyCoord or Frame
            Sky positions of
        c0: Scalar SkyCoord or Frame
            Sky center of coordinate system
    Returns:
        xi, eta: floats, in units of degrees (NOT astropy quantity)
        Tangent-plane coordinates centered on c0 for specified angular positions
    """

    # At one point implemented a frame to do this more directly (fewer trig functions),
    # but couldn't make it play nicely with astropy
    # should check c0 here
    frame = SkyOffsetFrame(origin=c0)
    c_lonlat = c.transform_to(frame=frame)
    lon_rad, lat_rad = c_lonlat.lon.radian, c_lonlat.lat.radian
    xi = np.tan(lon_rad)
    eta = np.tan(lat_rad) / np.cos(lon_rad)
    xi = np.rad2deg(xi)
    eta = np.rad2deg(eta)
    return xi, eta


def xieta2coords(xi, eta, c0):
    """Convert tangent-plane coordinates in degrees to an astropy skycoord.

    Uses scalar skycoord c0 to set center and orientation of tangent plane.
    Output coordinate will be in same frame as c0.

    Parameters:
        xi, eta: floats, units of degrees (NOT astropy quantity)
        c0: Scalar SkyCoord or Frame
            Sky center of coordinate system
    Returns:
        SkyCoord containing angular positions, transformed to same frame as c0 (i.e. ICRS or Galactic)
    """
    # At one point implemented a frame to do this more directly (fewer trig functions),
    # but couldn't make it play nicely with astropy
    # should check c0 here
    # frame0 = c0.
    frame = SkyOffsetFrame(origin=c0)
    xi_rad = np.deg2rad(xi)
    eta_rad = np.deg2rad(eta)
    L = np.arctan(xi_rad)
    B = np.arcsin(eta_rad / np.sqrt(1. + xi_rad**2 + eta_rad**2))
    c_lonlat = SkyCoord(L, B, unit=2*(u.rad,), frame=frame)
    c = c_lonlat.transform_to(c0.frame)
    return c


def diskcoords(c_data, extgalframe, linear=False):
    """Transformation of sky coordinates to coordinates in disk plane of provided galaxy frame.

    Given incomplete positional information (just sky positions, no distance)
    as coordinate object, and an external galaxy frame instance, return galaxy frame
    spatial coordinates x, y, z inferred to be in disk plane, where z=0 for all points.
    Recall that x points along minor axis and y along major axis, by means of PA definition.
    The transformation is nonlinear, so Newton's method is used to iterate to convergence.
    This may slow the solution, especially for galaxies with large angular sizes.

    Parameters:
        c_data: Scalar or Vector-valued SkyCoord or Frame with data
            Heliocentric sky coordinates of desired points
        extgalframe: ExternalGalaxyFrame instance
            Galaxy frame in which to put coordinates
    """

    # assert extgalframe isinstance ExternalGalaxyFrame
    if isinstance(c_data, BaseCoordinateFrame):
        c_data = SkyCoord(c_data)
    c0 = SkyCoord(extgalframe.gal_coord)  # this coord should be in icrs to match PA usage
    skyframe = SkyOffsetFrame(origin=c0)
    c_lonlat = c_data.transform_to(frame=skyframe)
    lon_rad, lat_rad = c_lonlat.lon.radian, c_lonlat.lat.radian

    # make initial guess from small-angle approx
    pa = extgalframe.PA.to(u.radian).value
    inc = extgalframe.inclination.to(u.radian).value
    distance = extgalframe.gal_distance
    xsky = lon_rad * distance
    ysky = lat_rad * distance
    xalign = np.cos(pa) * xsky - np.sin(pa) * ysky
    yalign = np.sin(pa) * xsky + np.cos(pa) * ysky
    xdisk = xalign / np.cos(inc)
    ydisk = yalign
    zdisk = xdisk * 0
    if linear:
        return xdisk, ydisk, zdisk  # small-angle approx, no refinement

    # refine xdisk, ydisk positions until agree with lon, lat using Newton's method
    dx = 1.e-5 * u.kpc  # this should work for nearby galaxies, for distant ones may be too small
    target = np.stack([lon_rad, lat_rad], axis=1)
    precision = max(dx, 1.e-5 * np.abs(xdisk).max(), 1.e-5 * np.abs(ydisk).max())
    input = np.stack([xdisk, ydisk], axis=1)
    converged = False
    for i in range(10):
        xdisk, ydisk = input.T
        c_test = SkyCoord(x=xdisk, y=ydisk, z=zdisk, frame=extgalframe)
        c_lonlat = c_test.transform_to(frame=skyframe)
        lon_rad0, lat_rad0 = c_lonlat.lon.radian, c_lonlat.lat.radian
        output = np.stack([lon_rad0, lat_rad0], axis=1)
        c_test = SkyCoord(x=xdisk + dx, y=ydisk, z=zdisk, frame=extgalframe)
        c_lonlat = c_test.transform_to(frame=skyframe)
        lon_rad1, lat_rad1 = c_lonlat.lon.radian, c_lonlat.lat.radian
        doutput1 = np.stack([lon_rad1, lat_rad1], axis=1) - output
        c_test = SkyCoord(x=xdisk, y=ydisk + dx, z=zdisk, frame=extgalframe)
        c_lonlat = c_test.transform_to(frame=skyframe)
        lon_rad2, lat_rad2 = c_lonlat.lon.radian, c_lonlat.lat.radian
        doutput2 = np.stack([lon_rad2, lat_rad2], axis=1) - output
        djac = np.stack([doutput1, doutput2], axis=2) / dx
        last_input = input.copy()
        input = input - np.einsum('...ij,...j->...i', np.linalg.inv(djac), (output - target))
        dmax = np.abs(input - last_input).max()
        print('iter: ', i, dmax)  # test
        if dmax < precision:
            converged = True
            break
    if not converged:
        print('WARNING: not converged, max offset', dmax)

    return xdisk, ydisk, zdisk


def get_galvel(pmra, pmdec, radial_velocity, distance):
    """Utility routine to retrieve a CartesianDifferential from proper motion, velocity, and distance.

    This is useful to initialize an ExternalGalaxyFrame, by passing the keyword galvel_heliocentric=get_galvel(
         1.0*u.mas/u.yr, 2.0*u.mas/u.yr, radial_velocity=250.*u.km/u.s, distance=400.*u.kpc)
    Parameters:
        pmra, pmdec: Quantity dimension of angle/time
            Proper motion in RA, dec directions. Here pmra means metric and not coordinate velocity
            (i.e., cos factor should be included)
        radial_velocity: Quantity, dimension of velocity
            Heliocentric radial velocity
        distance: Quantity, dimension of length
            Heliocentric distance
    """
    v_x = (pmra * distance / u.rad).to(u.km / u.s)
    v_y = (pmdec * distance / u.rad).to(u.km / u.s)
    v_z = radial_velocity.to(u.km / u.s)
    return r.CartesianDifferential(v_x, v_y, v_z)
