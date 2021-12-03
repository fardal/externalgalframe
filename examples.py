"""Examples of using external galaxy classes"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, ICRS, Galactocentric, representation
from astropy.coordinates.representation import CartesianDifferential
import astropy.units as u
from astropy.table import Table
Table.__iter__ = Table.itercols


from externalgalframe_revise.externalgalaxy import M31Frame

def get_m31galframe():
    return M31Frame(gal_distance=780. * u.kpc,
                    galvel_heliocentric=CartesianDifferential([125.2, -73.8, -300.] * (u.km / u.s)))


def gethalodistn():
    """Construct a toy model halo around M31.

    Here the M31-centric velocity is set to zero.
    Spatial distribution follows a 1/r density law."""
    rng = np.random.default_rng(12345)
    uniform = rng.uniform
    normal = rng.normal

    n = 1000
    mu = uniform(-1., 1., n)
    phi = uniform(0., 2.*np.pi, n)
    rmax = 250.
    t = uniform(0., 1., n)
    r = np.sqrt(t) * rmax
    costheta = mu
    sintheta = np.sqrt(1. - mu**2)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    x = sintheta * cosphi * r * u.kpc
    y = sintheta * sinphi * r * u.kpc
    z = costheta * r * u.kpc
    kms = u.km / u.s
    # vx = np.zeros(n) * kms
    # vy = np.zeros(n) * kms
    # vz = np.zeros(n) * kms
    sighalo = 150. * kms
    vx = normal(size=n) * sighalo
    vy = normal(size=n) * sighalo
    vz = normal(size=n) * sighalo
    frame = get_m31galframe()
    print(x[0], y[0], z[0], vx[0], vy[0], vz[0])
    c = SkyCoord(x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz, frame=frame)
    # plt.plot(x, y, 'k.')
    # foo = plt.hist(r, bins=30)
    # plt.show()
    return c


def writehalo():
    cgal = gethalodistn()
    tab = Table()
    tab['x'] = cgal.x.to_value(u.kpc)
    tab['y'] = cgal.y.to_value(u.kpc)
    tab['z'] = cgal.z.to_value(u.kpc)
    tab['vx'] = cgal.v_x.to_value(u.km / u.s)
    tab['vy'] = cgal.v_y.to_value(u.km / u.s)
    tab['vz'] = cgal.v_z.to_value(u.km / u.s)
    tab.write('halo_gal.dat', format='ascii.fixed_width', overwrite=True)


def writehalo_obs():
    cgal = gethalodistn()
    c = cgal.transform_to('icrs')
    tab = Table()
    tab['ra'] = c.ra.to_value(u.deg)
    tab['dec'] = c.dec.to_value(u.deg)
    tab['distance'] = c.distance.to_value(u.kpc)
    tab['vrad'] = c.radial_velocity.to_value(u.km / u.s)
    tab['pmra'] = c.proper_motion[0].to_value(u.mas / u.yr)
    tab['pmdec'] = c.proper_motion[1].to_value(u.mas / u.yr)
    tab.write('halo_obs.dat', format='ascii.fixed_width', overwrite=True)


def readhalo(as_table=False):
    tab = Table.read('halo_gal.dat', format='ascii.fixed_width')
    if as_table:
        return tab
    x = tab['x'] * u.kpc
    y = tab['y'] * u.kpc
    z = tab['z'] * u.kpc
    vx = tab['vx'] * u.km / u.s
    vy = tab['vy'] * u.km / u.s
    vz = tab['vz'] * u.km / u.s
    frame = get_m31galframe()
    c = SkyCoord(x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz, frame=frame)
    return c


def readhalo_obs(as_table=False):
    tab = Table.read('halo_obs.dat', format='ascii.fixed_width')
    if as_table:
        return tab
    ra = tab['ra'] * u.deg
    dec = tab['dec'] * u.deg
    distance = tab['distance'] * u.kpc
    vrad = tab['vrad'] * u.km / u.s
    pmra = tab['pmra'] * u.mas / u.yr
    pmdec = tab['pmdec'] * u.mas / u.yr
    c = SkyCoord(ra=ra, dec=dec, distance=distance,
                 radial_velocity=vrad, pm_ra_cosdec=pmra, pm_dec=pmdec, frame='icrs')
    return c


def example_obs2gal():
    """Example of loading observed coords and transforming to M31-centric coords"""
    c = readhalo_obs()  # a SkyCoord object
    frame = get_m31galframe()
    cgal = c.transform_to(frame)
    return cgal


def test_roundtrip():
    """Test whether transforming forward and back yields same results."""
    cgal_0 = readhalo()
    cgal = example_obs2gal()
    print('check differences small:')
    print((cgal.x - cgal_0.x).std(),
          (cgal.y - cgal_0.y).std(),
          (cgal.z - cgal_0.z).std(),
          (cgal.v_x - cgal_0.v_x).std(),
          (cgal.v_y - cgal_0.v_y).std(),
          (cgal.v_z - cgal_0.v_z).std())


def showhalodistn():
    cgal = gethalodistn()
    c0 = cgal.frame.gal_coord
    # print(dir(frame))
    # print(frame.gal_coord)
    c = cgal.transform_to('icrs')

    # plt.plot(np.mod(c.ra.deg+180,360)-180, c.dec.deg, 'k.')
    # plt.plot(c0.ra.deg, c0.dec.deg, 'ro')
    # plt.show()

    # xi, eta = coords2xieta(c, c0)
    # plt.plot(xi, eta, 'k.')
    # plt.show()

    # print(c)
    pmra, pmdec = c.proper_motion
    plt.plot(pmra, pmdec, 'k.')
    print(pmra[0])
    plt.show()



def do():
    c = example_obs2gal()
