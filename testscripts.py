

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, ICRS, Galactocentric, representation
import astropy.units as u


from externalgalaxy import M31Frame, M33Frame, LMCFrame, diskcoords, get_galvel, xieta2coords, coords2xieta


def test_m31frame_roundtrip():
    x = np.array([0., 1., 0., 0.])
    y = np.array([0., 0., 1., 0.])
    z = np.array([0., 0., 0., 1.])
    vx = x * 0.
    vy = x * 0.
    vz = x * 0.

    kms = u.km / u.s
    frame = M31Frame(x=x*u.kpc, y=y*u.kpc, z=z*u.kpc, v_x=vx*kms, v_y=vy*kms, v_z=vz*kms)
    c = SkyCoord(frame)
    print(c)
    c_icrs = c.transform_to('icrs')
    c2 = c_icrs.transform_to(M31Frame)
    print('c_icrs: ', c_icrs)
    print('c: ', c)
    print('c2: ', c2)


def test_m31galactocentric():
    """Try to reproduce the transformation of M31 velocity from heliocentric to galactocentric in vdm12"""
    kms = u.km / u.s
    c0 = SkyCoord(x=0.*u.kpc, y=0.*u.kpc, z=0.*u.kpc, v_x=0.*kms, v_y=0.*kms, v_z=0.*kms,
                  gal_distance=770.*u.kpc,
                  galvel_heliocentric=representation.CartesianDifferential([125.2, -73.8, -301.] * kms),
                  frame=M31Frame)
    print(c0)
    c_i = c0.transform_to(ICRS)
    print('proper motion:')
    print(c_i.proper_motion)

    gcframe = Galactocentric(galcen_distance=8.29 * u.kpc, galcen_v_sun=(11.1, 239. +12.24, 7.25) * kms,
                             z_sun=0. * u.kpc, roll=0. * u.deg)
    c = c0.transform_to(gcframe)
    answer = c.velocity.to_cartesian().xyz.value
    # print(c)
    # print(c.velocity)
    # print(type(c.velocity))
    # import pdb; pdb.set_trace()
    # return
    # print(np.array(c.velocity))
    vdm12 = np.array([66.1, -76.3, 45.1])
    print('velocity:')
    print(answer)
    print(answer - vdm12)

    # also do position
    answer = c.cartesian.xyz.value
    vdm12 = np.array([-378.9, 612.7, -283.1])
    print('position:')
    print(answer)
    print(answer - vdm12)


def m33velocity():
    """Calculate cartesian velocity to assign to M33, with vdm12 assumptions"""
    kms = u.km / u.s
    distance = 794. * u.kpc
    radial_velocity = -180. * kms
    # pmra = 35.5e-3 * u.mas / u.yr  # uncorrected, B05
    # pmdec = -12.5e-3 * u.mas / u.yr
    # pmra = 26.1e-3 * u.mas / u.yr  # corrected using maser motion expectation in B05
    # pmdec = -1.9e-3 * u.mas / u.yr
    # pmra = 4.7e-3 * u.mas / u.yr  # vdmG08 discussion
    # pmdec = -14.1e-3 * u.mas / u.yr
    pmra = 23.2e-3 * u.mas / u.yr  # including rot correction on p. 7 of brunthaler, but not solar term
    pmdec = 7.5e-3 * u.mas / u.yr
    print((pmra * distance / u.rad).to(kms))
    print((pmdec * distance / u.rad).to(kms))
    print(radial_velocity)


def test_m33galactocentric():
    """Try to reproduce the transformation of M33 velocity from heliocentric to galactocentric in vdm12"""
    kms = u.km / u.s

    c0 = SkyCoord(x=0.*u.kpc, y=0.*u.kpc, z=0.*u.kpc, v_x=0.*kms, v_y=0.*kms, v_z=0.*kms,
                  gal_distance=794.*u.kpc,
    #               galvel_heliocentric=representation.CartesianDifferential([133.6, -47.0, -180.] * kms))
    #               galvel_heliocentric=representation.CartesianDifferential([98.2, -7.2, -180.] * kms))
    #               galvel_heliocentric=representation.CartesianDifferential([17.7, -53.1, -180.] * kms))  no rot corr
                  galvel_heliocentric=representation.CartesianDifferential([87.3, 28.2, -180.] * kms),
                  frame=M33Frame)

    # test - closer to M31 props
    # c0 = M33Frame(x=0.*u.kpc, y=0.*u.kpc, z=0.*u.kpc, v_x=0.*kms, v_y=0.*kms, v_z=0.*kms,
    #               # gal_coord = ICRS(ra=10.68470833 * u.degree, dec=41.26875 * u.degree),
    #               gal_distance=794.*u.kpc,
    #               # galvel_heliocentric=representation.CartesianDifferential([125.2, -73.8, -301.] * kms))
    #               # galvel_heliocentric=representation.CartesianDifferential([125.2, -73.8, -180.] * kms))
    #               # galvel_heliocentric = representation.CartesianDifferential([125.2, -53.1, -180.] * kms))
    #               # galvel_heliocentric=representation.CartesianDifferential([17.7, -53.1, -180.] * kms))
    #               galvel_heliocentric=representation.CartesianDifferential([87.3, 28.2, -180.] * kms))

    print(c0)
    c_i = c0.transform_to(ICRS)
    print('proper motion:')
    print(c_i.proper_motion)
    print('radial velocity:')
    print(c_i.radial_velocity)

    gcframe = Galactocentric(galcen_distance=8.29 * u.kpc, galcen_v_sun=(11.1, 239. +12.24, 7.25) * kms,
                             z_sun=0. * u.kpc, roll=0. * u.deg)
    c = c0.transform_to(gcframe)
    answer = c.velocity.to_cartesian().xyz.value
    # print(c)
    # print(c.velocity)
    # print(type(c.velocity))
    # import pdb; pdb.set_trace()
    # return
    # print(np.array(c.velocity))
    vdm12 = np.array([43.1, 101.3, 138.8])
    print('velocity:')
    print(answer)
    print(answer - vdm12)

    # also do position
    answer = c.cartesian.xyz.value
    vdm12 = np.array([-476.1, 491.1, -412.9])
    print('position:')
    print(answer)
    print(answer - vdm12)


def lmcvelocity():
    """Calculate cartesian velocity to assign to M33, with vdm12 assumptions"""
    kms = u.km / u.s
    distance = 49.5 * u.kpc
    radial_velocity = 262.2 * kms
    pmra = 1.871 * u.mas / u.yr  # including rot correction on p. 7 of brunthaler, but not solar term
    pmdec = 0.391 * u.mas / u.yr
    print((pmra * distance / u.rad).to(kms))
    print((pmdec * distance / u.rad).to(kms))
    print(radial_velocity)


def lmcframe_points():
    x = np.array([0., 1., 0., 0.])
    y = np.array([0., 0., 1., 0.])
    z = np.array([0., 0., 0., 1.])
    c = SkyCoord(x=x, y=y, z=z, unit=3 * (u.kpc,), frame=LMCFrame)
    print(c)
    c = c.transform_to('icrs')
    print(c)
    print('cartesian: \n', c.cartesian)


def test_lmcframe_roundtrip():
    x = np.array([0., 1., 0., 0.])
    y = np.array([0., 0., 1., 0.])
    z = np.array([0., 0., 0., 1.])
    vx = x * 0.
    vy = x * 0.
    vz = x * 0.

    kms = u.km / u.s
    frame = LMCFrame(x=x*u.kpc, y=y*u.kpc, z=z*u.kpc, v_x=vx*kms, v_y=vy*kms, v_z=vz*kms)
    c = SkyCoord(frame)
    print(c)

    c_icrs = c.transform_to('icrs')
    c2 = c_icrs.transform_to(LMCFrame)
    print('c_icrs: ', c_icrs)
    print('c: ', c)
    print('c2: ', c2)


def test_deprojection():
    n = 100
    x = np.random.uniform(low=-100., high=100., size=n)
    y = np.random.uniform(low=-100., high=100., size=n)
    z = x * 0
    c = SkyCoord(x=x, y=y, z=z, unit=3 * (u.kpc,), frame=M33Frame)
    c = c.transform_to('icrs')

    xdisk, ydisk, zdisk = diskcoords(c, M33Frame())
    plt.plot(x, x, 'k,')
    plt.plot(x, xdisk, 'b.')
    plt.plot(y, ydisk, 'r.')
    plt.show()


def test_frameprop():
    """Check behavior of a specific frame is as expected - independent states for different instances"""
    frame1 = M31Frame()
    print('frame1: ', frame1)
    frame2 = M31Frame(gal_distance=800.*u.kpc)
    print('frame1: ', frame1)
    print('frame2: ', frame2)


def test_m33velocity():
    """Calculate cartesian velocity to assign to M33, with vdm12 assumptions"""
    foo = M33Frame()
    print(foo)
    kms = u.km / u.s
    distance = 794. * u.kpc
    radial_velocity = -180. * kms
    pmra = 23.2e-3 * u.mas / u.yr  # including rot correction on p. 7 of brunthaler, but not solar term
    pmdec = 7.5e-3 * u.mas / u.yr
    print((pmra * distance / u.rad).to(kms))
    print((pmdec * distance / u.rad).to(kms))
    print(radial_velocity)
    foo = M33Frame(galvel_heliocentric=get_galvel(pmra, pmdec, radial_velocity, distance))
    print(foo)


def do():
    test_m31frame_roundtrip()
    test_m31galactocentric()
    m33velocity()
    test_m33galactocentric()
    test_m33velocity()
    lmcvelocity()
    test_lmcframe_roundtrip()
    test_deprojection()
    test_frameprop()
