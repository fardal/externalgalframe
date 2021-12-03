
A module for specifying an astropy coordinate frame of external galaxy, along with some other utilities. Probably has suboptimal coding and use of astropy. M31, M33, and the LMC are provided as example frames. Some examples are in examples.py. For example, to put observed coordinates into the M31 disk-aligned system you can do something like:

```
import units as u
from astropy.coordinates import SkyCoord
from externalgalframe.externalgalaxy import M31Frame
c = SkyCoord(ra=ra, dec=dec, distance=distance,
             radial_velocity=vrad, pm_ra_cosdec=pmra, pm_dec=pmdec, frame='icrs')
frame = M31Frame()
cgal = c.transform_to(frame)
xg, yg, zg, vxg, vyg, vzg = cgal.x, cgal.y, cgal.z, cgal.v_x, cgal.v_y, cgal.v_z
```
    
	
