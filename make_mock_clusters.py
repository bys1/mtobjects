import os
from astropy.wcs import WCS
from scipy import signal
from astropy.io.fits import open as get_fits
import numpy as np
import scipy as sp
from astropy.io import fits


# ----------------------------------------------------------------------
def get_plane_distance(data, xc, yc):
    nx, ny = len(data[0]), len(data)
    x, y = np.ogrid[-yc:float(ny) - yc, -xc:float(nx) - xc]
    return np.sqrt(x ** 2. + y ** 2.)


# ----------------------------------------------------------------------
def gaussian(r, sigma, mu0):
    return np.sqrt(mu0 ** 2.) * np.exp(-0.5 * (r / sigma) ** 2.)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def exponential(r, h, mu0):
    return mu0 * np.exp(-r / h)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def moffat(r, I0, alfa, beta):
    '''
    Returns moffat intensity with given parameters and positions.
    '''
    mof = np.sqrt(I0 ** 2) * (1 + (r / alfa) ** 2.) ** (-beta)
    return mof


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def analytic_ocam_psf(r, mu0, mag=None):
    '''
    Returns the OmegaCAM r-band psf surface brightness on a given distance 
    (in arcsecs and magnitudes).
    '''
    # Relation between mag_auto and mu0 for stars:
    r = r * 5.
    coef = np.array([2.10164715e-02, 2.30883062e-11])  # for 1026 r band stars in field 11
    p = np.poly1d(coef)
    if mag:
        mu0 = p(10. ** (mag / (-2.5)))
    else:
        mu0 = 1.
    gauss_mu0, gauss_sigma = 0.061, 0.757 * 5.
    mof_I0, mof_alpha, mof_beta = 0.938, 0.635 * 5., 1.60
    exponential_h, exponential_I0 = 74.26 * 5., 6.022e-06

    gaussian_mod = gaussian(r, gauss_mu0, gauss_sigma) * mu0
    moffat_mod = moffat(r, mof_I0, mof_alpha, mof_beta) * mu0
    exponential_mod = exponential(r, exponential_h, exponential_I0) * mu0
    psf = gaussian_mod + moffat_mod + exponential_mod
    return psf


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def surface_brightness(flux, pix_scale=0.2):
    '''
    Returns surface brightness for a pixel.
    pix_scale       :     [float] Size of a pixel in asecs
    '''
    return -2.5 * np.log10(flux) + 2.5 * np.log10(pix_scale ** 2.)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def find_ocam_psf_sfb_rad(mag, sfb):
    r = 0
    ok = False
    while not ok:
        r += 1.
        sb = analytic_ocam_psf(r / 5., 0., mag=mag)
        if surface_brightness(sb) > sfb:
            return r
        if r > 8000:
            return 0


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def sersic(mue, n, r, reff, flux=False, mag=None):
    '''
    Returns sersic profile values for a given parameters.
    Flux: True/False for values in fluxes/magnitudes.
    '''
    # Macarthur approoximation:
    if n > 0.36:
        k = 2.0 * n - 1.0 / 3. + 4.0 / 405. / n + 46.0 / 25515. / n ** 2. + 131.0 / 1148175. / n ** 3. + 2194697. / 30690717750.0 / n ** 4.
    else:
        k = 0.01945 - 0.8902 * n + 10.95 * n ** 2. - 19.67 * n ** 3. + 13.43 * n ** 4.
    if not flux:
        profile = mu0 + 2.5 * k * (r / reff) ** (1. / n)
    else:
        if mag == None:
            profile = mue * np.exp(-k * ((r / reff) ** (1. / n) - 1.))
        else:
            mue = 10. ** (-mag / 2.5) / (
                        2. * np.pi * reff ** 2. * np.exp(k) * n * k ** (-2. * n) * sp.special.gamma(2. * n))
            profile = mue * np.exp(-k * ((r / reff) ** (1. / n) - 1.))
    return profile


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def make_galaxy(mue, reff, n, nx, arat, pa):
    '''
    Return a galaxy made with a given Sersic indices
    '''
    yc, xc = int(nx / 2), int(nx / 2)
    ny = nx
    # Define elliptical coordinates
    x, y = np.ogrid[-yc:float(ny) - yc, -xc:float(nx) - xc]
    angles = np.arctan(y / x)
    angles = np.pi / 2. - angles
    angles[:yc, :] = angles[:yc, :] + np.pi
    angles[yc, :xc] = np.pi
    distance = np.sqrt(x ** 2. + y ** 2.) / reff
    d, a = distance, angles
    A = 1.
    B = arat
    edist = d / (B / np.sqrt((B * np.cos(a - pa)) ** 2. + (A * np.sin(a - pa)) ** 2.))
    edist[yc, xc] = 0.
    edist2 = (d * reff) / (B / np.sqrt((B * np.cos(a - pa)) ** 2. + (A * np.sin(a - pa)) ** 2.))
    edist2[yc, xc] = 0.
    r = edist.reshape(nx * nx)
    flux = sersic(mue, n, r, 1., flux=True)  # / arat
    flux.resize(nx, nx)
    print(flux[yc,xc:])
    return flux


# ----------------------------------------------------------------------


### Make ten clusters ###
nclust = 1  # how many images to make
bg_noise = 1.1e-12  # background noise sigma
gain = 52062244760000.0  # gain gain*flux = Nelectron to CCD   #f[0].header['GAIN']
ncgal = 1  # Number of cluster galaxies
nbgal = 0  # number of background galaxies
nbris = 0  # number of stars
limbris = 15  # Limit for bright stars

for i in range(nclust):
    ### make arrays ###
    cgx, cgy, cgmu, cgreff, cgarat, cgn, cgpa = [], [], [], [], [], [], []
    bgx, bgy, bgmu, bgreff, bgarat, bgn, bgpa = [], [], [], [], [], [], []
    sx, sy, smag = [], [], []
    ### copy image ###
    name = 'cluster' + str(i + 1) + '.fits'
    print(name)
    print(' Adding galaxies...')
    os.system('cp template.fits ' + name)
    f = get_fits(name)
    f[0].data = np.zeros(f[0].data.shape).astype('float32')
    maxy = len(f[0].data) - 1
    maxx = len(f[0].data[0]) - 1
    w = WCS(name)
    ### Add cluster galaxies ###
    for g in range(ncgal):
        # generate galaxies with properties in the given range
        mue = 10. ** (-11.)# - np.random.random() * 4.)  # surface brightness at Re
        reff = 100.#(1. + np.random.random() * 99.) * 5.  # Re in pixel_s

        n = 8.0 #0.5 + np.random.random() * 1.5  # Sersic index
        
        nx = int(10 * reff * n)  # how far is the galaxy light profile extended
        arat = 0.3 + np.random.random() * 0.7  # axis ratio
        pa = np.random.random() * np.pi  # position angle in radians
        gal = make_galaxy(mue, reff, n, nx, arat, pa)  # generate Sersic profile
        x = int(0.5 * maxx)  # randomly position the galaxy to the image
        y = int(0.5 * maxy)
        xe, ye = gal.shape
        xe, ye = xe / 2., ye / 2.
        if xe % 1 == 0.5:
            des = 0.5
        else:
            des = 0
        # Reshape the galaxy array to fit the image
        y0 = np.max([int(y - ye - des), 0])
        y1 = np.min([int(y + ye - des), maxy + 1])
        x0 = np.max([int(x - xe - des), 0])
        x1 = np.min([int(x + xe - des), maxx + 1])
        if y0 == 0:
            gal = gal[0 - int(y - ye - des):]
        if x0 == 0:
            gal = gal[:, 0 - int(x - xe - des):]
        if y1 == maxy + 1:
            gal = gal[:maxy - int(y + ye - des)]
            y1 -= 1
        if x1 == maxx + 1:
            gal = gal[:, :maxx - int(x + xe - des)]
            x1 -= 1
        # Add the generated galaxy
        f[0].data[y0:y1, x0:x1] += gal
        # Collect the randomly generated structural parameters into a table
        cgx.append(x)
        cgy.append(y)
        cgmu.append(mue)
        cgreff.append(reff)
        cgarat.append(arat)
        cgn.append(n)
        cgpa.append(pa)

    ### Add background galaxies ###
    for g in range(nbgal):
        mue = 10. ** (-11. - np.random.random() * 3.)
        reff = (0.5 + np.random.random() * 3.) * 5.
        n = 2. + np.random.random() * 2.
        nx = int(10 * reff * n)
        arat = 0.3 + np.random.random() * 0.7
        pa = np.random.random() * np.pi
        gal = make_galaxy(mue, reff, n, nx, arat, pa)
        x = int(np.random.random() * maxx)
        y = int(np.random.random() * maxy)
        xe, ye = gal.shape
        xe, ye = xe / 2., ye / 2.
        if xe % 1 == 0.5:
            des = 0.5
        else:
            des = 0
        y0 = np.max([int(y - ye - des), 0])
        y1 = np.min([int(y + ye - des), maxy + 1])
        x0 = np.max([int(x - xe - des), 0])
        x1 = np.min([int(x + xe - des), maxx + 1])
        if y0 == 0:
            gal = gal[0 - int(y - ye - des):]
        if x0 == 0:
            gal = gal[:, 0 - int(x - xe - des):]
        if y1 == maxy + 1:
            gal = gal[:maxy - int(y + ye - des)]
            y1 -= 1
        if x1 == maxx + 1:
            gal = gal[:, :maxx - int(x + xe - des)]
            x1 -= 1
        f[0].data[y0:y1, x0:x1] += gal
        bgx.append(x)
        bgy.append(y)
        bgmu.append(mue)
        bgreff.append(reff)
        bgarat.append(arat)
        bgn.append(n)
        bgpa.append(pa)

    ### Convolve  and add noise###
    # Poisson
    print('Convolving')
    h = get_plane_distance(np.zeros([81, 81]), 40, 40) * 0.2  # generate distance plane
    h = analytic_ocam_psf(h, 1, mag=None)  # Make psf model using analytic function
    h = h / np.sum(h)  # Normalize psf
    np.save('mock/psf.npy', h)
    f[0].data = signal.fftconvolve(f[0].data, h, mode='same')  # Run convolution
    print('Adding Poisson noise')
    f[0].data[f[0].data < 0] = 0.  # Remove negative pizxels in casse FFT left such artefacts

    ### Add stars ###
    nbri = 0
    print('Adding stars...')
    while nbri < nbris:
        mag = 26 - np.random.exponential(scale=3)
        if mag < 13.5: continue  # exclude saturated stars
        rad = int(find_ocam_psf_sfb_rad(mag, surface_brightness(5e-14)))  # Define how far the star profile is extendded
        star = np.zeros([2 * rad, 2 * rad])
        star = get_plane_distance(star, rad, rad) * 0.2
        star = analytic_ocam_psf(star, 1, mag=mag)
        x = int(np.random.random() * maxx)
        y = int(np.random.random() * maxy)
        xe, ye = star.shape
        xe, ye = xe / 2., ye / 2.
        if xe % 1 == 0.5:
            des = 0.5
        else:
            des = 0
        y0 = np.max([int(y - ye - des), 0])
        y1 = np.min([int(y + ye - des), maxy + 1])
        x0 = np.max([int(x - xe - des), 0])
        x1 = np.min([int(x + xe - des), maxx + 1])
        if y0 == 0:
            star = star[0 - int(y - ye - des):]
        if x0 == 0:
            star = star[:, 0 - int(x - xe - des):]
        if y1 == maxy + 1:
            star = star[:maxy - int(y + ye - des)]
            y1 -= 1
        if x1 == maxx + 1:
            star = star[:, :maxx - int(x + xe - des)]
            x1 -= 1
        f[0].data[y0:y1, x0:x1] += star
        sx.append(x)
        sy.append(y)
        smag.append(mag)
        if mag < limbris: nbri += 1

    # In the end everything is detected as photons, so add poisson noise here.
    f[0].data = np.random.poisson(f[0].data * gain) / gain
    # Add background noise 
    f[0].data += np.random.normal(loc=0., scale=bg_noise, size=f[0].data.shape)
    f[0].data = f[0].data.astype('float32')
    f.writeto('mock/' + name, overwrite=True)

    cgx, cgy, cgmu, cgreff, cgarat, cgn, cgpa = np.array(cgx), np.array(cgy), np.array(cgmu), np.array(
        cgreff), np.array(cgarat), np.array(cgn), np.array(cgpa)
    bgx, bgy, bgmu, bgreff, bgarat, bgn, bgpa = np.array(bgx), np.array(bgy), np.array(bgmu), np.array(
        bgreff), np.array(bgarat), np.array(bgn), np.array(bgpa)
    sx, sy, smag = np.array(sx), np.array(sy), np.array(smag)

    c1 = fits.Column(name='x', format='E', array=np.array(cgx))
    c2 = fits.Column(name='y', format='E', array=np.array(cgy))
    c3 = fits.Column(name='mu', format='E', array=np.array(cgmu))
    c4 = fits.Column(name='Reff', format='E', array=np.array(cgreff))
    c5 = fits.Column(name='arat', format='E', array=np.array(cgarat))
    c6 = fits.Column(name='n', format='E', array=np.array(cgn))
    c7 = fits.Column(name='pa', format='E', array=np.array(cgpa))
    coldefs = fits.ColDefs([c1, c2, c3, c4, c5, c6, c7])
    bhdu = fits.BinTableHDU.from_columns(coldefs)
    bhdu.writeto('mock/cluster' + str(i + 1) + '_gal.fits', overwrite=True)

    c1 = fits.Column(name='x', format='E', array=np.array(bgx))
    c2 = fits.Column(name='y', format='E', array=np.array(bgy))
    c3 = fits.Column(name='mu', format='E', array=np.array(bgmu))
    c4 = fits.Column(name='Reff', format='E', array=np.array(bgreff))
    c5 = fits.Column(name='arat', format='E', array=np.array(bgarat))
    c6 = fits.Column(name='n', format='E', array=np.array(bgn))
    c7 = fits.Column(name='pa', format='E', array=np.array(bgpa))
    coldefs = fits.ColDefs([c1, c2, c3, c4, c5, c6, c7])
    bhdu = fits.BinTableHDU.from_columns(coldefs)
    bhdu.writeto('mock/cluster' + str(i + 1) + '_bgal.fits', overwrite=True)

    c1 = fits.Column(name='x', format='E', array=np.array(sx))
    c2 = fits.Column(name='y', format='E', array=np.array(sy))
    c3 = fits.Column(name='mag', format='E', array=np.array(smag))
    coldefs = fits.ColDefs([c1, c2, c3])
    bhdu = fits.BinTableHDU.from_columns(coldefs)
    bhdu.writeto('mock/cluster' + str(i + 1) + '_star.fits', overwrite=True)


