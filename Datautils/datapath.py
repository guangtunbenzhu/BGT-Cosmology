
from os.path import join

_parent_path = '/Users/Benjamin/AstroData'

# SDSS
def qso_path():
    """Path for absorption/emission lines
    """
    return join(_parent_path, 'Quasars')

# SDSS
def sdss_path():
    """Path for absorption/emission lines
    """
    return join(_parent_path, 'SDSS')

# Lines
def lines_path():
    """Path for absorption/emission lines
    """
    return join(_parent_path, 'Lines')

# Dust Templates
def sed_template_path():
    """Path for SED templates
    """
    return join(_parent_path, 'SEDtemplates')

# Dust Maps
def dustmap_path():
    """Path for Dust
    """
    return join(_parent_path, 'Dust/SFD')

def sed_template_filename(sedtype)
    """Files for SED templates
    """
    path = sed_template_path()
    filename = 'SEDtemplate_'+sedtype.lower()+'.fits'
    return join(path, filename)

def dustmap_filename(maps):
    """Files for dust maps
    """
    path = dustmap_path()
    files = {'EBV': ['SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits']),
            'MASK': ['SFD_mask_4096_ngp.fits', 'SFD_mask_4096_sgp.fits']),
               'T': ['SFD_temp_4096_ngp.fits', 'SFD_temp_4096_sgp.fits']),
               'X': ['SFD_xmap_4096_ngp.fits', 'SFD_xmap_4096_sgp.fits']),
            }.get(maps.upper(),['SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits'])

    fullnames = []
    for thisfile in files: fullnames.append(os.path.join(path,thisfile))


