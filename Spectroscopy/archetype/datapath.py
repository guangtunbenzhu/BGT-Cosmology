"""
Path to various datasets
Need to clean up once the package is well rewritten
"""

from os import getenv
from os.path import join

# This should be moved to __inti__.py in the future
_parent_path = getenv('ASTRODATA', '/Users/Benjamin/AstroData')

# HSTFOS
def hstfos_path():
    """Path to HST FOS data
    """
    return join(_parent_path, 'HST/FOS/')

# All in One
def allinone_path():
    """Path to all_in_one files 
    """
    return join(_parent_path, 'AllInOne')

# Quasars
def qso_path():
    """Path to quasar information
    """
    return join(_parent_path, 'Quasars')

# SDSS
def sdss_path():
    """Path to SDSS spectroscopic/imaging/photometric data
    """
    return join(_parent_path, 'SDSS')

# Lines
def lines_path():
    """Path to absorption/emission lines
    """
    return join(_parent_path, 'Lines')

# Dust Templates
def sed_template_path():
    """Path to SED templates
    """
    return join(_parent_path, 'SEDtemplates')

# Dust Maps
def dustmap_path():
    """Path to Dust
    """
    return join(_parent_path, 'Dust/SFD')

# NMF
def nmf_path():
    """Path to Dust
    """
    return join(_parent_path, 'NMF')

# Absorbers
def absorber_path():
    """Path to Dust
    """
    return join(_parent_path, 'Absorbers')

# atomic
def atomic_path():
    """Path to Dust
    """
    return join(_parent_path, 'Atomic')

