import fitsio
import datapath
from os.path import join

def sed_template_filename(sedtype):
    """Files for SED templates
    """
    path = datapath.sed_template_path()
    filename = 'SEDtemplate_'+sedtype.lower()+'.fits'
    return join(path, filename)

class Templates(object):
      """
      """

      __slots__ = ('_templatelist')
      _templatelist = ('quiescent', 'starforming', 'starburst', 'quasar', 'absorption', 'sky', 'type1a')

      def __init__(self): 
          pass
          
      @property
      def templatelist(cls):
          """The templates that are available
          """
          return cls._templatelist

      def read(cls, sedtype): 
          """Read in a desired template
          """
          if sedtype.lower() not in cls._templatelist:
             raise ValueError("%s is not in the availabe templates: %s".format(sedtype, cls._templatelist))

          infile = sed_template_filename(sedtype)
          print infile
          pass
          return
