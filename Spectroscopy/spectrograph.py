
# Class Spectrograph
class Spectrograph:
   """A spectrograph.

   These objects are spectrographs used in physics and astronomy
   """

   __slots__ = ('_name', '_telescope', '_instrument', '_minwave', '_maxwave', '_width', '_resolution')

   def __init__(self, name, instrument, telescope, minwave, maxwave, width, resolution):
       self._name = name
       self._instrument = instrument 
       self._telescope = telescope
       self._minwave = minwave
       self._maxwave = maxwave
       self._width = width
       self._resolution = resolution

   def __repr__(self):
       return ('<Spectrograph name={0!r} instrument={1!r} telescope={2!r} minwave={3} maxwave={4} '
               ' width={5} resolution={6}>'.format(self._name, self._instrument, self._telescope, self._minwave, 
               self._maxwave, self._width, self._resolution))

   def __str__(self):
       return (' Name = {0}\n'
               ' Instrument = {1}\n'
               ' Telescope = {2}\n'
               ' Minwave = {3} Ang\n'
               ' Maxwave = {4} Ang\n'
               ' width = {5} Ang\n'
               ' Resolution = {6}'.format(self._name, self._instrument, self._telescope, self._minwave, 
               self._maxwave, self._width, self._resolution))

   @property
   def name(self):
       """The full name of the spectrograph."""
       return self._name

   @property
   def instrument(self):
       """The full name of the instrument."""
       return self._instrument

   @property
   def telesceop(self):
       """The full name of the telescope."""
       return self._telescope
# Done Class Spectrograph 

# HST FOS: http://ned.ipac.caltech.edu/level5/Golombek/Golombek5_6.html
FOSG130H = Spectrograph('G130H', 'FOS', 'HST', 1140., 1606., 1.0, 1300.)
FOSG190H = Spectrograph('G190H', 'FOS', 'HST', 1573., 2330., 1.45, 1300.)
FOSG270H = Spectrograph('G270H', 'FOS', 'HST', 2221., 3301., 2.05, 1300.)

# SDSS
SDSS = Spectrograph('SDSS', 'SDSS', 'SDSS', 3800., 9200., 3.0, 1800.)
BOSS = Spectrograph('BOSS', 'SDSS', 'SDSS', 3650., 10400., 3.0, 1800.)
