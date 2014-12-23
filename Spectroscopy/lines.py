# Is this really necessary?
# Define Base Class lines
class BaseLines(object):
   """Base Class for Lines

   """

   __slots__ = ('_name', '_wave', '_unit')

   def __init__(self, name, wave, unit):
       self._name = name
       self._wave = wave
       self._unit = unit

   @property
   def name(self):
       """The name of the line"""
       return self._name

   @property
   def wave(self):
       """The vacuum wavelength of the line"""
       return self._wave

   @property
   def unit(self):
       """The vacuum wavelength of the line"""
       return self._unit

class AtomicLines(BaseLines):
   """Atomic Line Class

   """

   __slots__ = ('_element', '_ionization')

   def __init__(self, name, wave, unit, element, ionization):
       super(AtomicLines, self).__init__(name, wave, unit)
       self._element = element
       self._ionization = ionization

   # Define printable representation
   def __repr__(self):
       return ('<name={0!r} wavelength={1} unit={2!r} element={3!r} '
               ' ionization={4!r}>'.format(self._name, self._wave, self._unit, self._element, self._ionization)) 

   # Define string version
   def __str__(self):
       return (' Name = {0}\n'
               ' Wavelength = {1}\n'
               ' Wavelength unit = {2}\n'
               ' Element = {3}\n'
               ' Ionization = {4}'.format(self._name, self._wave, self._unit, self._element, self._ionization))

   @property
   def element(self):
       """The element of the line"""
       return self._element

   @property
   def ionization(self):
       """The ionization state of the line"""
       return self._ionization

class MolecularLines(BaseLines):
   """Molecular Line Class

   """

   __slots__ = ('_molecule', '_transitiontype')

   def __init__(self, name, wave, unit, molecule, transitiontype):
       super(MolecularLines, self).__init__(name, wave, unit)
       self._molecule = molecule
       self._transitiontype = transitiontype

   # !r calls repr(), !s calls str()
   # Define printable representation
   def __repr__(self):
       return ('<name={0!r} wavelength={1} unit={2!r} molecule={3!r} '
               ' transitiontype={4!r}>'.format(self._name, self._wave, self._unit, self._molecule, self._transitiontype)) 

   # Define string version
   def __str__(self):
       return (' Name = {0}\n'
               ' Wavelength = {1}\n'
               ' Wavelength unit = {2}\n'
               ' Molecule = {3}\n'
               ' Transition type = {4}'.format(self._name, self._wave, self._unit, self._molecule, self._transitiontype))

   @property
   def molecule(self):
       """The molecule of the line"""
       return self._molecule

   @property
   def transitiontype(self):
       """The transitiontype of the line"""
       return self._transitiontype
