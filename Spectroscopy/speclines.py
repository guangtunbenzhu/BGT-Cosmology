
# Define Class Spectrograph
class SpecLine(object):
   """A line

   These objects are spectrographs used in astronomy.
   Units for minwave/maxwave/width are Angstrom.
   """

   # - Attribute names are fixed. Therefore the __slots__
   # - Ideally their values should be fixed as well (private) once initialized
   #      Therefore the prefixing underscore _ (meaning semi-private) and the (seemingly redundant) properties
   __slots__ = ('_name', '_wave', '_EinsteinA', '_EinsteinBul', '_EinsteinBlu', '_strength', '_oscillator', '_reference1', '_reference2')

   def __init__(self, name, wave, EinsteinA, EinsteinBul, EinsteinBlu, strength, oscillator, reference1, reference2):
       self._name = name
       self._wave = wave
       self._EinsteinA = EinsteinA
       self._EinsteinBul = EinsteinBul
       self._EinsteinBlu = EinsteinBlu
       self._strength = strength
       self._oscillator = oscillator
       self._reference1 = reference1
       self._reference2 = reference2

   # Define printable representation
   def __repr__(self):
       return ('<SpecLine name={0!r} wave={1} EinsteinA={2}, EinsteinBul={3} EinsteinBlu={4}  strength={5} oscillator={6} reference1={7!r} reference2={8!r}>'.format(
               self._name, self._wave, self._EinsteinA, self._EinsteinBul, self._EinsteinBlu, self._strength, self._oscillator, self._reference1, self._reference2))

   # Define string version
   def __str__(self):
       return (' Name = {0}\n'
               ' Vacuum Wavelength = {1} Ang\n'
               ' Einstein A = {2} \n'
               ' Einstein B (u->l) = {3} \n'
               ' Einstein B (l->u) = {4} \n'
               ' Strength = {5} \n'
               ' Oscillator Strength = {6}\n'
               ' Reference 1 = {7} \n'
               ' Reference 2 = {8} \n'.format(self._name, self._wave, self._EinsteinA, self._EinsteinBul, self._EinsteinBlu, 
               self._strength, self._oscillator, self._reference1, self._reference2))

   @property
   def name(self):
       """The full name of the line."""
       return self._name

   @property
   def wave(self):
       """The vacuum wavelength of the line."""
       return self._wave

   @property
   def EinsteinA(self):
       """The Einstein A coefficient of the line."""
       return self._EinsteinA

   @property
   def EinsteinBul(self):
       """The Einstein B(u->l) coefficient of the line."""
       return self._EinsteinBul

   @property
   def EinsteinBlu(self):
       """The Einstein B(l->u) coefficient of the line."""
       return self._EinsteinBlu

   @property
   def strength(self):
       """The strength of the line."""
       return self._strength

   @property
   def oscillator(self):
       """The oscillator strength of the line."""
       return self._oscillator

   @property
   def reference1(self):
       """Reference"""
       return self._reference1

   @property
   def reference2(self):
       """Reference"""
       return self._reference2

# Done Class SpecLine

# Initialize common spectrographs:

# 2200 -> 2400
#                   '_name',     '_wave', '_EinsteinA', '_EinsteinBul', '_EinsteinBlu', '_strength', '_oscillator', '_reference1', '_reference2'
FeII2250 = SpecLine('FeII  2250', 2249.875, 3.00E+6, 0., 0., 1.35E-1, 1.82E-3, 'NIST', 'NIST')
FeII2261 = SpecLine('FeII  2261', 2260.779, 3.18E+6, 0., 0., 1.82E-1, 2.44E-3, 'NIST', 'NIST')
CIII2298 = SpecLine('CIII  2298', 2297.579, 1.38E+8, 0., 0., 4.12E+0, 1.82E-1, 'NIST', 'NIST')
CII2326  = SpecLine('CII]  2326', 2326.110, 4.43E+1, 0., 0., 1.65E-6, 5.40E-8, 'NIST', 'NIST')
CII2328  = SpecLine('CII]  2328', 2327.640, 8.49E+0, 0., 0., 2.11E-7, 6.90E-9, 'NIST', 'NIST')
CII2329  = SpecLine('CII]  2329', 2328.830, 6.78E+1, 0., 0., 8.45E-7, 2.76E-8, 'NIST', 'NIST')
FeII2344 = SpecLine('FeII  2344', 2344.213, 1.73E+8, 0., 0., 8.80E+0, 1.14E-1, 'NIST', 'NIST')
FeII2366 = SpecLine('FeII* 2366', 2365.550, 5.90E+7, 0., 0., 3.08E+0, 4.95E-2, 'NIST', 'NIST')
FeII2374 = SpecLine('FeII  2374', 2374.460, 4.25E+7, 0., 0., 2.81E+0, 3.59E-2, 'NIST', 'NIST')
FeII2383 = SpecLine('FeII  2383', 2382.764, 3.13E+8, 0., 0., 2.51E+1, 3.20E-1, 'NIST', 'NIST')
FeII2396 = SpecLine('FeII* 2396', 2396.355, 2.59E+8, 0., 0., 1.76E+1, 2.79E-1, 'NIST', 'NIST')
NeIV2422 = SpecLine('[NeIV] 2422',2422.561, 5.20E-3, 0., 0., 1.10E-5, 0., 'NIST', 'NIST')
NeIV2425 = SpecLine('[NeIV] 2425',2425.139, 1.80E-4, 0., 0., 5.71E-7, 0., 'NIST', 'NIST')
OII2470 = SpecLine('[OII] 2470',2470.966, 2.12E-2, 0., 0., 2.37E-5, 0., 'NIST', 'NIST')
OII2471 = SpecLine('[OII] 2471',2471.088, 5.22E-2, 0., 0., 1.17E-4, 0., 'NIST', 'NIST')

# 2500 -> 2700
#                   '_name',     '_wave', '_EinsteinA', '_EinsteinBul', '_EinsteinBlu', '_strength', '_oscillator', '_reference1', '_reference2'
MnII2577 = SpecLine('MnII  2577', 2576.875, 2.80E+8, 0., 0., 2.13E+1, 3.58E-1, 'NIST', 'NIST')
FeII2587 = SpecLine('FeII  2587', 2586.649, 8.94E+7, 0., 0., 6.11E+0, 7.17E-2, 'NIST', 'NIST')
MnII2594 = SpecLine('MnII  2594', 2594.496, 2.76E+8, 0., 0., 1.66E+1, 2.79E-1, 'NIST', 'NIST')
FeII2600 = SpecLine('FeII  2600', 2600.172, 2.35E+8, 0., 0., 2.04E+1, 2.39E-1, 'NIST', 'NIST')
MnII2606 = SpecLine('MnII  2606', 2606.459, 2.69E+8, 0., 0., 1.18E+1, 1.96E-1, 'NIST', 'NIST')
FeII2613 = SpecLine('FeII* 2613', 2612.653, 1.20E+8, 0., 0., 8.43E+0, 1.22E-1, 'NIST', 'NIST')
FeII2626 = SpecLine('FeII* 2626', 2626.450, 3.52E+7, 0., 0., 3.15E+0, 4.55E-2, 'NIST', 'NIST')
FeII2632 = SpecLine('FeII* 2632', 2632.107, 6.29E+7, 0., 0., 4.53E+0, 8.70E-2, 'NIST', 'NIST')

# 2700 -> 2900
#                   '_name',     '_wave', '_EinsteinA', '_EinsteinBul', '_EinsteinBlu', '_strength', '_oscillator', '_reference1', '_reference2'
MgII2796 = SpecLine('MgII  2796', 2796.352, 2.60E+8, 0., 0., 1.12E+1, 6.08E-1, 'Kelleher & Podobedova 2008', 'NIST')
MgII2803 = SpecLine('MgII  2803', 2803.531, 2.57E+8, 0., 0., 5.60E+0, 3.03E-1, 'Kelleher & Podobedova 2008', 'NIST')
MgI2853  = SpecLine('MgI   2853', 2852.964, 4.91E+8, 0., 0., 1.69E+1, 1.80E+0, 'Kelleher & Podobedova 2008', 'NIST')

# 3000 -> 3600
TiII3067 = SpecLine('TiII  3067', 3067.237, 3.47E+7, 0., 0., 1.98E+0, 4.89E-2, 'NIST', 'NIST')
TiII3074 = SpecLine('TiII  3074', 3073.863, 1.71E+8, 0., 0., 4.90E+0, 1.21E-1, 'NIST', 'NIST')
#TiII3204 = SpecLine('TiII  3204', 3204.357, 1.61E+6, 0., 0., 1.57E-1, 3.72E-3, 'NIST', 'NIST')
TiII3230 = SpecLine('TiII  3230', 3230.122, 2.93E+7, 0., 0., 2.92E+0, 6.87E-2, 'NIST', 'NIST')
TiII3243 = SpecLine('TiII  3243', 3242.918, 1.47E+8, 0., 0., 9.90E+0, 2.32E-1, 'NIST', 'NIST')
TiII3385 = SpecLine('TiII  3385', 3384.730, 1.39E+8, 0., 0., 1.60E+1, 3.58E-1, 'NIST', 'NIST')

# HI Balmer Series
Balmerbreak = SpecLine('Balmerbreak 3647', 3647.000, 0.00E+0, 0., 0., 0.00E+0, 0.00E+0, 'NoRef', 'NoRef')
Balmerkappa = SpecLine('Balmerkappa 3751', 3751.217, 2.83E+4, 0., 0., 2.13E-1, 2.15E-3, 'NIST', 'NIST')
Balmeriota  = SpecLine('Balmeriota  3772', 3771.704, 4.40E+4, 0., 0., 2.82E-1, 2.84E-3, 'NIST', 'NIST')
Balmertheta = SpecLine('Balmertheta 3799', 3798.987, 7.12E+4, 0., 0., 3.85E-1, 3.85E-3, 'NIST', 'NIST')
Balmereta   = SpecLine('Balmereta   3836', 3836.485, 1.22E+5, 0., 0., 5.49E-1, 5.43E-3, 'NIST', 'NIST')
Balmerzeta  = SpecLine('Balmerzeta  3890', 3890.166, 2.21E+5, 0., 0., 8.24E-1, 8.04E-3, 'NIST', 'NIST')
Balmerepsilon  = SpecLine('Balmerepsilon  3971', 3971.198, 4.39E+5, 0., 0., 1.33E+0, 1.27E-2, 'NIST', 'NIST')
Balmerdelta = SpecLine('Balmerdelta 4103', 4102.892, 9.73E+5, 0., 0., 2.39E+0, 2.21E-2, 'NIST', 'NIST')
Balmergamma = SpecLine('Balmergamma 4342', 4341.692, 2.53E+6, 0., 0., 5.11E+0, 4.47E-2, 'NIST', 'NIST')
Balmerbeta  = SpecLine('Balmerbeta  4863', 4862.691, 8.42E+6, 0., 0., 1.53E+1, 1.19E-1, 'NIST', 'NIST')
Balmeralpha = SpecLine('Balmeralpha 6564', 6564.600, 4.41E+7, 0., 0., 1.12E+2, 6.41E-1, 'NIST', 'NIST')

# HeI
HeI3189 = SpecLine('HeI 3189', 3189.667, 5.64E+6, 0., 0., 2.70E-1, 8.59E-3, 'NIST', 'NIST') 
HeI3890 = SpecLine('HeI 3890', 3889.750, 9.48E+6, 0., 0., 1.38E+0, 3.58E-2, 'NIST', 'NIST') 
HeI5877 = SpecLine('HeI 5877', 5877.249, 7.07E+7, 0., 0., 4.96E+1, 5.13E-1, 'NIST', 'NIST') 
HeI6680 = SpecLine('HeI 6680', 6679.995, 6.37E+7, 0., 0., 4.69E+1, 7.10E-1, 'NIST', 'NIST')

# 3600 -> 4000
#                   '_name',     '_wave', '_EinsteinA', '_EinsteinBul', '_EinsteinBlu', '_strength', '_oscillator', '_reference1', '_reference2'
OII3727   = SpecLine('[OII]   3727', 3727.100, 1.59e-04, 0., 0., 1.22E-6, 0., 'NIST', 'NIST')
OII3730   = SpecLine('[OII]   3730', 3729.860, 2.86e-05, 0., 0., 1.11E-4, 0., 'NIST', 'NIST')
NeIII3870 = SpecLine('[NeIII] 3870', 3869.860, 1.74e-01, 0., 0., 1.87E-3, 0., 'NIST', 'NIST')
CaII3934  = SpecLine('CaIIK 3934', 3934.777, 1.397E8, 0., 0., 1.77E+1, 0.6485, 'Safronova & Safronova 2011', 'NIST')
CaII3969  = SpecLine('CaIIH 3969', 3969.592, 1.360E8, 0., 0., 8.60E+0, 0.3213, 'Safronova & Safronova 2011', 'NIST')

# 4000 -> 6000
#                   '_name',     '_wave', '_EinsteinA', '_EinsteinBul', '_EinsteinBlu', '_strength', '_oscillator', '_reference1', '_reference2'
SII4070  = SpecLine('[SII] 4070',  4069.749, 1.92E-1, 0., 0., 1.92E-3, 0., 'NIST', 'NIST') 
SII4077  = SpecLine('[SII] 4077',  4077.500, 7.72E-2, 0., 0., 3.88E-4, 0., 'NIST', 'NIST') 
OIII4363 = SpecLine('[OIII] 4364', 4364.435, 1.71E+0, 0., 0., 2.42E+0, 0., 'NIST', 'NIST') 
OIII4960 = SpecLine('[OIII] 4960', 4960.295, 6.21E-3, 0., 0., 1.40E-4, 0., 'NIST', 'NIST') 
OIII5008 = SpecLine('[OIII] 5008', 5008.240, 1.81E-2, 0., 0., 4.21E-4, 0., 'NIST', 'NIST') 
MgI5168  = SpecLine('MgIb 5168', 5168.761, 1.13E+7, 0., 0., 2.30E+0, 1.35E-1, 'Kelleher & Podobedova 2008', 'NIST') 
MgI5174  = SpecLine('MgIb 5174', 5174.125, 3.37E+7, 0., 0., 6.92E+0, 1.35E-1, 'Kelleher & Podobedova 2008', 'NIST') 
MgI5185  = SpecLine('MgIb 5185', 5185.048, 5.61E+7, 0., 0., 1.16E+1, 1.36E-1, 'Kelleher & Podobedova 2008', 'NIST') 
NI5199   = SpecLine('[NI]   5199', 5199.349, 4.34E-6, 0., 0., 5.89E-5, 0., 'NIST', 'NIST') 
NI5201   = SpecLine('[NI]   5201', 5201.705, 6.59E-6, 0., 0., 1.35E-4, 0., 'NIST', 'NIST') 
NaI5891  = SpecLine('NaID 5891', 5891.583, 6.16E+7, 0., 0., 2.49E+1, 6.41E-1, 'Kelleher & Podobedova 2008', 'NIST')
NaI5897  = SpecLine('NaID 5897', 5897.558, 6.14E+7, 0., 0., 1.24E+1, 3.20E-1, 'Kelleher & Podobedova 2008', 'NIST')

# 6000 -> 8000
OI6302   = SpecLine('[OI]  6302', 6302.046, 5.63E-3, 0., 0., 2.61E-4, 0., 'NIST', 'NIST') 
SIII6314 = SpecLine('[SIII] 6314', 6314.431, 2.04E+7, 0., 0., 1.01E+1, 8.13E-2, 'NIST', 'NIST')
NII6550  = SpecLine('[NII] 6550', 6549.860, 9.84E-4, 0., 0., 5.13E-5, 0., 'NIST', 'NIST') 
NII6585  = SpecLine('[NII] 6585', 6585.270, 2.91E-3, 0., 0., 1.54E-4, 0., 'NIST', 'NIST') 
SII6718  = SpecLine('[SII] 6718', 6718.294, 1.88E-4, 0., 0., 1.38E-2, 0., 'NIST', 'NIST')
SII6733  = SpecLine('[SII] 6733', 6732.673, 5.63E-4, 0., 0., 2.54E-5, 0., 'NIST', 'NIST')
ArIII7138 = SpecLine('[ArIII] 7138', 7137.770, 3.21E-1, 0., 0., 2.16E-2, 0., 'NIST', 'NIST')
ArIII7753 = SpecLine('[ArIII] 7753', 7753.190, 8.30E-2, 0., 0., 7.20E-3, 0., 'NIST', 'NIST')

# 8000 -> 10000
#                   '_name',     '_wave', '_EinsteinA', '_EinsteinBul', '_EinsteinBlu', '_strength', '_oscillator', '_reference1', '_reference2'
#CaII8500 = SpecLine('CaII 8500', 8500.35, 9.972E5, 0., 0., 0., 1.080E-2, 'Safronova & Safronova 2011', 'Reference')
#CaII8544 = SpecLine('CaII 8544', 8544.44, 8.876E6, 0., 0., 0., 6.477E-2, 'Safronova & Safronova 2011', 'Reference')
#CaII8664 = SpecLine('CaII 8664', 8664.52, 9.452E6, 0., 0., 0., 5.319E-2, 'Safronova & Safronova 2011', 'Reference')
#SIII9071 = SpecLine('[SIII 9071', 9071.1, 0., 'Reference')
#SIII9533 = SpecLine('[SIII 9533', 9533.2, 0., 'Reference')

# HI Paschen Series
#Paschenbreak = SpecLine('Paschenbreak 8204', 8204.00, 0., 'Reference')
#Paschenkappa = SpecLine('Paschenkappa 8300', 8300.15, 0., 'Reference')
#Pascheniota  = SpecLine('Pascheniota  8700', 8700.15, 0., 'Reference')
#Paschentheta = SpecLine('Paschentheta 8865', 8865.22, 0., 'Reference')
#Pascheneta   = SpecLine('Pascheneta   9017', 9017.38, 0., 'Reference')
#Paschenzeta  = SpecLine('Paschenzeta  9231', 9231.55, 0., 'Reference')
#Paschenepsilon  = SpecLine('Paschenepsilon  9548', 9548.59, 0., 'Reference')
#Paschendelta = SpecLine('Paschendelta 10052', 10052.13, 0., 'Reference')
#Paschengamma = SpecLine('Paschengamma 10941', 10941.09, 0., 'Reference')
#Paschenbeta  = SpecLine('Paschenbeta  12821', 12821.59, 0., 'Reference')
#Paschenalpha = SpecLine('Paschenalpha 18756', 18756.13, 0., 'Reference')
