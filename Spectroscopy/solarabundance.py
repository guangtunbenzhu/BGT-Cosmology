"""
Table of element abundance in solar photosphere.

From: The Chemical Composition of the Sun, Asplund et al., 2009, ARA&A, 47, 481
http://adsabs.harvard.edu/abs/2009ARA%26A..47..481A

To do: add abundance in meteorites as well

See also: http://www.reflectometry.org/danse/docs/elements/guide/extending.html
"""

import periodictable

def init(table, reload=False):
    if 'solar' in table.properties and not reload: return
    table.properties.append('solar')

    # Set the default, if any
    periodictable.core.Element.solar = 0.

    # Not numeric, so no discoverer_units

    # Load the data
    for name,solar in data.items():
        el = table.name(name)
        el.solar = solar-12.

data = dict(
  hydrogen = 12.,
  helium = 10.93,
  lithium = 1.05,
  beryllium = 1.38,
  boron = 2.70,
  carbon = 8.43,
  nitrogen = 7.83,
  oxygen = 8.69, 
  fluorine = 4.56,
  neon = 7.93, 
  sodium = 6.24, 
  magnesium = 7.60,
  aluminum = 6.45, 
  silicon = 7.51,
  phosphorus = 5.41,
  sulfur = 7.12,
  chlorine = 5.50,
  argon = 6.40,
  potassium = 5.03,
  calcium = 6.34, 
  scandium = 3.15, 
  titanium = 4.95, 
  vanadium = 3.93, 
  chromium = 5.64, 
  manganese = 5.43,
  iron = 7.50, 
  cobalt = 4.99, 
  nickel = 6.22, 
  copper = 4.19, 
  zinc = 4.56,
  gallium= 3.04,
  germanium= 3.65,
  krypton= 3.25,
  rubidium= 2.52,
  strontium= 2.87,
  yttrium= 2.21,
  zirconium= 2.58,
  niobium= 1.46,
  molybdenum= 1.88,
  ruthenium= 1.75,
  rhodium= 0.91,
  palladium= 1.57,
  silver= 0.94,
  indium= 0.80,
  tin= 2.04,
  xenon= 2.24,
  barium= 2.18,
  lanthanum= 1.10,
  cerium= 1.58,
  praseodymium= 0.72,
  neodymium= 1.42,
  samarium= 0.96,
  europium= 0.52,
  gadolinium= 1.07,
  terbium= 0.30,
  dysprosium= 1.10,
  holmium= 0.48,
  erbium= 0.92,
  thulium= 0.10,
  ytterbium= 0.84,
  lutetium= 0.10,
  hafnium= 0.85,
  tungsten= 0.85,
  osmium= 1.40,
  iridium= 1.38,
  gold= 0.92,
  thallium= 0.90,
  lead= 1.75,
  thorium= 0.02
)

init(periodictable.elements)
