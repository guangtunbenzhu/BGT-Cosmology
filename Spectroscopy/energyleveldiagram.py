from os.path import isfile, join
import datapath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['text.usetex'] = False

def plot_energylevel_diagram(low_state, high_state, title, absorption=None, emission=None, finestructure=None, figfile='temp.eps'):
    """Main plotting routine
    """

    # Set up the figure, hardcoded for convenience
    figsize = (12,10)
    xlimits = (0,1)
    ylimits = (0, 1.2)

    dyline = 0.05 # Energy separation
    xpos_energy = 0.18
    xpos_jvalue = 0.93
    xline = [0.2, 0.8]
    ypos_low_state = 0.07
    ypos_high_state = 0.78
    xpos_config = 0.93
    ypos_low_config = 0.01
    ypos_high_config = ypos_high_state-0.11+dyline
    ypos_text = ypos_high_state-0.035+dyline
    xpos_term = 0.820
    xpos_title = 0.5
    ypos_title=1.16

    fontsize_energyunit = 17
    fontsize_energy = 16
    fontsize_jvalue = 16
    fontsize_config = 20
    fontsize_term = 22
    fontsize_transition = 14
    fontsize_finestructure = 14
    fontsize_title = 22

    # Initialize the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    # Energy unit, fixed
    xpos_energyunit = 0.15
    ypos_energyunit = 0.05
    ax.text(xpos_energyunit, ypos_energyunit, r'E/hc', fontsize=fontsize_energyunit, ha='center')
    ax.text(xpos_energyunit, ypos_energyunit-0.06, r'(cm$^{-1}$)', fontsize=fontsize_energyunit, ha='center')

    # energy, lines, J value of both energy levels
    for ypos_config, ypos_state, this_state in zip((ypos_low_config, ypos_high_config), (ypos_low_state, ypos_high_state), (low_state, high_state)):
        NJ = this_state['NJ']
        energy_config = this_state['CONFIG']
        ax.text(xpos_config, ypos_config, energy_config, fontsize=fontsize_config, ha='right')
        ypos_term = ypos_state+dyline*NJ+dyline/4
        energy_term = this_state['TERM']
        ax.text(xpos_term, ypos_term, energy_term, fontsize=fontsize_term, ha='left')
        for i in np.arange(NJ):
            energy = this_state['ENERGY'][i]
            jvalue = this_state['J'][i]
            yline = ypos_state+dyline*(i+1)
            ax.text(xpos_energy, yline-dyline/4, energy, fontsize=fontsize_energy, ha='right')
            ax.text(xpos_jvalue, yline-dyline/4, jvalue, fontsize=fontsize_jvalue, ha='right')
            ax.plot(xline, [yline, yline], 'k')
   
    ax.text(xpos_title, ypos_title, title, fontsize=fontsize_title, weight='bold', ha='center')

    # emission transitions
    if emission is not None:
       for this_emission in emission:
           xpos_emission = this_emission['XPOS']
           low_jvalue = this_emission['LOWJ']
           low_ilevel = (np.nonzero((low_state['J']==low_jvalue)))[0][0]
           ypos_low_level = ypos_low_state+dyline*(low_ilevel+1)
           high_jvalue = this_emission['HIGHJ']
           high_ilevel = (np.nonzero((high_state['J']==high_jvalue)))[0][0]
           ypos_high_level = ypos_high_state+dyline*(high_ilevel+1)
           transition_text = this_emission['TEXT']

           head_width=0.008
           head_length=0.015
           ax.arrow(xpos_emission, ypos_high_level, 0., -(ypos_high_level-ypos_low_level-head_length-0.004), 
                    head_width=head_width, head_length=head_length, color='green')
           ax.text(xpos_emission-0.038, ypos_text, transition_text,
                   fontsize=fontsize_transition, rotation='vertical', va='top')

    # absorption transitions
    if absorption is not None:
       for this_absorption in absorption:
           xpos_absorption = this_absorption['XPOS']
           low_jvalue = this_absorption['LOWJ']
           low_ilevel = (np.nonzero((low_state['J']==low_jvalue)))[0][0]
           ypos_low_level = ypos_low_state+dyline*(low_ilevel+1)
           high_jvalue = this_absorption['HIGHJ']
           high_ilevel = (np.nonzero((high_state['J']==high_jvalue)))[0][0]
           ypos_high_level = ypos_high_state+dyline*(high_ilevel+1)

           head_width=0.008
           head_length=0.015
           ax.arrow(xpos_absorption, ypos_low_level, 0., +(ypos_high_level-ypos_low_level-head_length-0.000), 
                    head_width=head_width, head_length=head_length, color='black')

    # fine-structure transitions
    if finestructure is not None:
       for this_finestructure in finestructure:
           xpos_finestructure = this_finestructure['XPOS']
           low_jvalue = this_finestructure['LOWJ']
           low_ilevel = (np.nonzero((low_state['J']==low_jvalue)))[0][0]
           ypos_low_level = ypos_low_state+dyline*(low_ilevel+1)
           high_jvalue = this_finestructure['HIGHJ']
           high_ilevel = (np.nonzero((low_state['J']==high_jvalue)))[0][0]
           ypos_high_level = ypos_low_state+dyline*(high_ilevel+1)
           ypos_text_finestructure = ypos_low_level+0.009
           transition_text = this_finestructure['TEXT']

           head_width=0.005
           head_length=0.007
           ax.arrow(xpos_finestructure, ypos_high_level, 0., -(ypos_high_level-ypos_low_level-head_length-0.004), lw=1, 
                    head_width=head_width, head_length=head_length, color='brown')
           ax.arrow(xpos_finestructure, ypos_low_level, 0., ypos_high_level-ypos_low_level-head_length-0.000, lw=1, 
                    head_width=head_width, head_length=head_length, color='brown')
           ax.text(xpos_finestructure+0.010, ypos_text_finestructure, transition_text,
                   fontsize=fontsize_finestructure, ha='left')

    fig.show()
    fig.savefig(figfile)

_transition_dtype = [('XPOS', 'f4'), ('LOWJ', 'S4'), ('HIGHJ', 'S4'), ('TEXT', 'S30')]

############
# Fe II
############
_FeII_ground_state =    {'NJ':5, 'CONFIG':r'3d$^6$4s', 'TERM':r'a  $^6\!{\rm D}$',           'J':np.array(['9/2', '7/2', '5/2', '3/2', '1/2']), 'ENERGY':np.array(['0', '384.79', '667.68', '862.61', '977.05'])}
_xfine = 0.70
_FeII_ground_finestructure = np.zeros(4, dtype=_transition_dtype)
_FeII_ground_finestructure[0] = (_xfine, '9/2', '7/2', r'25.99$\mu{\rm m}$')
_FeII_ground_finestructure[1] = (_xfine, '7/2', '5/2', r'35.35$\mu{\rm m}$')
_FeII_ground_finestructure[2] = (_xfine, '5/2', '3/2', r'51.30$\mu{\rm m}$')
_FeII_ground_finestructure[3] = (_xfine, '3/2', '1/2', r'87.38$\mu{\rm m}$')
_FeII_excited_state_1 = {'NJ':5, 'CONFIG':r'3d$^6$4p', 'TERM':r'z  $^6\!{\rm D}^{\rm o}\!$', 'J':np.array(['9/2', '7/2', '5/2', '3/2', '1/2']), 'ENERGY':np.array(['38458.99', '38660.05', '38858.97', '39013.22', '39109.32'])}
_FeII_excited_state_2 = {'NJ':6, 'CONFIG':r'3d$^6$4p', 'TERM':r'z  $^6\!{\rm F}^{\rm o}\!$', 'J':np.array(['11/2', '9/2', '7/2', '5/2', '3/2', '1/2']), 'ENERGY':np.array(['41968.07', '42114.82', '42237.06', '42334.84', '42401.32', '42439.85'])}
_FeII_excited_state_3 = {'NJ':3, 'CONFIG':r'3d$^6$4p', 'TERM':r'z  $^6\!{\rm P}^{\rm o}\!$', 'J':np.array(['7/2', '5/2', '3/2']), 'ENERGY':np.array(['42658.24', '43238.61', '43620.98'])}
_FeII_excited_state_4 = {'NJ':4, 'CONFIG':r'3d$^6$4p', 'TERM':r'z  $^4\!{\rm F}^{\rm o}\!$', 'J':np.array(['9/2', '7/2', '5/2', '3/2']), 'ENERGY':np.array(['44232.54', '44753.81', '45079.90', '45289.82'])}
_FeII_excited_state_5 = {'NJ':4, 'CONFIG':r'3d$^6$4p', 'TERM':r'z  $^4\!{\rm D}^{\rm o}\!$', 'J':np.array(['7/2', '5/2', '3/2', '1/2']), 'ENERGY':np.array(['44446.91', '44784.79', '45044.19', '45206.47'])}
_FeII_excited_state_6 = {'NJ':3, 'CONFIG':r'3d$^6$4p', 'TERM':r'z  $^4\!{\rm P}^{\rm o}\!$', 'J':np.array(['5/2', '3/2', '1/2']), 'ENERGY':np.array(['46967.48', '47389.81', '47626.11'])}

def plot_feiiuv1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','FeII_UV1.eps')
    title = 'Fe II UV1'

    #Low and high state
    low_state = _FeII_ground_state
    high_state = _FeII_excited_state_1

    xoff = 0.08
    xoff1 = 0.17

    # emission
    nemiss = 5
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '9/2', '9/2', '2600.17   2.35E8   2.39E-1')
    emission[1] = (0.20+xoff+0.07, '7/2', '9/2', '*2626.45   3.52E7   4.55E-2')
    emission[2] = (0.20+xoff+0.07+xoff1, '9/2', '7/2', '2586.65   8.94E7   7.17E-2')
    emission[3] = (0.20+xoff+0.07+xoff1+0.07, '7/2', '7/2', '*2612.65   1.20E8   1.22E-1')
    emission[4] = (0.20+xoff+0.07+xoff1+0.07+0.07, '5/2', '7/2', '*2632.11   6.29E7   8.87E-2')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '9/2', '9/2', '2600.17   2.35E8   2.39E-1')
    absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '9/2', '7/2', '2586.65   8.94E7   7.17E-2')
    # fine structure
    finestructure = _FeII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=finestructure, figfile=figfile)

def plot_feiiuv2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','FeII_UV2.eps')
    title = 'Fe II UV2'

    #Low and high state
    low_state = _FeII_ground_state
    high_state = _FeII_excited_state_2

    xoff = 0.08
    xoff1 = 0.17

    # emission
    nemiss = 3
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '9/2', '11/2', '2382.76   3.13E8   3.20E-1')
    emission[1] = (0.20+xoff+0.07+xoff1, '9/2', '9/2', '2374.46   4.25E7   3.59E-2')
    emission[2] = (0.20+xoff+0.07+xoff1+0.07, '7/2', '9/2', '*2396.36   2.59E8   2.79E-1')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '9/2', '11/2', '2382.76   3.13E8   3.20E-1')
    absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '9/2', '9/2', '2374.46   4.25E7   3.59E-2')
    # fine structure
    finestructure = _FeII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=finestructure, figfile=figfile)

def plot_feiiuv3():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','FeII_UV3.eps')
    title = 'Fe II UV3'

    #Low and high state
    low_state = _FeII_ground_state
    high_state = _FeII_excited_state_3

    xoff = 0.08
    xoff1 = 0.15

    # emission
    nemiss = 6
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '9/2', '7/2', '2344.21   1.73E8   1.14E-1')
    emission[1] = (0.20+xoff+0.06, '7/2', '7/2', '*2365.55   5.90E7   4.95E-2')
    emission[2] = (0.20+xoff+0.06*2, '5/2', '7/2', '*2381.49   3.10E7   3.51E-2')
    emission[3] = (0.20+xoff+0.06*2+xoff1, '7/2', '5/2', '*2333.51   1.31E8   8.00E-2')
    emission[4] = (0.20+xoff+0.06*2+xoff1+0.06, '5/2', '5/2', '*2349.02   1.15E8   9.50E-2')
    emission[5] = (0.20+xoff+0.06*2+xoff1+0.06*2, '3/2', '5/2', '*2359.83   5.00E7   6.30E-2')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '9/2', '7/2', '2344.21   1.73E8   1.14E-1')
    absorption[1] = (0.20+xoff+0.06*2+xoff1-0.008, '7/2', '5/2', '*2333.51   1.31E8   8.00E-2')
    # fine structure
    finestructure = _FeII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=finestructure, figfile=figfile)

def plot_feiiuv4():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','FeII_UV4.eps')
    title = 'Fe II UV4'

    #Low and high state
    low_state = _FeII_ground_state
    high_state = _FeII_excited_state_4

    xoff = 0.08
    xoff1 = 0.17

    # emission
    nemiss = 5
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '9/2', '9/2', '2260.78   3.18E6   2.44E-3')
    emission[1] = (0.20+xoff+0.07, '7/2', '9/2', '*2280.62   4.49E6   4.38E-3')
    emission[2] = (0.20+xoff+0.07+xoff1, '9/2', '7/2', '2234.45   0.00E0   0.00E-0')
    emission[3] = (0.20+xoff+0.07+xoff1+0.07, '7/2', '7/2', '*2253.82   4.40E6   3.40E-3')
    emission[4] = (0.20+xoff+0.07+xoff1+0.07+0.07, '5/2', '7/2', '*2268.29   3.69E6   3.80E-3')

    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '9/2', '9/2', '2260.78   3.18E6   2.44E-3')
    absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '9/2', '7/2', '2234.45   0.00E0   7.17E-2')
    # fine structure
    finestructure = _FeII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=finestructure, figfile=figfile)

def plot_feiiuv5():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','FeII_UV5.eps')
    title = 'Fe II UV5'

    #Low and high state
    low_state = _FeII_ground_state
    high_state = _FeII_excited_state_5

    xoff = 0.08
    xoff1 = 0.17

    # emission
    nemiss = 4
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '9/2', '7/2', '2249.87   3.00E6   1.82E-3')
    emission[1] = (0.20+xoff+0.07, '7/2', '7/2', '*2269.52   4.00E5   3.10E-4')
    emission[2] = (0.20+xoff+0.07+xoff1, '7/2', '5/2', '*2252.52   1.00E6   6.00E-4')
    emission[3] = (0.20+xoff+0.07+xoff1+0.07, '5/2', '5/2', '*2266.70   1.00E6   8.00E-4')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '9/2', '7/2', '2249.87   3.00E6   1.82E-3')
    absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '7/2', '5/2', '*2252.52   1.00E6   6.00E-4')
    # fine structure
    finestructure = _FeII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=finestructure, figfile=figfile)

############
# Mg II
############
_MgII_ground_state =    {'NJ':1, 'CONFIG':r'2p$^6$3s', 'TERM':r'   $^2\!{\rm S}$',           'J':np.array(['1/2']), 'ENERGY':np.array(['0'])}
_MgII_excited_state_1 =    {'NJ':2, 'CONFIG':r'2p$^6$3p', 'TERM':r'   $^2\!{\rm P}^{\rm o}\!$','J':np.array(['1/2', '3/2']), 'ENERGY':np.array(['35669.31', '35760.88'])}
def plot_mgiiuv1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','MgII_UV1.eps')
    title = 'Mg II UV1'

    #Low and high state
    low_state = _MgII_ground_state
    high_state = _MgII_excited_state_1

    xoff = 0.14
    xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '1/2', '1/2', '2796.35   2.60E8   6.08E-1')
    emission[1] = (0.20+xoff+0.07+xoff1, '1/2', '3/2', '2803.53   2.57E8   3.03E-1')

    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '1/2', '1/2', '2796.35   2.60E8   6.08E-1')
    absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '1/2', '3/2', '2803.53   2.57E8   3.03E-1')
    # fine structure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)

############
# Mg I
############
_MgI_ground_state =    {'NJ':1, 'CONFIG':r'2p$^6$3s$^2$', 'TERM':'  '+r'   $^1\!{\rm S}$',           'J':np.array(['0']), 'ENERGY':np.array(['0'])}
_MgI_excited_state_1 =    {'NJ':3, 'CONFIG':r'2p$^6$3s3p', 'TERM':r'   $^3\!{\rm P}^{\rm o}\!$','J':np.array(['0', '1', '2']), 'ENERGY':np.array(['21850.4', '21870.5', '21911.2'])}
_MgI_excited_state_2 =    {'NJ':1, 'CONFIG':r'2p$^6$3s3p', 'TERM':r'   $^1\!{\rm P}^{\rm o}\!$','J':np.array(['1']), 'ENERGY':np.array(['35051.26'])}
_MgI_excited_state_3 =    {'NJ':1, 'CONFIG':r'2p$^6$3s4s', 'TERM':r'   $^3\!{\rm S}$','J':np.array(['1']), 'ENERGY':np.array(['41197.4'])}
def plot_mgiuv1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','MgI_UV1.eps')
    title = 'Mg I UV1'

    #Low and high state
    low_state = _MgI_ground_state
    high_state = _MgI_excited_state_2

    xoff = 0.24
    xoff1 = 0.17

    # emission
    nemiss = 1
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '0', '1', '2852.58   2.60E8   6.08E-1')

    # absorption
    nabs = 1
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '0', '1', '2852.58   2.60E8   6.08E-1')
    # fine structure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)

def plot_mgino2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','MgI_No2.eps')
    title = 'Mg I No2'

    #Low and high state
    low_state = _MgI_excited_state_1
    high_state = _MgI_excited_state_3

    xoff = 0.20
    #xoff1 = 0.17

    # emission
    nemiss = 3
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '0', '1', '5168.76   1.13E7    1.35E-1')
    emission[1] = (0.20+xoff+0.08, '1', '1', '5174.12   3.37E7     1.35E-1')
    emission[2] = (0.20+xoff+0.08*2, '2', '1', '5185.05   5.61E7    1.36E-1')
    #emission[2] = (0.20+xoff+0.07+xoff1, '3/2', '1/2', '*2328.83   6.78E1   2.76E-8')
    #emission[3] = (0.20+xoff+0.07+xoff1+0.07, '3/2', '3/2', '*2327.64   8.49E0   6.90E-9')
    #emission[4] = (0.20+xoff+0.07+xoff1+0.07+0.07, '3/2', '5/2', '*2326.11   4.43E1   5.40E-8')
    # absorption
    nabs = 3
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '0', '1', '5168.76   1.13E7    1.35E-1')
    absorption[1] = (0.20+xoff+0.08-0.008, '1', '1', '5174.12   3.37E7     1.35E-1')
    absorption[2] = (0.20+xoff+0.08*2-0.008, '2', '1', '5185.05   5.61E7    1.36E-1')
    #absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '9/2', '7/2', '2586.65   8.94E7   7.17E-2')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)


############
# Mn II
############
_MnII_ground_state =    {'NJ':1, 'CONFIG':r'3d$^5$4s', 'TERM':r'a  $^7\!{\rm S}$',         'J':np.array(['3']), 'ENERGY':np.array(['0'])}
_MnII_excited_state_1 = {'NJ':3, 'CONFIG':r'3d$^5$4p', 'TERM':r'z  $^7\!{\rm P}^{\rm o}\!$', 'J':np.array(['2', '3', '4']), 'ENERGY':np.array(['38366.23', '38543.12', '38806.69'])}
def plot_mniiuv1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','MnII_UV1.eps')
    title = 'Mn II UV1'

    #Low and high state
    low_state = _MnII_ground_state
    high_state = _MnII_excited_state_1

    xoff = 0.13

    # emission
    nemiss = 3
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3', '2', '2576.88   2.80E8   3.58E-1')
    emission[1] = (0.20+xoff*2, '3', '3', '2594.50   2.76E8   2.79E-1')
    emission[2] = (0.20+xoff*3, '3', '4', '2606.46   2.69E8   1.96E-1')

    # absorption
    nabs = 3
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3', '2', '2576.88   2.80E8   3.58E-1')
    absorption[1] = (0.20+xoff*2-0.008, '3', '3', '2594.50   2.76E8   2.79E-1')
    absorption[2] = (0.20+xoff*3-0.008, '3', '4', '2606.46   2.69E8   1.96E-1')
    # fine structure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)

############
# C II
############
_CII_ground_state =    {'NJ':2, 'CONFIG':r'2s$^2$2p', 'TERM':r'   $^2\!{\rm P}^{\rm o}\!$', 'J':np.array(['1/2', '3/2']), 'ENERGY':np.array(['0', '63.42'])}
_CII_ground_finestructure = np.zeros(1, dtype=_transition_dtype)
_CII_ground_finestructure[0] = (_xfine, '1/2', '3/2', r'157.7$\mu{\rm m}$')
_CII_excited_state_1 = {'NJ':3, 'CONFIG':r'2s2p$^2$', 'TERM':r'   $^4\!{\rm P}$', 'J':np.array(['1/2', '3/2', '5/2']), 'ENERGY':np.array(['43003.3', '43025.3', '43053.6'])}

def plot_ciiuv1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','CII_UV1.eps')
    title = 'C II UV1'

    #Low and high state
    low_state = _CII_ground_state
    high_state = _CII_excited_state_1

    xoff = 0.08
    xoff1 = 0.17

    # emission
    nemiss = 5
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '1/2', '1/2', '2325.40   5.99E1   4.86E-8')
    emission[1] = (0.20+xoff+0.07, '1/2', '3/2', '2324.21   1.40E0   2.27E-9')
    emission[2] = (0.20+xoff+0.07+xoff1, '3/2', '1/2', '*2328.83   6.78E1   2.76E-8')
    emission[3] = (0.20+xoff+0.07+xoff1+0.07, '3/2', '3/2', '*2327.64   8.49E0   6.90E-9')
    emission[4] = (0.20+xoff+0.07+xoff1+0.07+0.07, '3/2', '5/2', '*2326.11   4.43E1   5.40E-8')
    # absorption
    #nabs = 2
    #absorption = np.zeros(nabs, dtype=_transition_dtype)
    #absorption[0] = (0.20+xoff-0.008, '9/2', '9/2', '2600.17   2.35E8   2.39E-1')
    #absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '9/2', '7/2', '2586.65   8.94E7   7.17E-2')
    # fine structure
    finestructure = _CII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=finestructure, figfile=figfile)

############
# Ti II
############
_TiII_ground_state =    {'NJ':4, 'CONFIG':r'3d$^2$4s', 'TERM':r'a  $^4\!{\rm F}$', 'J':np.array(['3/2', '5/2', '7/2', '9/2']), 'ENERGY':np.array(['0', '94.1142', '225.7039', '393.4459'])}
_TiII_ground_finestructure = np.zeros(3, dtype=_transition_dtype)
_TiII_ground_finestructure[0] = (_xfine, '3/2', '5/2', r'106.2\mu{\rm m}$')
_TiII_ground_finestructure[1] = (_xfine, '5/2', '7/2', r'75.99$\mu{\rm m}$')
_TiII_ground_finestructure[2] = (_xfine, '7/2', '9/2', r'59.62$\mu{\rm m}$')

_TiII_excited_state_1 = {'NJ':4, 'CONFIG':r'3d$^2$4p', 'TERM':r'z  $^4\!{\rm G}^{\rm o}\!$', 'J':np.array(['5/2', '7/2', '9/2', '11/2']), 'ENERGY':np.array(['29544.4', '292734.6', '29968.3', '30240.9'])}
_TiII_excited_state_2 = {'NJ':4, 'CONFIG':r'3d$^2$4p', 'TERM':r'z  $^4\!{\rm F}^{\rm o}\!$', 'J':np.array(['3/2', '5/2', '7/2', '9/2']), 'ENERGY':np.array(['30836.4', '30958.6', '31113.7', '31301.1'])}
_TiII_excited_state_5 = {'NJ':4, 'CONFIG':r'3d$^2$4p', 'TERM':r'z  $^4\!{\rm D}^{\rm o}\!$', 'J':np.array(['1/2', '3/2', '5/2', '7/2']), 'ENERGY':np.array(['32532.4', '32602.6', '32698.1', '32767.2'])}

def plot_tiiino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','TiII_No1.eps')
    title = 'Ti II No1'

    #Low and high state
    low_state = _TiII_ground_state
    high_state = _TiII_excited_state_1

    xoff = 0.28
    #xoff1 = 0.17

    # emission
    nemiss = 1
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '5/2', '3384.73   1.39E8    3.58E-1')
    #emission[1] = (0.20+xoff+0.07, '1/2', '3/2', '2324.21   1.40E0   2.27E-9')
    #emission[2] = (0.20+xoff+0.07+xoff1, '3/2', '1/2', '*2328.83   6.78E1   2.76E-8')
    #emission[3] = (0.20+xoff+0.07+xoff1+0.07, '3/2', '3/2', '*2327.64   8.49E0   6.90E-9')
    #emission[4] = (0.20+xoff+0.07+xoff1+0.07+0.07, '3/2', '5/2', '*2326.11   4.43E1   5.40E-8')
    # absorption
    nabs = 1
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '5/2', '3384.73   1.39E8    3.58E-1')
    #absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '9/2', '7/2', '2586.65   8.94E7   7.17E-2')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)

def plot_tiiino2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','TiII_No2.eps')
    title = 'Ti II No2'

    #Low and high state
    low_state = _TiII_ground_state
    high_state = _TiII_excited_state_2

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '3/2', '3242.92   1.47E8    2.32E-1')
    emission[1] = (0.20+xoff+0.09, '3/2', '5/2', '3230.12   2.93E7     6.87E-2')
    #emission[2] = (0.20+xoff+0.07+xoff1, '3/2', '1/2', '*2328.83   6.78E1   2.76E-8')
    #emission[3] = (0.20+xoff+0.07+xoff1+0.07, '3/2', '3/2', '*2327.64   8.49E0   6.90E-9')
    #emission[4] = (0.20+xoff+0.07+xoff1+0.07+0.07, '3/2', '5/2', '*2326.11   4.43E1   5.40E-8')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '3/2', '3242.92   1.47E8    2.32E-1')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '5/2', '3230.12   2.93E7     6.87E-2')
    #absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '9/2', '7/2', '2586.65   8.94E7   7.17E-2')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)

def plot_tiiino5():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','TiII_No5.eps')
    title = 'Ti II No5'

    #Low and high state
    low_state = _TiII_ground_state
    high_state = _TiII_excited_state_5

    xoff = 0.20
    #xoff1 = 0.17

    # emission
    nemiss = 3
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '1/2', '3073.86   1.71E8    1.21E-1')
    emission[1] = (0.20+xoff+0.08, '3/2', '3/2', '3067.23   3.47E7     4.89E-2')
    emission[2] = (0.20+xoff+0.08*2, '3/2', '5/2', '3058.28   1.98E6    4.16E-3')
    #emission[2] = (0.20+xoff+0.07+xoff1, '3/2', '1/2', '*2328.83   6.78E1   2.76E-8')
    #emission[3] = (0.20+xoff+0.07+xoff1+0.07, '3/2', '3/2', '*2327.64   8.49E0   6.90E-9')
    #emission[4] = (0.20+xoff+0.07+xoff1+0.07+0.07, '3/2', '5/2', '*2326.11   4.43E1   5.40E-8')
    # absorption
    nabs = 3
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '1/2', '3073.86   1.71E8    1.21E-1')
    absorption[1] = (0.20+xoff+0.08-0.008, '3/2', '3/2', '3067.23   3.47E7     4.89E-2')
    absorption[2] = (0.20+xoff+0.08*2-0.008, '3/2', '5/2', '3058.28   1.98E6    4.16E-3')
    #absorption[1] = (0.20+xoff+0.07+xoff1-0.008, '9/2', '7/2', '2586.65   8.94E7   7.17E-2')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)

############
# Ca II
############
_CaII_ground_state =    {'NJ':1, 'CONFIG':r'3p$^6$4s', 'TERM':r'   $^2\!{\rm S}$', 'J':np.array(['1/2']), 'ENERGY':np.array(['0'])}
_CaII_excited_state_1 = {'NJ':2, 'CONFIG':r'3p$^6$3d', 'TERM':r'   $^2\!{\rm D}$', 'J':np.array(['3/2', '5/2']), 'ENERGY':np.array(['13650.2', '13710.9'])}
_CaII_excited_state_2 = {'NJ':2, 'CONFIG':r'3p$^6$4p', 'TERM':r'   $^2\!{\rm P}^{\rm o}\!$', 'J':np.array(['1/2', '3/2']), 'ENERGY':np.array(['25191.5', '25414.4'])}

def plot_caiino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','CaII_No1.eps')
    title = 'Ca II No1'

    #Low and high state
    low_state = _CaII_ground_state
    high_state = _CaII_excited_state_2

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '1/2', '1/2', '3934.77   1.47E8   6.82E-1')
    emission[1] = (0.20+xoff+0.09, '1/2', '3/2', '3969.59   1.40E8   3.03E-1')
    #emission[2] = (0.20+xoff+0.07+xoff1, '3/2', '1/2', '*2328.83   6.78E1   2.76E-8')
    #emission[3] = (0.20+xoff+0.07+xoff1+0.07, '3/2', '3/2', '*2327.64   8.49E0   6.90E-9')
    #emission[4] = (0.20+xoff+0.07+xoff1+0.07+0.07, '3/2', '5/2', '*2326.11   4.43E1   5.40E-8')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '1/2', '1/2', '3934.77   1.47E8   6.82E-1')
    absorption[1] = (0.20+xoff+0.09-0.008, '1/2', '3/2', '3969.59   1.40E8   3.03E-1')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)

def plot_caiino2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','CaII_No2.eps')
    title = 'Ca II No2'

    #Low and high state
    low_state = _CaII_excited_state_1
    high_state = _CaII_excited_state_2

    xoff = 0.14
    xoff1 = 0.17

    # emission
    nemiss = 3
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '1/2', '8662.14   1.06E7    5.97E-2')
    emission[1] = (0.20+xoff+xoff1, '3/2', '3/2', '8498.02   1.11E6    1.20E-2')
    emission[2] = (0.20+xoff+xoff1+0.08, '5/2', '3/2', '8542.09   9.90E6     7.2E-2')
    # absorption
    nabs = 3
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '1/2', '8662.14   1.06E7    5.97E-2')
    absorption[1] = (0.20+xoff+xoff1-0.008, '3/2', '3/2', '8498.02   1.11E6    1.20E-2')
    absorption[2] = (0.20+xoff+xoff1+0.08-0.008, '5/2', '3/2', '8542.09   9.90E6     7.2E-2')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)

############
# Na I
############
_NaI_ground_state =    {'NJ':1, 'CONFIG':r'2p$^6$3s', 'TERM':r'   $^2\!{\rm S}$', 'J':np.array(['1/2']), 'ENERGY':np.array(['0'])}
_NaI_excited_state_1 = {'NJ':2, 'CONFIG':r'2p$^6$3p', 'TERM':r'   $^2\!{\rm P}^{\rm o}\!$', 'J':np.array(['1/2', '3/2']), 'ENERGY':np.array(['16956.17', '16973.37'])}
_NaI_excited_state_2 = {'NJ':1, 'CONFIG':r'2p$^6$4s', 'TERM':r'   $^2\!{\rm S}$', 'J':np.array(['1/2']), 'ENERGY':np.array(['25740.00'])}

def plot_naino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','NaI_No1.eps')
    title = 'Na I No1'

    #Low and high state
    low_state = _NaI_ground_state
    high_state = _NaI_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '1/2', '1/2', '5891.58   6.14E7   3.20E-1')
    emission[1] = (0.20+xoff+0.09, '1/2', '3/2', '5897.56   6.16E7   6.41E-1')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '1/2', '1/2', '5891.58   6.14E7   3.20E-1')
    absorption[1] = (0.20+xoff+0.09-0.008, '1/2', '3/2', '5897.56   6.16E7   6.41E-1')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)

def plot_naino3():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','NaI_No3.eps')
    title = 'Na I No3'

    #Low and high state
    low_state = _NaI_excited_state_1
    high_state = _NaI_excited_state_2

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '1/2', '1/2', '11406.9   1.76E7    1.71E-1')
    emission[1] = (0.20+xoff+0.09, '3/2', '1/2', '11384.6   8.80E6    1.71E-1')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '1/2', '1/2', '11406.9   1.76E7    1.71E-1')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '1/2', '11384.6   8.80E6    1.71E-1')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=absorption, finestructure=None, figfile=figfile)


############
# Ne III
############
_NeIII_ground_state =    {'NJ':3, 'CONFIG':r'2s$^2$2p$^4$', 'TERM':r'   $^3\!{\rm P}$', 'J':np.array(['2', '1', '0']), 'ENERGY':np.array(['0', '642.876', '920.550'])}
_NeIII_excited_state_1 = {'NJ':1, 'CONFIG':r'2p$^2$2p$^4$', 'TERM':r'   $^1\!{\rm D}$', 'J':np.array(['2']), 'ENERGY':np.array(['25840.72'])}
#_NeIII_excited_state_2 = {'NJ':1, 'CONFIG':r'2p$^6$4s', 'TERM':r'   $^2\!{\rm S}$', 'J':np.array(['1/2']), 'ENERGY':np.array(['25740.00'])}

def plot_neiiino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','NeIII_No1.eps')
    title = 'Ne III No1'

    #Low and high state
    low_state = _NeIII_ground_state
    high_state = _NeIII_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '2', '2', '3869.86   1.74E-1          ')
    emission[1] = (0.20+xoff+0.09, '1', '2', '3968.59   5.40E-2          ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '1/2', '1/2', '5891.58   6.14E7   3.20E-1')
    absorption[1] = (0.20+xoff+0.09-0.008, '1/2', '3/2', '5897.56   6.16E7   6.41E-1')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

############
# Ne IV
############
_NeIV_ground_state =    {'NJ':1, 'CONFIG':r'2s$^2$2p$^3$', 'TERM':r'   $^4\!{\rm S}^{\rm o}\!$', 'J':np.array(['3/2']), 'ENERGY':np.array(['0'])}
_NeIV_excited_state_1 = {'NJ':2, 'CONFIG':r'2s$^2$2p$^3$', 'TERM':r'   $^2\!{\rm D}^{\rm o}\!$', 'J':np.array(['5/2', '3/2']), 'ENERGY':np.array(['41234.43', '41278.89'])}
_NeIV_excited_state_2 = {'NJ':2, 'CONFIG':r'2s$^2$2p$^3$', 'TERM':r'   $^2\!{\rm P}^{\rm o}\!$', 'J':np.array(['1/2', '3/2']), 'ENERGY':np.array(['62434.45', '62440.78'])}

def plot_neivuv1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','NeIV_UV1.eps')
    title = 'Ne IV UV1'

    #Low and high state
    low_state = _NeIV_ground_state
    high_state = _NeIV_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '5/2', '2425.14   4.00E-4          ')
    emission[1] = (0.20+xoff+0.09, '3/2', '3/2', '2422.56   2.70E-4          ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '5/2', '2425.14   4.00E-4          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '3/2', '2422.56   2.70E-4          ')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

def plot_neivuv2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','NeIV_UV2.eps')
    title = 'Ne IV UV2'

    #Low and high state
    low_state = _NeIV_ground_state
    high_state = _NeIV_excited_state_2

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '1/2', '1601.70   5.30E-1          ')
    emission[1] = (0.20+xoff+0.09, '3/2', '3/2', '1601.50   1.33E-0         ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '1/2', '1601.70   5.30E-1          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '3/2', '1601.50   1.33E-0         ')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

############
# O II
############
_OII_ground_state =    {'NJ':1, 'CONFIG':r'2s$^2$2p$^3$', 'TERM':r'   $^4\!{\rm S}^{\rm o}\!$', 'J':np.array(['3/2']), 'ENERGY':np.array(['0'])}
_OII_excited_state_1 = {'NJ':2, 'CONFIG':r'2s$^2$2p$^3$', 'TERM':r'   $^2\!{\rm D}^{\rm o}\!$', 'J':np.array(['5/2', '3/2']), 'ENERGY':np.array(['26810.55', '26830.57'])}
_OII_excited_state_2 = {'NJ':2, 'CONFIG':r'2s$^2$2p$^3$', 'TERM':r'   $^2\!{\rm P}^{\rm o}\!$', 'J':np.array(['1/2', '3/2']), 'ENERGY':np.array(['40468.01', '40470.00'])}

def plot_oiino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','OII_No1.eps')
    title = 'O II No1'

    #Low and high state
    low_state = _OII_ground_state
    high_state = _OII_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '5/2', '3729.86   1.98E-6          ')
    emission[1] = (0.20+xoff+0.09, '3/2', '3/2', '3727.10   1.59E-4          ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '5/2', '3729.86   1.98E-6          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '3/2', '3727.10   1.59E-4          ')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

def plot_oiino2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','OII_No2.eps')
    title = 'O II No2'

    #Low and high state
    low_state = _OII_ground_state
    high_state = _OII_excited_state_2

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '1/2', '2470.97   2.12E-2          ')
    emission[1] = (0.20+xoff+0.09, '3/2', '3/2', '2471.09   5.22E-2         ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '1/2', '2470.97   2.12E-2          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '3/2', '2471.09   5.22E-2         ')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

############
# O III
############
_OIII_ground_state =    {'NJ':3, 'CONFIG':r'2s$^2$2p$^2$', 'TERM':r'   $^3\!{\rm P}$','J':np.array(['0', '1', '2']), 'ENERGY':np.array(['0', '113.178', '306.174'])}
_OIII_excited_state_1 =    {'NJ':1, 'CONFIG':r'2s$^2$2p$^2$', 'TERM':r'   $^1\!{\rm D}$','J':np.array(['2']), 'ENERGY':np.array(['20273.27'])}
_OIII_excited_state_2 =    {'NJ':1, 'CONFIG':r'2s$^2$2p$^2$', 'TERM':r'   $^1\!{\rm S}$','J':np.array(['0']), 'ENERGY':np.array(['43185.74'])}
_OIII_ground_finestructure = np.zeros(2, dtype=_transition_dtype)
_OIII_ground_finestructure[0] = (_xfine, '0', '1', r'88.36$\mu{\rm m}$')
_OIII_ground_finestructure[1] = (_xfine, '1', '2', r'51.81$\mu{\rm m}$')
def plot_oiiino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','OIII_No1.eps')
    title = 'O III No1'

    #Low and high state
    low_state = _OIII_ground_state
    high_state = _OIII_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '1', '2', '4960.30   6.21E-3          ')
    emission[1] = (0.20+xoff+0.09, '2', '2', '5008.24   1.81E-2          ')

    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '1', '2', '4960.30   6.21E-3          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '2', '2', '5008.24   1.81E-2          ')
    # fine structure
    finestructure = _OIII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=finestructure, figfile=figfile)

def plot_oiiino2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','OIII_No2.eps')
    title = 'O III No2'

    #Low and high state
    low_state = _OIII_excited_state_1
    high_state = _OIII_excited_state_2

    xoff = 0.24
    #xoff1 = 0.17

    # emission
    nemiss = 1
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '2', '0', '4364.44   1.71E0          ')

    # absorption
    nabs = 1
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '2', '0', '4364.44   1.71E0          ')
    # fine structure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

############
# O I
############
_OI_ground_state =    {'NJ':3, 'CONFIG':r'2s$^2$2p$^4$', 'TERM':r'   $^3\!{\rm P}$','J':np.array(['2', '1', '0']), 'ENERGY':np.array(['0', '158.265', '226.977'])}
_OI_excited_state_1 =    {'NJ':1, 'CONFIG':r'2s$^2$2p$^4$', 'TERM':r'   $^1\!{\rm D}$','J':np.array(['2']), 'ENERGY':np.array(['15867.862'])}
_OI_excited_state_2 =    {'NJ':1, 'CONFIG':r'2s$^2$2p$^4$', 'TERM':r'   $^1\!{\rm S}$','J':np.array(['0']), 'ENERGY':np.array(['33792.583'])}
def plot_oino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','OI_No1.eps')
    title = 'O I No1'

    #Low and high state
    low_state = _OI_ground_state
    high_state = _OI_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '2', '2', '6302.05   5.63E-3          ')
    emission[1] = (0.20+xoff+0.09, '1', '2', '6365.54   1.82E-3          ')

    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '2', '2', '6302.05   5.63E-3          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '1', '2', '6365.54   1.82E-3          ')
    # fine structure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

def plot_oino2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','OI_No2.eps')
    title = 'O I No2'

    #Low and high state
    low_state = _OI_excited_state_1
    high_state = _OI_excited_state_2

    xoff = 0.24
    #xoff1 = 0.17

    # emission
    nemiss = 1
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '2', '0', '5578.89   1.26E0          ')

    # absorption
    nabs = 1
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '2', '0', '5578.89   1.26E0          ')
    # fine structure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

############
# N II
############
_NII_ground_state =    {'NJ':3, 'CONFIG':r'2s$^2$2p$^2$', 'TERM':r'   $^3\!{\rm P}$','J':np.array(['0', '1', '2']), 'ENERGY':np.array(['0', '48.7', '130.8'])}
_NII_excited_state_1 =    {'NJ':1, 'CONFIG':r'2s$^2$2p$^2$', 'TERM':r'   $^1\!{\rm D}$','J':np.array(['2']), 'ENERGY':np.array(['15316.2'])}
_NII_excited_state_2 =    {'NJ':1, 'CONFIG':r'2s$^2$2p$^2$', 'TERM':r'   $^1\!{\rm S}$','J':np.array(['0']), 'ENERGY':np.array(['32688.8'])}
_NII_ground_finestructure = np.zeros(2, dtype=_transition_dtype)
_NII_ground_finestructure[0] = (_xfine, '0', '1', r'205.3$\mu{\rm m}$')
_NII_ground_finestructure[1] = (_xfine, '1', '2', r'121.8$\mu{\rm m}$')

def plot_niino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','NII_No1.eps')
    title = 'N II No1'

    #Low and high state
    low_state = _NII_ground_state
    high_state = _NII_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '1', '2', '6549.86   9.84E-4          ')
    emission[1] = (0.20+xoff+0.09, '2', '2', '6585.27   2.91E-3          ')

    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '1', '2', '6549.86   9.84E-4          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '2', '2', '6585.27   2.91E-3          ')
    # fine structure
    finestructure = _NII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=finestructure, figfile=figfile)

def plot_niino2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','NII_No2.eps')
    title = 'N II No2'

    #Low and high state
    low_state = _NII_excited_state_1
    high_state = _NII_excited_state_2

    xoff = 0.24
    #xoff1 = 0.17

    # emission
    nemiss = 1
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '2', '0', '5756.19   1.14E0          ')

    # absorption
    nabs = 1
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '2', '0', '5756.19   1.14E0          ')
    # fine structure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

############
# N I
############
_NI_ground_state =    {'NJ':1, 'CONFIG':r'2s$^2$2p$^3$', 'TERM':r'   $^4\!{\rm S}^{\rm o}\!$', 'J':np.array(['3/2']), 'ENERGY':np.array(['0'])}
_NI_excited_state_1 = {'NJ':2, 'CONFIG':r'2s$^2$2p$^3$', 'TERM':r'   $^2\!{\rm D}^{\rm o}\!$', 'J':np.array(['5/2', '3/2']), 'ENERGY':np.array(['19224.46', '19233.18'])}
_NI_excited_state_2 = {'NJ':2, 'CONFIG':r'2s$^2$2p$^3$', 'TERM':r'   $^2\!{\rm P}^{\rm o}\!$', 'J':np.array(['1/2', '3/2']), 'ENERGY':np.array(['28838.92', '28839.31'])}

def plot_nino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','NI_No1.eps')
    title = 'N I No1'

    #Low and high state
    low_state = _NI_ground_state
    high_state = _NI_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '5/2', '5201.29   6.59E-6          ')
    emission[1] = (0.20+xoff+0.09, '3/2', '3/2', '5199.35   1.60E-5          ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '5/2', '5201.29   6.59E-6          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '3/2', '5199.35   1.60E-5          ')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

def plot_nino2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','NI_No2.eps')
    title = 'N I No2'

    #Low and high state
    low_state = _NI_ground_state
    high_state = _NI_excited_state_2

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '1/2', '3467.54   2.60E-3          ')
    emission[1] = (0.20+xoff+0.09, '3/2', '3/2', '3467.49   6.5E-3         ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '1/2', '3467.54   2.60E-3          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '3/2', '3467.49   6.5E-3         ')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

############
# S II
############
_SII_ground_state =    {'NJ':1, 'CONFIG':r'3s$^2$3p$^3$', 'TERM':r'   $^4\!{\rm S}^{\rm o}\!$', 'J':np.array(['3/2']), 'ENERGY':np.array(['0'])}
_SII_excited_state_1 = {'NJ':2, 'CONFIG':r'3s$^2$3p$^3$', 'TERM':r'   $^2\!{\rm D}^{\rm o}\!$', 'J':np.array(['3/2', '5/2']), 'ENERGY':np.array(['14852.94', '14884.73'])}
_SII_excited_state_2 = {'NJ':2, 'CONFIG':r'3s$^2$3p$^3$', 'TERM':r'   $^2\!{\rm P}^{\rm o}\!$', 'J':np.array(['1/2', '3/2']), 'ENERGY':np.array(['24524.83', '24571.54'])}

def plot_siino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','SII_No1.eps')
    title = 'S II No1'

    #Low and high state
    low_state = _SII_ground_state
    high_state = _SII_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '3/2', '6732.67   5.63E-4          ')
    emission[1] = (0.20+xoff+0.09, '3/2', '5/2', '6718.30   1.88E-4          ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '3/2', '6732.67   5.63E-4          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '5/2', '6718.30   1.88E-4          ')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

def plot_siino2():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','SII_No2.eps')
    title = 'S II No2'

    #Low and high state
    low_state = _SII_ground_state
    high_state = _SII_excited_state_2

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '3/2', '1/2', '4077.50   7.72E-2          ')
    emission[1] = (0.20+xoff+0.09, '3/2', '3/2', '4069.75   1.92E-1         ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '3/2', '1/2', '4077.50   7.72E-2          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '3/2', '3/2', '4069.75   1.92E-1         ')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)

############
# Ar III
############
_ArIII_ground_state =    {'NJ':3, 'CONFIG':r'3s$^2$3p$^4$', 'TERM':r'   $^3\!{\rm P}$', 'J':np.array(['2', '1', '0']), 'ENERGY':np.array(['0', '1112.17', '1570.23'])}
_ArIII_excited_state_1 = {'NJ':1, 'CONFIG':r'3p$^2$3p$^4$', 'TERM':r'   $^1\!{\rm D}$', 'J':np.array(['2']), 'ENERGY':np.array(['14010.00'])}
#_ArIII_excited_state_2 = {'NJ':1, 'CONFIG':r'2p$^6$4s', 'TERM':r'   $^2\!{\rm S}$', 'J':np.array(['1/2']), 'ENERGY':np.array(['25740.00'])}

def plot_ariiino1():
    figfile = join(datapath.atomic_path(),'EnergyDiagram','ArIII_No1.eps')
    title = 'Ar III No1'

    #Low and high state
    low_state = _ArIII_ground_state
    high_state = _ArIII_excited_state_1

    xoff = 0.22
    #xoff1 = 0.17

    # emission
    nemiss = 2
    emission = np.zeros(nemiss, dtype=_transition_dtype)
    emission[0] = (0.20+xoff, '2', '2', '7137.76   3.21E-1          ')
    emission[1] = (0.20+xoff+0.09, '1', '2', '7753.24   8.30E-2          ')
    # absorption
    nabs = 2
    absorption = np.zeros(nabs, dtype=_transition_dtype)
    absorption[0] = (0.20+xoff-0.008, '2', '2', '7137.76   3.21E-1          ')
    absorption[1] = (0.20+xoff+0.09-0.008, '1', '2', '7753.24   8.30E-2          ')
    # fine structure
    #finestructure = _TiII_ground_finestructure

    plot_energylevel_diagram(low_state, high_state, title, emission=emission, absorption=None, finestructure=None, figfile=figfile)


