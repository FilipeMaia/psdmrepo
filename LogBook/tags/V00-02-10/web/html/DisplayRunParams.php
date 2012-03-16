<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use RegDB\RegDBHtml;
use RegDB\RegDBException;

/*
 * This script will process a request for displaying parameters of a run.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "run identifier can't be empty" );
} else
    die( "no valid run identifier" );

/* Build a dictionary of sections
*/
$sections = array(

    /* Groups of parameters which are common for all instruments
     * and experiments
     */
    'HEADER' => array(

        array(

            'SECTION' => 'BEAMS',
            'TITLE'   => '&nbsp;E l e c t r o n &nbsp;&nbsp; a n d &nbsp;&nbsp; P h o t o n &nbsp;&nbsp; b e a m s',
            'PARAMS'  => array(

                array( 'name' => 'BEND:DMP1:400:BDES',       'descr' => 'electron beam energy' ),
                array( 'name' => 'EVNT:SYS0:1:LCLSBEAMRATE', 'descr' => 'beam rep rate' ),
                array( 'name' => 'BPMS:DMP1:199:TMIT1H',     'descr' => 'Particle N_electrons' ),
                array( 'name' => 'SIOC:SYS0:ML00:AO289',     'descr' => 'E.Vernier' ),
                array( 'name' => 'BEAM:LCLS:ELEC:Q',         'descr' => 'Charge' ),
                array( 'name' => 'SIOC:SYS0:ML00:AO195',     'descr' => 'Peak current after second bunch compressor' ),
                array( 'name' => 'SIOC:SYS0:ML00:AO820',     'descr' => 'Pulse length' ),
                array( 'name' => 'SIOC:SYS0:ML00:AO569',     'descr' => 'ebeam energy loss converted to photon mJ' ),
                array( 'name' => 'SIOC:SYS0:ML00:AO580',     'descr' => 'Calculated number of photons' ),
                array( 'name' => 'SIOC:SYS0:ML00:AO541',     'descr' => 'Photon beam energy' ),
                array( 'name' => 'SIOC:SYS0:ML00:AO627',     'descr' => 'Photon beam energy' ),
                array( 'name' => 'SIOC:SYS0:ML00:AO192',     'descr' => 'Wavelength' )
            )
        ),

        array(

            'SECTION' => 'FEE',
            'TITLE'   => '&nbsp;F E E',
            'PARAMS'  => array(

                array( 'name' => 'VGPR:FEE1:311:PSETPOINT_DES', 'descr' => 'Gas attenuator setpoint' ),
                array( 'name' => 'VGCP:FEE1:311:P',             'descr' => 'Gas attenuator actual pressure' ),
                array( 'name' => 'GATT:FEE1:310:R_ACT',         'descr' => 'Gas attenuator calculated transmission' ),
                array( 'name' => 'SATT:FEE1:321:STATE',         'descr' => 'Solid attenuator 1' ),
                array( 'name' => 'SATT:FEE1:322:STATE',         'descr' => 'Solid attenuator 2' ),
                array( 'name' => 'SATT:FEE1:323:STATE',         'descr' => 'Solid attenuator 3' ),
                array( 'name' => 'SATT:FEE1:324:STATE',         'descr' => 'Solid attenuator 4' ),
                array( 'name' => 'SATT:FEE1:320:TACT',          'descr' => 'Total attenuator length' ),
                array( 'name' => 'LVDT:FEE1:1811:LVPOS',        'descr' => 'FEE mirror LVDT position' ),
                array( 'name' => 'LVDT:FEE1:1812:LVPOS',        'descr' => 'FEE mirror LVDT position' ),
                array( 'name' => 'STEP:FEE1:1811:MOTR.RBV',     'descr' => 'FEE mirror RBV position' ),
                array( 'name' => 'STEP:FEE1:1812:MOTR.RBV',     'descr' => 'FEE mirror RBV position' )
            )
        )
    ),

    'FOOTER' => array(

        'TITLE'     => '&nbsp;A d d i t i o n a l &nbsp;&nbsp; P a r a m e t e r s',
        'PAR2DESCR' => array(
            'AMO:DIA:GMP:06:PMON' => 'Pressure (Torr)'
        )
    ),

    /* Instrument-specific groups of parameters.
     */
    'AMO' => array(

        array(

            'SECTION' => 'HFP',
            'TITLE'   => '&nbsp;H F P',
            'PARAMS'  => array(

                array( 'name' => 'AMO:HFP:GCC:01:PMON', 'descr' => 'pressure' ),
                array( 'name' => 'AMO:HFP:MMS:table.Z', 'descr' => 'z-position' )
            )
        ),

        array(

            'SECTION' => 'ETOF',
            'TITLE'   => '&nbsp;e T O F',
            'PARAMS'  => array(

                array( 'name' => 'AMO:R14:IOC:10:ao0:out1',                'descr' => '1' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS0:CH0:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS0:CH1:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS0:CH2:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS0:CH3:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS7:CH0:VoltageMeasure', 'descr' => '' ),

                array( 'name' => 'AMO:R14:IOC:10:ao0:out2',                'descr' => '2' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS1:CH0:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS1:CH1:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS1:CH2:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS1:CH3:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS7:CH1:VoltageMeasure', 'descr' => '' ),

                array( 'name' => 'AMO:R14:IOC:10:ao0:out3',                'descr' => '3' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS2:CH0:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS2:CH1:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS2:CH2:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS2:CH3:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS7:CH2:VoltageMeasure', 'descr' => '' ),

                array( 'name' => 'AMO:R14:IOC:10:ao0:out4',                'descr' => '4' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS3:CH0:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS3:CH1:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS3:CH2:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS3:CH3:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS7:CH3:VoltageMeasure', 'descr' => '' ),

                array( 'name' => 'AMO:R14:IOC:10:ao0:out5',                'descr' => '5' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS4:CH0:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS4:CH1:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS4:CH2:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS4:CH3:VoltageMeasure', 'descr' => '' ),
                array( 'name' => 'AMO:R14:IOC:10:VHS8:CH0:VoltageMeasure', 'descr' => '' )
            )
        ),

        array(

            'SECTION' => 'ITOF',
            'TITLE'   => '&nbsp;i T O F',
            'PARAMS'  => array(

                array( 'name' => 'AMO:R14:IOC:21:VHS2:CH0:VoltageMeasure', 'descr' => 'repeller' ),
                array( 'name' => 'AMO:R14:IOC:21:VHS0:CH0:VoltageMeasure', 'descr' => 'extractor' ),
                array( 'name' => 'AMO:R14:IOC:21:VHS0:CH1:VoltageMeasure', 'descr' => 'acceleration' ),
                array( 'name' => 'AMO:R14:IOC:21:VHS0:CH2:VoltageMeasure', 'descr' => 'MCP in' ),
                array( 'name' => 'AMO:R14:IOC:21:VHS2:CH2:VoltageMeasure', 'descr' => 'MCP out' ),
                array( 'name' => 'AMO:R14:IOC:21:VHS2:CH1:VoltageMeasure', 'descr' => 'Anode' )
            )
        ),

        array(

            'SECTION' => 'HFP_GAS',
            'TITLE'   => '&nbsp;H F P &nbsp;&nbsp; G a s',
            'PARAMS'  => array(

                array( 'name' => 'AMO:HFP:GCC:03:PMON',                    'descr' => 'pressure' ),
                array( 'name' => 'AMO:R14:IOC:21:VHS7:CH0:VoltageMeasure', 'descr' => 'piezo voltage' ),
                array( 'name' => 'AMO:R14:EVR:21:CTRL.DG2D',               'descr' => 'piezo timing delay' ),
                array( 'name' => 'AMO:R14:EVR:21:CTRL.DG2W',               'descr' => 'piezo timing width' ),
                array( 'name' => 'AMO:HFP:MMS:72.RBV',                     'descr' => 'gasjet x-position (rel. distance)' ),
                array( 'name' => 'AMO:HFP:MMS:71.RBV',                     'descr' => 'gasjet y-position (rel. distance)' ),
                array( 'name' => 'AMO:HFP:MMS:73.RBV',                     'descr' => 'Gas Jet motor Z axis (mm)' )
            )
        ),

        array(

            'SECTION' => 'DIA',
            'TITLE'   => '&nbsp;D I A ',
            'PARAMS'  => array(

                array( 'name' => 'AMO:DIA:GCC:01:PMON', 'descr' => 'pressure' )
            )
        ),

        array(

            'SECTION' => 'MBES',
            'TITLE'   => '&nbsp;M B E S',
            'PARAMS'  => array(

                array( 'name' => 'AMO:DIA:SHC:11:I', 'descr' => 'coil 1' ),
                array( 'name' => 'AMO:DIA:SHC:12:I', 'descr' => 'coil 2' ),

                array( 'name' => 'AMO:R15:IOC:40:VHS0:CH0:VoltageSet', 'descr' => '' ),
                array( 'name' => 'AMO:R15:IOC:40:VHS0:CH1:VoltageSet', 'descr' => '' ),
                array( 'name' => 'AMO:R15:IOC:40:VHS2:CH1:VoltageSet', 'descr' => '' ),
                array( 'name' => 'AMO:R15:IOC:40:VHS2:CH2:VoltageSet', 'descr' => '' )
            )
        )
    ),
        
    'SXR' => array(
    
        array(

            'SECTION' => 'COL',
            'TITLE'   => '&nbsp;C o l l i m a t o r',
            'PARAMS'  => array(

                array( 'name' => 'SXR:COL:GCC:01:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:COL:PIP:01:PMON', 'descr' => '' )
            )
        ),

        array(

            'SECTION' => 'EXS',
            'TITLE'   => '&nbsp;E x i t &nbsp;&nbsp; S l i t',
            'PARAMS'  => array(

                array( 'name' => 'SXR:EXS:GCC:01:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:EXS:MMS:01.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:EXS:PIP:01:PMON', 'descr' => '' )
            )
        ),

        array(

            'SECTION' => 'FLX',
            'TITLE'   => '&nbsp;F l u x &nbsp;&nbsp; c h a m b e r',
            'PARAMS'  => array(

                array( 'name' => 'SXR:FLX:GCC:01:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:FLX:STC:01', 'descr' => '' )
            )
        ),

        array(

            'SECTION' => 'KBO',
            'TITLE'   => '&nbsp;K B O &nbsp;&nbsp; O p t i c s',
            'PARAMS'  => array(

                array( 'name' => 'SXR:KBO:GCC:01:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:KBO:GCC:02:PMON', 'descr' => '' )
            )
        ),

        array(

            'SECTION' => 'LIN',
            'TITLE'   => '&nbsp;L a s e r &nbsp;&nbsp; I n c o u p l i n g',
            'PARAMS'  => array(

                array( 'name' => 'SXR:LIN:GCC:01:PMON', 'descr' => '' )
            )
        ),

        array(

            'SECTION' => 'MON',
            'TITLE'   => '&nbsp;G r a t i n g',
            'PARAMS'  => array(

                array( 'name' => 'SXR:MON:GCC:01:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:MON:MMS:01.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:MON:MMS:02.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:MON:MMS:03.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:MON:MMS:04.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:MON:MMS:05.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:MON:MMS:06.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:MON:MMS:07.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:MON:MMS:08.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:MON:PIP:01:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:MON:STC:01',      'descr' => '' ),
                array( 'name' => 'SXR:MON:STC:02',      'descr' => '' )
            )
        ),

        array(

            'SECTION' => 'PST',
            'TITLE'   => '&nbsp;P h o t o n &nbsp;&nbsp; S t o p p e r',
            'PARAMS'  => array(

                array( 'name' => 'SXR:PST:PIP:01:PMON', 'descr' => '' ),
                array( 'name' => 'PPS:NEH1:2:S2STPRSUM', 'descr' => '' )
            )
        ),

        array(

            'SECTION' => 'SPS',
            'TITLE'   => '&nbsp;S i n g l e &nbsp;&nbsp; P u l s e &nbsp;&nbsp; S h u t t e r',
            'PARAMS'  => array(

                array( 'name' => 'SXR:SPS:GCC:01:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:SPS:MPA:01:OUT',  'descr' => '' ),
                array( 'name' => 'SXR:SPS:PIP:01:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:SPS:STC:01',      'descr' => '' )
            )
        ),

        array(

            'SECTION' => 'TSS',
            'TITLE'   => '&nbsp;T r a n s m i s s i o n &nbsp;&nbsp; S a m p l e &nbsp;&nbsp; S t a g e',
            'PARAMS'  => array(

                array( 'name' => 'SXR:TSS:GCC:01:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:TSS:GCC:02:PMON', 'descr' => '' ),
                array( 'name' => 'SXR:TSS:MMS:01.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:TSS:MMS:02.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:TSS:MMS:03.VAL',  'descr' => '' ),
                array( 'name' => 'SXR:TSS:PIP:01:PMON', 'descr' => '' )
            )
        )
    ),

    'XPP' => array(
    ),

    'CXI' => array(

    	array(

            'SECTION' => 'undulator',
            'TITLE'   => '&nbsp;U n d u l a t o r &nbsp;&nbsp; S t a t u s',
            'PARAMS'  => array(
                array( 'descr' => 'Undulator  1 X pos', 'name' => 'USEG:UND1:150:XIN' ),
                array( 'descr' => 'Undulator  2 X pos', 'name' => 'USEG:UND1:250:XIN' ),
                array( 'descr' => 'Undulator  3 X pos', 'name' => 'USEG:UND1:350:XIN' ),
                array( 'descr' => 'Undulator  4 X pos', 'name' => 'USEG:UND1:450:XIN' ),
                array( 'descr' => 'Undulator  5 X pos', 'name' => 'USEG:UND1:550:XIN' ),
                array( 'descr' => 'Undulator  6 X pos', 'name' => 'USEG:UND1:650:XIN' ),
                array( 'descr' => 'Undulator  7 X pos', 'name' => 'USEG:UND1:750:XIN' ),
                array( 'descr' => 'Undulator  8 X pos', 'name' => 'USEG:UND1:850:XIN' ),
                array( 'descr' => 'Undulator  9 X pos', 'name' => 'USEG:UND1:950:XIN' ),
                array( 'descr' => 'Undulator 10 X pos', 'name' => 'USEG:UND1:1050:XIN' ),
                array( 'descr' => 'Undulator 11 X pos', 'name' => 'USEG:UND1:1150:XIN' ),
                array( 'descr' => 'Undulator 12 X pos', 'name' => 'USEG:UND1:1250:XIN' ),
                array( 'descr' => 'Undulator 13 X pos', 'name' => 'USEG:UND1:1350:XIN' ),
                array( 'descr' => 'Undulator 14 X pos', 'name' => 'USEG:UND1:1450:XIN' ),
                array( 'descr' => 'Undulator 15 X pos', 'name' => 'USEG:UND1:1550:XIN' ),
                array( 'descr' => 'Undulator 16 X pos', 'name' => 'USEG:UND1:1650:XIN' ),
                array( 'descr' => 'Undulator 17 X pos', 'name' => 'USEG:UND1:1750:XIN' ),
                array( 'descr' => 'Undulator 18 X pos', 'name' => 'USEG:UND1:1850:XIN' ),
                array( 'descr' => 'Undulator 19 X pos', 'name' => 'USEG:UND1:1950:XIN' ),
                array( 'descr' => 'Undulator 20 X pos', 'name' => 'USEG:UND1:2050:XIN' ),
                array( 'descr' => 'Undulator 21 X pos', 'name' => 'USEG:UND1:2150:XIN' ),
                array( 'descr' => 'Undulator 22 X pos', 'name' => 'USEG:UND1:2250:XIN' ),
                array( 'descr' => 'Undulator 23 X pos', 'name' => 'USEG:UND1:2350:XIN' ),
                array( 'descr' => 'Undulator 24 X pos', 'name' => 'USEG:UND1:2450:XIN' ),
                array( 'descr' => 'Undulator 25 X pos', 'name' => 'USEG:UND1:2550:XIN' ),
                array( 'descr' => 'Undulator 26 X pos', 'name' => 'USEG:UND1:2650:XIN' ),
                array( 'descr' => 'Undulator 27 X pos', 'name' => 'USEG:UND1:2750:XIN' ),
                array( 'descr' => 'Undulator 28 X pos', 'name' => 'USEG:UND1:2850:XIN' )
            )
        ),
    	array(

            'SECTION' => 'FEE data',
            'TITLE'   => '&nbsp;F E E &nbsp;&nbsp; D a t a',
            'PARAMS'  => array(
                array( 'descr' => 'FEE mask slit X- blade', 'name' => 'STEP:FEE1:151:MOTR.RBV'   ),
                array( 'descr' => 'FEE mask slit X+ blade', 'name' => 'STEP:FEE1:152:MOTR.RBV'   ),
                array( 'descr' => 'FEE mask slit Y- blade', 'name' => 'STEP:FEE1:153:MOTR.RBV'   ),
                array( 'descr' => 'FEE mask slit Y+ blade', 'name' => 'STEP:FEE1:154:MOTR.RBV'   ),
                array( 'descr' => 'FEE mask slit X center', 'name' => 'SLIT:FEE1:ACTUAL_XCENTER' ),
                array( 'descr' => 'FEE mask slit X width',  'name' => 'SLIT:FEE1:ACTUAL_XWIDTH'  ),
                array( 'descr' => 'FEE mask slit Y center', 'name' => 'SLIT:FEE1:ACTUAL_YCENTER' ),
                array( 'descr' => 'FEE mask slit Y width',  'name' => 'SLIT:FEE1:ACTUAL_YWIDTH'  ),
                array( 'descr' => 'FEE M1H X pos',          'name' => 'STEP:FEE1:611:MOTR.RBV'   ),
                array( 'descr' => 'FEE M1H dX',             'name' => 'STEP:FEE1:612:MOTR.RBV'   ),
                array( 'descr' => 'FEE M2H X pos',          'name' => 'STEP:FEE1:861:MOTR.RBV'   ),
                array( 'descr' => 'FEE M2H dX',             'name' => 'STEP:FEE1:862:MOTR.RBV'   ),
                array( 'descr' => 'FEE M1H X LVDT',         'name' => 'LVDT:FEE1:611:LVPOS'      ),
                array( 'descr' => 'FEE M1H dX LVDT',        'name' => 'LVDT:FEE1:612:LVPOS'      ),
                array( 'descr' => 'FEE M2H X LVDT',         'name' => 'LVDT:FEE1:861:LVPOS'      ),
                array( 'descr' => 'FEE M2H dX LVDT',        'name' => 'LVDT:FEE1:862:LVPOS'      ),
                array( 'descr' => 'FEE attenuator 1',       'name' => 'SATT:FEE1:321:STATE'      ),
                array( 'descr' => 'FEE attenuator 2',       'name' => 'SATT:FEE1:322:STATE'      ),
                array( 'descr' => 'FEE attenuator 3',       'name' => 'SATT:FEE1:323:STATE'      ),
                array( 'descr' => 'FEE attenuator 4',       'name' => 'SATT:FEE1:324:STATE'      ),
                array( 'descr' => 'FEE attenuator 5',       'name' => 'SATT:FEE1:325:STATE'      ),
                array( 'descr' => 'FEE attenuator 6',       'name' => 'SATT:FEE1:326:STATE'      ),
                array( 'descr' => 'FEE attenuator 7',       'name' => 'SATT:FEE1:327:STATE'      ),
                array( 'descr' => 'FEE attenuator 8',       'name' => 'SATT:FEE1:328:STATE'      ),
                array( 'descr' => 'FEE attenuator 9',       'name' => 'SATT:FEE1:329:STATE'      ),
                array( 'descr' => 'FEE total attenuator',   'name' => 'SATT:FEE1:320:TACT'       )
    		)
        ),
    	array(

            'SECTION' => 'CXI',
            'TITLE'   => '&nbsp;C X I',
            'PARAMS'  => array(
                array( 'descr' => 'Unfocused',        'name' => 'CXI:MPS:CFG:1_MPSC' ),
                array( 'descr' => '10 um XRT lens',   'name' => 'CXI:MPS:CFG:2_MPSC' ),
                array( 'descr' => '1 um DG2 lens',    'name' => 'CXI:MPS:CFG:3_MPSC' ),
                array( 'descr' => '1 um KB mirror',   'name' => 'CXI:MPS:CFG:4_MPSC' ),
                array( 'descr' => '100 nm KB mirror', 'name' => 'CXI:MPS:CFG:5_MPSC' )
    		)
        ),
    	array(

            'SECTION' => 'DIA',
            'TITLE'   => '&nbsp;D I A',
            'PARAMS'  => array(
                array( 'descr' => '20 um Si foil',    'name' => 'XRT:DIA:MMS:02.RBV' ),
                array( 'descr' => '40 um Si foil',    'name' => 'XRT:DIA:MMS:03.RBV' ),
                array( 'descr' => '80 um Si foil',    'name' => 'XRT:DIA:MMS:04.RBV' ),
                array( 'descr' => '160 um Si foil',   'name' => 'XRT:DIA:MMS:05.RBV' ),
                array( 'descr' => '320 um Si foil',   'name' => 'XRT:DIA:MMS:06.RBV' ),
                array( 'descr' => '640 um Si foil',   'name' => 'XRT:DIA:MMS:07.RBV' ),
                array( 'descr' => '1280 um Si foil',  'name' => 'XRT:DIA:MMS:08.RBV' ),
                array( 'descr' => '2560 um Si foil',  'name' => 'XRT:DIA:MMS:09.RBV' ),
                array( 'descr' => '5120 um Si foil',  'name' => 'XRT:DIA:MMS:10.RBV' ),
                array( 'descr' => '10240 um Si foil', 'name' => 'XRT:DIA:MMS:11.RBV' ),
                array( 'descr' => 'XRT lens out',     'name' => 'XRT:DIA:MMS:14.HLS' ),
                array( 'descr' => 'XRT lens Y pos',   'name' => 'XRT:DIA:MMS:14.RBV' )
	   		)
        ),
    	array(

            'SECTION' => 'DG1',
            'TITLE'   => '&nbsp;D G 1',
            'PARAMS'  => array(
                array( 'descr' => 'DG1 slit X center', 'name' => 'CXI:DG1:JAWS:XTRANS.C' ),
                array( 'descr' => 'DG1 slit X width',  'name' => 'CXI:DG1:JAWS:YTRANS.C' ),
                array( 'descr' => 'DG1 slit Y center', 'name' => 'CXI:DG1:JAWS:XTRANS.D' ),
                array( 'descr' => 'DG1 slit Y width',  'name' => 'CXI:DG1:JAWS:YTRANS.D' ),
                array( 'descr' => 'DG1 Navitar Zoom',  'name' => 'CXI:DG1:CLZ:01.RBV'    )
            )
        ),
    	array(

            'SECTION' => 'DG2',
            'TITLE'   => '&nbsp;D G 2',
            'PARAMS'  => array(
                array( 'descr' => 'DS2/DG2 valve open',   'name' => 'CXI:DG2:VGC:01:OPN_DI' ),
                array( 'descr' => 'DS2/DG2 valve closed', 'name' => 'CXI:DG2:VGC:01:CLS_DI' ),
                array( 'descr' => 'DG2 slit X center',    'name' => 'CXI:DG2:JAWS:XTRANS.C' ),
                array( 'descr' => 'DG2 slit X width',     'name' => 'CXI:DG2:JAWS:YTRANS.C' ),
                array( 'descr' => 'DG2 slit Y center',    'name' => 'CXI:DG2:JAWS:XTRANS.D' ),
                array( 'descr' => 'DG2 slit Y width',     'name' => 'CXI:DG2:JAWS:YTRANS.D' ),
                array( 'descr' => 'DG2 IPM diode Y pos',  'name' => 'CXI:DG2:MMS:08.RBV'    ),
                array( 'descr' => 'DG2 IPM target Y pos', 'name' => 'CXI:DG2:MMS:10.RBV'    ),
                array( 'descr' => 'DG2 lens out',         'name' => 'CXI:DG2:MMS:06.HLS'    ),
                array( 'descr' => 'DG2 lens Y pos',       'name' => 'CXI:DG2:MMS:06.RBV'    ),
                array( 'descr' => 'DG2 Navitar zoom',     'name' => 'CXI:DG2:CLZ:01.RBV'    ),
                array( 'descr' => 'DG2/DSU valve open',   'name' => 'CXI:DG2:VGC:02:OPN_DI' ),
                array( 'descr' => 'DG2/DSU valve closed', 'name' => 'CXI:DG2:VGC:02:CLS_DI' )
    		)
        ),
    	array(

            'SECTION' => 'KB1',
            'TITLE'   => '&nbsp;K B 1',
            'PARAMS'  => array(
                array( 'descr' => 'KB1 chamber pressure',   'name' => 'CXI:KB1:GCC:02:PMON' ),
                array( 'descr' => 'KB1 slit (US) X center', 'name' => 'CXI:KB1:JAWS:US:XTRANS.C' ),
                array( 'descr' => 'KB1 slit (US) X width',  'name' => 'CXI:KB1:JAWS:US:YTRANS.C' ),
                array( 'descr' => 'KB1 slit (US) Y center', 'name' => 'CXI:KB1:JAWS:US:XTRANS.D' ),
                array( 'descr' => 'KB1 slit (US) Y width',  'name' => 'CXI:KB1:JAWS:US:YTRANS.D' ),
                array( 'descr' => 'KB1 Horizontal X pos',   'name' => 'CXI:KB1:MMS:05.RBV' ),
                array( 'descr' => 'KB1 Horizontal Y pos',   'name' => 'CXI:KB1:MMS:06.RBV' ),
                array( 'descr' => 'KB1 Horizontal pitch',   'name' => 'CXI:KB1:MMS:07.RBV' ),
                array( 'descr' => 'KB1 Horizontal roll',    'name' => 'CXI:KB1:MMS:08.RBV' ),
                array( 'descr' => 'KB1 Vertical X pos',     'name' => 'CXI:KB1:MMS:09.RBV' ),
                array( 'descr' => 'KB1 Vertical Y pos',     'name' => 'CXI:KB1:MMS:10.RBV' ),
                array( 'descr' => 'KB1 Vertical pitch',     'name' => 'CXI:KB1:MMS:11.RBV' ),
                array( 'descr' => 'KB1 slit (DS) X center', 'name' => 'CXI:KB1:JAWS:DS:XTRANS.C' ),
                array( 'descr' => 'KB1 slit (DS) X width',  'name' => 'CXI:KB1:JAWS:DS:YTRANS.C' ),
                array( 'descr' => 'KB1 slit (US) Y center', 'name' => 'CXI:KB1:JAWS:DS:XTRANS.D' ),
                array( 'descr' => 'KB1 slit (US) Y width',  'name' => 'CXI:KB1:JAWS:DS:YTRANS.D' ),
                array( 'descr' => 'KB1 Navitar Zoom',       'name' => 'CXI:KB1:CLZ:01.RBV' )
    		)
        ),
    	array(

            'SECTION' => 'KB2',
            'TITLE'   => '&nbsp;K B 2',
            'PARAMS'  => array(
                array( 'descr' => 'DSU slit X center', 'name' => 'CXI:DSU:JAWS:XTRANS.C' ),
                array( 'descr' => 'DSU slit X width',  'name' => 'CXI:DSU:JAWS:YTRANS.C' ),
                array( 'descr' => 'DSU slit Y center', 'name' => 'CXI:DSU:JAWS:XTRANS.D' ),
                array( 'descr' => 'DSU slit Y width',  'name' => 'CXI:DSU:JAWS:YTRANS.D' )
    		)
        ),
    	array(

            'SECTION' => 'DSU',
            'TITLE'   => '&nbsp;D S U',
            'PARAMS'  => array(
    		)
        ),
    	array(

            'SECTION' => 'SC1',
            'TITLE'   => '&nbsp;S C 1',
            'PARAMS'  => array(
                array( 'descr' => 'SC1 chamber pressure',   'name' => 'CXI:SC1:GCC:01:PMON'           ),
                array( 'descr' => 'DSU/SC1 valve open',     'name' => 'CXI:SC1:VGC:01:OPN_DI'         ),
                array( 'descr' => 'DSU/SC1 valve closed',   'name' => 'CXI:SC1:VGC:01:CLS_DI'         ),
                array( 'descr' => 'SC1 MZM aperture 1 X',   'name' => 'CXI:SC1:MZM:01:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM aperture 1 Y',   'name' => 'CXI:SC1:MZM:02:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM aperture 2 X',   'name' => 'CXI:SC1:MZM:03:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM aperture 2 Y',   'name' => 'CXI:SC1:MZM:04:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM aperture 3 X',   'name' => 'CXI:SC1:MZM:05:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM aperture 3 Y',   'name' => 'CXI:SC1:MZM:06:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM aperture 3 Z',   'name' => 'CXI:SC1:MZM:07:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM sample X',       'name' => 'CXI:SC1:MZM:08:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM sample Y',       'name' => 'CXI:SC1:MZM:09:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM sample Z',       'name' => 'CXI:SC1:MZM:10:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM part aper X',    'name' => 'CXI:SC1:MZM:12:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM part aper Z',    'name' => 'CXI:SC1:MZM:13:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM view mirror X',  'name' => 'CXI:SC1:MZM:14:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 MZM view mirror Y',  'name' => 'CXI:SC1:MZM:15:ENCPOSITIONGET' ),
                array( 'descr' => 'SC1 sample yaw',         'name' => 'CXI:SC1:PIC:01.RBV'            ),
                array( 'descr' => 'SC1 sample pitch/yaw 1', 'name' => 'CXI:SC1:PIC:02.RBV'            ),
                array( 'descr' => 'SC1 sample pitch/yaw 2', 'name' => 'CXI:SC1:PIC:03.RBV'            ),
                array( 'descr' => 'SC1 sample pitch/yaw 3', 'name' => 'CXI:SC1:PIC:04.RBV'            ),
                array( 'descr' => 'SC1 sample x (long)',    'name' => 'CXI:SC1:MMS:02.RBV'            ),
                array( 'descr' => 'SC1/DS1 valve open',     'name' => 'CXI:SC1:VGC:02:OPN_DI'         ),
                array( 'descr' => 'SC1/DS1 valve closed',   'name' => 'CXI:SC1:VGC:02:CLS_DI'         )
    		)
        ),
    	array(

            'SECTION' => 'DS1',
            'TITLE'   => '&nbsp;D S 1',
            'PARAMS'  => array(
                array( 'descr' => 'DS1 chamber pressure',  'name' => 'CXI:DS1:GCC:01:PMON'              ),
                array( 'descr' => 'DS1 detector Z pos',    'name' => 'CXI:DS1:MMS:06.RBV'               ),
                array( 'descr' => 'DS1 stick Y pos',       'name' => 'CXI:DS1:MMS:07.RBV'               ),
                array( 'descr' => 'DS1 chiller temp',      'name' => 'CXI:DS1:TEMPERATURE'              ),
                array( 'descr' => 'DS1 chiller flowmeter', 'name' => 'CXI:DS1:FLOW_METER'               ),
                array( 'descr' => 'DS1 quad 0 temp',       'name' => 'CXI:DS1:TE-TECH1:ACTUAL_TEMP'     ),
                array( 'descr' => 'DS1 quad 1 temp',       'name' => 'CXI:DS1:TE-TECH1:ACTUAL_SEC_TEMP' ),
                array( 'descr' => 'DS1 quad 2 temp',       'name' => 'CXI:DS1:TE-TECH2:ACTUAL_TEMP'     ),
                array( 'descr' => 'DS1 quad 3 temp',       'name' => 'CXI:DS1:TE-TECH2:ACTUAL_SEC_TEMP' ),
                array( 'descr' => 'DS1 bias voltage',      'name' => 'CXI:DS1:BIAS'                     )
    		)
        ),
    	array(

            'SECTION' => 'DSD',
            'TITLE'   => '&nbsp;D S D',
            'PARAMS'  => array(
                array( 'descr' => 'DSD chamber pressure',  'name' => 'CXI:DSD:GCC:01:PMON'   ),
                array( 'descr' => 'DSD detector Z pos',    'name' => 'CXI:DSD:MMS:06.RBV'    ),
                array( 'descr' => 'DSD chiller temp',      'name' => 'CXI:DS1:TEMPERATURE'   ),
                array( 'descr' => 'DSD chiller flowmeter', 'name' => 'CXI:DS1:FLOW_METER'    ),
                array( 'descr' => '1MS/DG3 valve open',    'name' => 'CXI:DS1:VGC:01:OPN_DI' ),
                array( 'descr' => '1MS/DG3 valve closed',  'name' => 'CXI:DS1:VGC:01:CLS_DI' )
    		)
        ),
    	array(

            'SECTION' => 'DG4',
            'TITLE'   => '&nbsp;D G 4',
            'PARAMS'  => array(
                array( 'descr' => 'DG4 IPM diode Y',  'name' => 'CXI:DG4:MMS:02.RBV' ),
                array( 'descr' => 'DG4 IPM target Y', 'name' => 'CXI:DG4:MMS:03.RBV' ),
                array( 'descr' => 'DG4 Navitar zoom', 'name' => 'CXI:DG4:CLZ:01.RBV' )
    		)
        ),
    	array(

            'SECTION' => 'USR',
            'TITLE'   => '&nbsp;U S R',
            'PARAMS'  => array(
                array( 'descr' => 'User motor ch  1',        'name' => 'CXI:USR:MMS:01.RBV'   ),
                array( 'descr' => 'User motor ch  2',        'name' => 'CXI:USR:MMS:02.RBV'   ),
                array( 'descr' => 'User motor ch  3',        'name' => 'CXI:USR:MMS:03.RBV'   ),
                array( 'descr' => 'User motor ch  4',        'name' => 'CXI:USR:MMS:04.RBV'   ),
                array( 'descr' => 'User motor ch  5',        'name' => 'CXI:USR:MMS:05.RBV'   ),
                array( 'descr' => 'User motor ch  6',        'name' => 'CXI:USR:MMS:06.RBV'   ),
                array( 'descr' => 'User motor ch 17',        'name' => 'CXI:USR:MMS:17.RBV'   ),
                array( 'descr' => 'User motor ch 18',        'name' => 'CXI:USR:MMS:18.RBV'   ),
                array( 'descr' => 'User motor ch 19',        'name' => 'CXI:USR:MMS:19.RBV'   ),
                array( 'descr' => 'Current gas pressure 1',  'name' => 'CXI:R52:AI:PRES_REG1' ),
                array( 'descr' => 'Total pressure change 1', 'name' => 'CXI:R52:UPDATE1'      ),
                array( 'descr' => 'Current gas pressure 2',  'name' => 'CXI:R52:AI:PRES_REG2' ),
                array( 'descr' => 'Total pressure change 2', 'name' => 'CXI:R52:UPDATE2'      )
    		)
        )
    )
);

/* A dictionary of known per-run attribute sections
 */
$attribute_sections = array (
    'DAQ_Detectors' => '&nbsp;D A Q&nbsp;&nbsp;D e t e c t o r s'
);

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $run = $logbook->find_run_by_id( $id )
        or die( "no such run" );

    $experiment = $run->parent();
    $instrument = $experiment->instrument();

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to access any information about the experiment' ));
        exit;
    }

    // Get values for existing run parameters
    //
    $return_dict = true;
    $values = $run->values( '', $return_dict );

    // Proceed to the operation
    //
    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $value_color = 'maroon';
    $label_color = '#b0b0b0';

    $num_cols = 775;

    $con = new RegDBHtml( 0, 0, $num_cols );

    $row = 0;

    /* Normal processing for common and instrument-specific parameters
     */
    $used_names = array();  // Remember names of parameters displayed
                            // in common and instrument-specific sections.
                            // We're going to use this information later to put
                            // remaining parameters into the footer.

    foreach( array( 'HEADER', $instrument->name()) as $area ) {

    	foreach( $sections[$area] as $s ) {

            $con->label_1(0, $row, $s['TITLE'], $num_cols );
            $row += 30;

          	foreach( $s['PARAMS'] as $p ) {
                $name  = $p['name'];
                $value = array_key_exists( $name, $values ) ? $values[$name]->value() : '&lt; no data &gt;';
                $decsr = $p['descr'];
                $con->value( 10, $row, '<i>'.$decsr.'</i>' )
                    ->value(300, $row, $value, $value_color )
                    ->label(500, $row, $name, true, $label_color );
                $row += 20;
                
                $used_names[$name] = True;
            }
            $row += 15;
  	    }
    }

    /* Special processing for experiment-specific parameters  not found
     * in the dictionary.
     */
    $con->label_1(0, $row, $sections['FOOTER']['TITLE'], $num_cols );
    $row += 30;

    foreach( $values as $p ) {
        $name  = $p->name();
        $value = $p->value();
        $descr = array_key_exists( $name, $sections['FOOTER']['PAR2DESCR'] ) ? $sections['FOOTER']['PAR2DESCR'][$name] : $name;
        if( array_key_exists( $name, $used_names )) continue;
        $con->value( 10, $row, '<i>'.$descr.'</i>' )
            ->value(300, $row, $value, $value_color )
            ->label(500, $row, $name, true, $label_color );
    	$row += 20;
    }
    $row += 15;

    /* Display per-run attributes in a separate section
     */
    foreach( $run->attr_classes() as $class_name ) {
        $title = array_key_exists($class_name, $attribute_sections) ? $attribute_sections[$class_name] : $class_name;
        $con->label_1(0, $row, $title, $num_cols );
        $row += 30;

        foreach( $run->attributes($class_name) as $attr ) {
            $con->value( 10, $row, '<i>'.substr( $attr->description(), 0, 32).'</i>' )
                ->value(300, $row, $attr->name(), $value_color )
                ->label(500, $row, substr( $attr->val(), 0, 32), true, $label_color );
        	$row += 20;
        }
        $row += 15;
    }

    echo $con->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>
