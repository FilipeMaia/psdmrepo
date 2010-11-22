<?php

require_once( 'LogBook/LogBook.inc.php' );
require_once( 'RegDB/RegDB.inc.php' );

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
    )
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

    $num_cols = 780;

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
    
    echo $con->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>
