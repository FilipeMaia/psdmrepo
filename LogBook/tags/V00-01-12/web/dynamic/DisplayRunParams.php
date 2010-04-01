<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for displaying parameters of a run.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "run identifier can't be empty" );
} else
    die( "no valid run identifier" );

define( BEAMS_TITLE,   "&nbsp;E l e c t r o n &nbsp;&nbsp; a n d &nbsp;&nbsp; P h o t o n &nbsp;&nbsp; b e a m s");
define( FEE_TITLE,     "&nbsp;F E E" );
define( HFP_TITLE,     "&nbsp;H F P" );
define( ETOF_TITLE,    "&nbsp;e T O F" );
define( ITOF_TITLE,    "&nbsp;i T O F" );
define( HFP_GAS_TITLE, "&nbsp;H F P &nbsp;&nbsp; G a s" );
define( DIA_TITLE,     "&nbsp;D I A " );
define( MBES_TITLE,    "&nbsp;M B E S" );
define( EXTRA_TITLE,    "&nbsp;E x p e r i m e n t &nbsp;&nbsp; S p e c i f i c" );

define( BEAMS,   0 );
define( FEE,     BEAMS   + 10 );
define( HFP,     FEE     +  8 );
define( ETOF,    HFP     +  2 );
define( ITOF,    ETOF    + 30 );
define( HFP_GAS, ITOF    +  6 );
define( DIA,     HFP_GAS +  7 );
define( MBES,    DIA     +  1 );
define( EXTRA,   MBES    +  4 );
define( END_,    EXTRA   +  0 );

$pardefs = array(
    //
    // Electron beam and Photon beam data
    //
    array( 'name' => 'BEND:DMP1:400:BDES',   'descr' => 'electron beam energy' ),
    array( 'name' => 'BPMS:DMP1:199:TMIT1H', 'descr' => 'Particle N_electrons' ),
    array( 'name' => 'SIOC:SYS0:ML00:AO289', 'descr' => 'E.Vernier' ),
    array( 'name' => 'BEAM:LCLS:ELEC:Q',     'descr' => 'Charge' ),
    array( 'name' => 'SIOC:SYS0:ML00:AO195', 'descr' => 'Peak current after second bunch compressor' ),
    array( 'name' => 'SIOC:SYS0:ML00:AO820', 'descr' => 'Pulse length' ),
    array( 'name' => 'SIOC:SYS0:ML00:AO569', 'descr' => 'ebeam energy loss converted to photon mJ' ),
    array( 'name' => 'SIOC:SYS0:ML00:AO580', 'descr' => 'Calculated number of photons' ),
    array( 'name' => 'SIOC:SYS0:ML00:AO541', 'descr' => 'Photon beam energy' ),
    array( 'name' => 'SIOC:SYS0:ML00:AO192', 'descr' => 'Wavelength' ),
    //
    // FEE data
    //
    array( 'name' => 'VGPR:FEE1:311:PSETPOINT_DES', 'descr' => 'Gas attenuator setpoint' ),
    array( 'name' => 'VGCP:FEE1:311:P',             'descr' => 'Gas attenuator actual pressure' ),
    array( 'name' => 'GATT:FEE1:310:R_ACT',         'descr' => 'Gas attenuator calculated transmission' ),
    array( 'name' => 'SATT:FEE1:321:STATE',         'descr' => 'Solid attenuator 1' ),
    array( 'name' => 'SATT:FEE1:322:STATE',         'descr' => 'Solid attenuator 2' ),
    array( 'name' => 'SATT:FEE1:323:STATE',         'descr' => 'Solid attenuator 3' ),
    array( 'name' => 'SATT:FEE1:324:STATE',         'descr' => 'Solid attenuator 4' ),
    array( 'name' => 'SATT:FEE1:320:TACT',          'descr' => 'Total attenuator length' ),
    //
    // HFP data
    //
    array( 'name' => 'AMO:HFP:GCC:01:PMON', 'descr' => 'pressure' ),
    array( 'name' => 'AMO:HFP:MMS:table.Z', 'descr' => 'z-position' ),
    //
    // etof settings
    //
    //   etof 1
    //
    array( 'name' => 'AMO:R14:IOC:10:ao0:out1',                'descr' => '1' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS0:CH0:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS0:CH1:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS0:CH2:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS0:CH3:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS7:CH0:VoltageMeasure', 'descr' => '' ),
    //
    //   etof 2
    //
    array( 'name' => 'AMO:R14:IOC:10:ao0:out2',                'descr' => '2' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS1:CH0:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS1:CH1:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS1:CH2:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS1:CH3:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS7:CH1:VoltageMeasure', 'descr' => '' ),
    //
    //   etof 3
    //
    array( 'name' => 'AMO:R14:IOC:10:ao0:out3',                'descr' => '3' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS2:CH0:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS2:CH1:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS2:CH2:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS2:CH3:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS7:CH2:VoltageMeasure', 'descr' => '' ),
    //
    //   etof 4
    //
    array( 'name' => 'AMO:R14:IOC:10:ao0:out4',                'descr' => '4' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS3:CH0:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS3:CH1:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS3:CH2:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS3:CH3:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS7:CH3:VoltageMeasure', 'descr' => '' ),
    //
    //   etof 5
    //
    array( 'name' => 'AMO:R14:IOC:10:ao0:out5',                'descr' => '5' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS4:CH0:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS4:CH1:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS4:CH2:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS4:CH3:VoltageMeasure', 'descr' => '' ),
    array( 'name' => 'AMO:R14:IOC:10:VHS8:CH0:VoltageMeasure', 'descr' => '' ),
    //
    // itof settings
    //
    array( 'name' => 'AMO:R14:IOC:21:VHS2:CH0:VoltageMeasure', 'descr' => 'repeller' ),
    array( 'name' => 'AMO:R14:IOC:21:VHS0:CH0:VoltageMeasure', 'descr' => 'extractor' ),
    array( 'name' => 'AMO:R14:IOC:21:VHS0:CH1:VoltageMeasure', 'descr' => 'acceleration' ),
    array( 'name' => 'AMO:R14:IOC:21:VHS0:CH2:VoltageMeasure', 'descr' => 'MCP in' ),
    array( 'name' => 'AMO:R14:IOC:21:VHS2:CH2:VoltageMeasure', 'descr' => 'MCP out' ),
    array( 'name' => 'AMO:R14:IOC:21:VHS2:CH1:VoltageMeasure', 'descr' => 'Anode' ),
    //
    // HFP Gas data
    //
    array( 'name' => 'AMO:HFP:GCC:03:PMON',                    'descr' => 'pressure' ),
    array( 'name' => 'AMO:R14:IOC:21:VHS7:CH0:VoltageMeasure', 'descr' => 'piezo voltage' ),
    array( 'name' => 'AMO:R14:EVR:21:CTRL.DG2D',               'descr' => 'piezo timing delay' ),
    array( 'name' => 'AMO:R14:EVR:21:CTRL.DG2W',               'descr' => 'piezo timing width' ),
    array( 'name' => 'AMO:HFP:MMS:72.RBV',                     'descr' => 'gasjet x-position (rel. distance)' ),
    array( 'name' => 'AMO:HFP:MMS:71.RBV',                     'descr' => 'gasjet y-position (rel. distance)' ),
    array( 'name' => 'AMO:HFP:MMS:73.RBV',                     'descr' => 'Gas Jet motor Z axis (mm)' ),
    //
    // DIA data
    //
    array( 'name' => 'AMO:DIA:GCC:01:PMON', 'descr' => 'pressure' ),
    //
    // mbes settings
    //
    //   coils
    //
    array( 'name' => 'AMO:DIA:SHC:11:I', 'descr' => 'coil 1' ),
    array( 'name' => 'AMO:DIA:SHC:12:I', 'descr' => 'coil 2' ),
    //
    array( 'name' => 'AMO:R15:IOC:40:VHS0:CH0:VoltageSet', 'descr' => '' ),
    array( 'name' => 'AMO:R15:IOC:40:VHS0:CH1:VoltageSet', 'descr' => '' )
 );

$par2descr = array();
foreach( $pardefs as $p ) $par2descr[$p['name']] = $p['descr'];

// Add extra parameters
//
$par2descr['AMO:DIA:GMP:06:PMON'] = 'Pressure (Torr)';

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
    $num_rows = count( $values ) * 20 + 9 * ( 30 + 15 );
    $con = new RegDBHtml( 0, 0, 780, $num_rows );
    $row = 0;

    $con->label_1(0, $row, BEAMS_TITLE, 780 );
    $row += 30;
    for( $i = BEAMS; $i < FEE; $i++ ) {
    	$key = $pardefs[$i]['name'];
    	$val = array_key_exists( $key, $values ) ? $values[$key]->value() : '&lt; no data &gt;';
        $con->value( 10, $row, '<i>'.$pardefs[$i]['descr'].'</i>' )
            ->value(300, $row, $val, $value_color )
            ->label(500, $row, $key, true, $label_color );
        $row += 20;
    }
    $row += 15;

    $con->label_1(0, $row, FEE_TITLE, 780 );
    $row += 30;
    for( $i = FEE; $i < HFP; $i++ ) {
    	$key = $pardefs[$i]['name'];
    	$val = array_key_exists( $key, $values ) ? $values[$key]->value() : '&lt; no data &gt;';
        $con->value( 10, $row, '<i>'.$pardefs[$i]['descr'].'</i>' )
            ->value(300, $row, $val, $value_color )
            ->label(500, $row, $key, true, $label_color );
    	$row += 20;
    }
    $row += 15;

    $con->label_1(0, $row, HFP_TITLE, 780 );
    $row += 30;
    for( $i = HFP; $i < ETOF; $i++ ) {
    	$key = $pardefs[$i]['name'];
    	$val = array_key_exists( $key, $values ) ? $values[$key]->value() : '&lt; no data &gt;';
        $con->value( 10, $row, '<i>'.$pardefs[$i]['descr'].'</i>' )
            ->value(300, $row, $val, $value_color )
            ->label(500, $row, $key, true, $label_color );
    	$row += 20;
    }
    $row += 15;

    $con->label_1(0, $row, ETOF_TITLE, 780 );
    $row += 30;
    for( $i = ETOF; $i < ITOF; $i++ ) {
    	$key = $pardefs[$i]['name'];
    	$val = array_key_exists( $key, $values ) ? $values[$key]->value() : '&lt; no data &gt;';
        $con->value( 10, $row, '<i>'.$pardefs[$i]['descr'].'</i>' )
            ->value(300, $row, $val, $value_color )
            ->label(500, $row, $key, true, $label_color );
    	$row += 20;
    }
    $row += 15;

    $con->label_1(0, $row, ITOF_TITLE, 780 );
    $row += 30;
    for( $i = ITOF; $i < HFP_GAS; $i++ ) {
    	$key = $pardefs[$i]['name'];
    	$val = array_key_exists( $key, $values ) ? $values[$key]->value() : '&lt; no data &gt;';
        $con->value( 10, $row, '<i>'.$pardefs[$i]['descr'].'</i>' )
            ->value(300, $row, $val, $value_color )
            ->label(500, $row, $key, true, $label_color );
    	$row += 20;
    }
    $row += 15;
    
    $con->label_1(0, $row, HFP_GAS_TITLE, 780 );
    $row += 30;
    for( $i = HFP_GAS; $i < DIA; $i++ ) {
    	$key = $pardefs[$i]['name'];
    	$val = array_key_exists( $key, $values ) ? $values[$key]->value() : '&lt; no data &gt;';
        $con->value( 10, $row, '<i>'.$pardefs[$i]['descr'].'</i>' )
            ->value(300, $row, $val, $value_color )
            ->label(500, $row, $key, true, $label_color );
    	$row += 20;
    }
    $row += 15;

    $con->label_1(0, $row, DIA_TITLE, 780 );
    $row += 30;
    for( $i = DIA; $i < MBES; $i++ ) {
    	$key = $pardefs[$i]['name'];
    	$val = array_key_exists( $key, $values ) ? $values[$key]->value() : '&lt; no data &gt;';
        $con->value( 10, $row, '<i>'.$pardefs[$i]['descr'].'</i>' )
            ->value(300, $row, $val, $value_color )
            ->label(500, $row, $key, true, $label_color );
    	$row += 20;
    }
    $row += 15;

    $con->label_1(0, $row, MBES_TITLE, 780 );
    $row += 30;
    for( $i = MBES; $i < EXTRA; $i++ ) {
    	$key = $pardefs[$i]['name'];
    	$val = array_key_exists( $key, $values ) ? $values[$key]->value() : '&lt; no data &gt;';
        $con->value( 10, $row, '<i>'.$pardefs[$i]['descr'].'</i>' )
            ->value(300, $row, $val, $value_color )
            ->label(500, $row, $key, true, $label_color );
    	$row += 20;
    }
    $row += 15;

    $con->label_1(0, $row, EXTRA_TITLE, 780 );
    $row += 30;
    $pardefs_keys = array();
    foreach( $pardefs as $p ) $pardefs_keys[$p['name']] = $p;
    foreach( $values as $p ) {
        $key = $p->name();
        $val = $p->value();
        $descr = array_key_exists( $key, $par2descr ) ? $par2descr[$key] : $key;
        if( array_key_exists( $key, $pardefs_keys )) continue;
        $con->value( 10, $row, '<i>'.$descr.'</i>' )
            ->value(300, $row, $val, $value_color )
            ->label(500, $row, $key, true, $label_color );
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