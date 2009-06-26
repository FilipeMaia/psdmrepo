<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for creating a new instrument
 * in the database.
 */
print_r( $_POST['params'] );
$params = $_POST['params'];
for( $i = 0; $i < $params; $i++ ) {
    $param = array(
        'name' => $_POST['param_name_'.$i],
        'value' => $_POST['param_value_'.$i],
        'description' => $_POST['param_description_'.$i] );
    echo "<br>";
    print_r( $param );
}

return;

if( isset( $_POST['instrument_name'] )) {
    $instrument_name = trim( $_POST['instrument_name'] );
    if( $instrument_name == '' )
        die( "instrument name can't be empty" );
} else
    die( "no valid instrument name" );

if( isset( $_POST['description'] )) {
    $description = trim( $_POST['description'] );
    if( $description == '' )
        die( "instrument description field can't be empty" );
} else
    die( "no valid instrument description" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $instrument = $regdb->register_instrument (
        $instrument_name, $description );

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'home' )
            header( 'Location: RegDB_v1.php' );
        else if( $actionSuccess == 'list_instruments' )
            header( 'Location: RegDB_v1.php?action=list_instruments' );
        else if( $actionSuccess == 'view_instrument' )
            header( 'Location: RegDB_v1.php?action=view_instrument&id='.$instrument->id().'&name='.$instrument->name());
        else if( $actionSuccess == 'edit_instrument' )
            header( 'Location: RegDB_v1.php?action=edit_instrument&id='.$instrument->id().'&name='.$instrument->name());
        else
            ;
    }
    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>