<?php

/**
 * This service will destroy an association (if any) between the specified cable
 * and a connector and return an updated dictionary of all known cable and
 * connector types.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

    $cable_id     = NeoCaptarUtils::get_param_GET('cable_id');
    $connector_id = NeoCaptarUtils::get_param_GET('connector_id');

    $authdb = AuthDB::instance();
    $authdb->begin();

    $neocaptar = NeoCaptar::instance();
    $neocaptar->begin();

    $cable     = $neocaptar->find_dict_cable_by_id( $cable_id );
    $connector = $neocaptar->find_dict_cable_by_id( $connector_id );
    if( !is_null( $cable) && !is_null($connector))
        $cable->unlink( $connector->id());

    $types = NeoCaptarUtils::dict_types2array($neocaptar);

    $authdb->commit();
    $neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'type' => $types ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
