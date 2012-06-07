<?php

/**
 * This service will update connector documentation in the dictionary and
 * return an updated dictionary of all known cables and connectors.
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

    $id            = NeoCaptarUtils::get_param_GET('id');
    $documentation = NeoCaptarUtils::get_param_GET('documentation',true,true);

    $authdb = AuthDB::instance();
    $authdb->begin();

    $neocaptar = NeoCaptar::instance();
    $neocaptar->begin();

    $connector = $neocaptar->find_dict_connector_by_id( $id );
    if( !is_null($connector))
        $connector->update( $documentation );

    $types = NeoCaptarUtils::dict_types2array($neocaptar);

    $authdb->commit();
    $neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'type' => $types ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
