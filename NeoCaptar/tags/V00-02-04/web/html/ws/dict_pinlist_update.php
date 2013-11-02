<?php

/**
 * This service will update the specified pinlist and return an updated dictionary
 * of the known pinlists.
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
    // Mandatory parameter
    //
    $id = NeoCaptarUtils::get_param_GET('id');

    // Optional parameters
    //
    $required    = false;
    $allow_empty = true;

    $documentation              = NeoCaptarUtils::get_param_GET('documentation',         $required, $allow_empty);
    $cable_type_name            = NeoCaptarUtils::get_param_GET('cable',                 $required, $allow_empty);
    $origin_connector_name      = NeoCaptarUtils::get_param_GET('origin_connector',      $required, $allow_empty);
    $destination_connector_name = NeoCaptarUtils::get_param_GET('destination_connector', $required, $allow_empty);

	// Check for proper authorization and get the current UID
	//
	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $pinlist = $neocaptar->find_dict_pinlist_by_id($id);
    if( !is_null( $pinlist )) {
        if( !is_null($documentation)) {
            $documentation = trim($documentation);
            $pinlist->update_documentation($documentation);
        }
        if( !is_null($cable_type_name)) {
            $cable_type_name = trim($cable_type_name);
            if( $cable_type_name == '' ) {
                $pinlist->update_cable();
            } else {
                $cable = $neocaptar->find_dict_cable_by_name($cable_type_name);
                if( is_null($cable)) NeoCaptarUtils::report_error("no such cable type: '{$cable_type_name}'");
                $pinlist->update_cable($cable->id());
            }
        }
        if( !is_null($origin_connector_name)) {
            $origin_connector_name = trim($origin_connector_name);
            if( $origin_connector_name == '' ) {
                $pinlist->update_origin_connector();
            } else {
                $connector = $neocaptar->find_dict_connector_by_name($origin_connector_name);
                if( is_null($connector)) NeoCaptarUtils::report_error("no such connector: '{$origin_connector_name}'");
                $pinlist->update_origin_connector($connector->id());
            }
        }
        if( !is_null($destination_connector_name)) {
            $destination_connector_name = trim($destination_connector_name);
            if( $destination_connector_name == '' ) {
                $pinlist->update_destination_connector();
            } else {
                $connector = $neocaptar->find_dict_connector_by_name($destination_connector_name);
                if( is_null($connector)) NeoCaptarUtils::report_error("no such connector: '{$destination_connector_name}'");
                $pinlist->update_destination_connector($connector->id());
            }
        }
    }
    $pinlists = NeoCaptarUtils::dict_pinlists2array($neocaptar);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'pinlist' => $pinlists ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
