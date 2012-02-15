<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

use LusiTime\LusiTimeException;

/**
 * This service will update cable parameters in the database. The cable should
 * already exist.
 *
 * The script has two modes of operation, depending if 'status' is present
 * among the list of parameters:
 * 
 *   if present: ignore any other parameters (if any found), verify the current
 *               configuration of a cable, and make the required status transition
 *               if it's allowed.
 *
 *   if not:     verify an object status, and if it permits then apply requested
 *               modifications.
 */
header( "Content-type: application/json" );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

	$authdb = AuthDB::instance();
	$authdb->begin();

    $neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    // Required parameters, no empty values allowed
    //
    $cable_id = NeoCaptarUtils::get_param_POST('cable_id');

    $cable = $neocaptar->find_cable_by_id($cable_id);
    if( is_null($cable)) NeoCaptarUtils::report_error('no cable exists for id: '.$cable_id);

    // Determine a mode of operation
    //
    $status = NeoCaptarUtils::get_param_POST('status',false,false);

    if( is_null($status)) {

        ////////////////////////////////////////////
        // Just changing attributes of the cable  //
        ////////////////////////////////////////////

        // Optional parameters, empty values are allowed
        //
        $param_names = array(
            "device",
            "func",
            "cable_type",
            "length",
            "routing",
            "origin_name",
            "origin_loc",
            "origin_rack",
            "origin_ele",
            "origin_side",
            "origin_slot",
            "origin_conn",
            "origin_pinlist",
            "origin_station",
            "origin_conntype",
            "origin_instr",
            "destination_name",
            "destination_loc",
            "destination_rack",
            "destination_ele",
            "destination_side",
            "destination_slot",
            "destination_conn",
            "destination_pinlist",
            "destination_station",
            "destination_conntype",
            "destination_instr" );

        $params = array();
        foreach( $param_names as $i => $name ) {
            $val = NeoCaptarUtils::get_param_POST($name,false,true);
            if( !is_null($val)) $params[$name] = $val;
        }
        $cable = $cable->update_self($params);

    } else {

        //////////////////////////////////////////////
        // Making a status transition for the cable //
        //////////////////////////////////////////////

        function assure_status($condition,$cable,$status) {
            if(!$condition)
                NeoCaptarUtils::report_error("can't change cable status from '{$cable->status()}' to '{$status}' for cable id: {$cable->id()}");
        }
        switch( $status ) {
            case 'Registered':
                assure_status($cable->status() == 'Planned',$cable,$status);
                $cable = $neocaptar->register_cable($cable);
                break;
            case 'Labeled':
                assure_status($cable->status() == 'Registered',$cable,$status);
                $cable = $neocaptar->label_cable($cable);
                break;
            case 'Fabrication':
                assure_status($cable->status() == 'Labeled',$cable,$status);
                $cable = $neocaptar->fabricate_cable($cable);
                break;
            case 'Ready':
                assure_status($cable->status() == 'Fabrication',$cable,$status);
                $cable = $neocaptar->ready_cable($cable);
                break;
            case 'Installed':
                assure_status($cable->status() == 'Ready',$cable,$status);
                $cable = $neocaptar->install_cable($cable);
                break;
            case 'Commissioned':
                assure_status($cable->status() == 'Installed',$cable,$status);
                $cable = $neocaptar->commission_cable($cable);
                break;
            case 'Damaged':
                assure_status($cable->status() == 'Commissioned',$cable,$status);
                $cable = $neocaptar->damage_cable($cable);
                break;
            case 'Retired':
                assure_status($cable->status() == 'Damaged',$cable,$status);
                $cable = $neocaptar->retire_cable($cable);
                break;
            default:
                assure_status(false,$cable,$status);
                break;
        }
    }
    $cable2return = NeoCaptarUtils::cable2array($cable);

	$neocaptar->commit();
	$authdb->commit();

    NeoCaptarUtils::report_success( array( 'cable' => $cable2return ));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  
?>
