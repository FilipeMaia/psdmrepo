<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

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
    $comments = NeoCaptarUtils::get_param_POST('comments', false, true);  // not required and allowed to be empty
    $comments = is_null($comments) ? '' : trim($comments);

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
            "description",
            "device",
            "device_location",
            "device_region",
            "device_component",
            "device_counter",
            "device_suffix",
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
        $cable = $cable->update_self($params, $comments);

    } else {

        //////////////////////////////////////////////
        // Making a status transition for the cable //
        //////////////////////////////////////////////

        function assert_status($cable,$status) {
            NeoCaptarUtils::report_error("can't change cable status from '{$cable->status()}' to '{$status}' for cable id: {$cable->id()}");
        }
        switch( $status ) {
            case 'Planned':
                if     ($cable->status() == 'Registered'  ) $cable = $neocaptar->un_register_cable  ($cable, $comments);
                else                                        assert_status($cable,$status);
                break;
            case 'Registered':
                if     ($cable->status() == 'Planned'     ) $cable = $neocaptar->register_cable     ($cable, $comments);
                else if($cable->status() == 'Labeled'     ) $cable = $neocaptar->un_label_cable     ($cable, $comments);
                else                                        assert_status($cable,$status);
                break;
            case 'Labeled':
                if     ($cable->status() == 'Registered'  ) $cable = $neocaptar->label_cable        ($cable, $comments);
                else if($cable->status() == 'Fabrication' ) $cable = $neocaptar->un_fabricate_cable ($cable, $comments);
                else                                        assert_status($cable,$status);
                break;
            case 'Fabrication':
                if     ($cable->status() == 'Labeled'     ) $cable = $neocaptar->fabricate_cable    ($cable, $comments);
                else if($cable->status() == 'Ready'       ) $cable = $neocaptar->un_ready_cable     ($cable, $comments);
                else                                        assert_status($cable,$status);
                break;
            case 'Ready':
                if     ($cable->status() == 'Fabrication' ) $cable = $neocaptar->ready_cable        ($cable, $comments);
                else if($cable->status() == 'Installed'   ) $cable = $neocaptar->un_install_cable   ($cable, $comments);
                else                                        assert_status($cable,$status);
                break;
            case 'Installed':
                if     ($cable->status() == 'Ready'       ) $cable = $neocaptar->install_cable      ($cable, $comments);
                else if($cable->status() == 'Commissioned') $cable = $neocaptar->un_commission_cable($cable, $comments);
                else                                        assert_status($cable,$status);
                break;
            case 'Commissioned':
                if     ($cable->status() == 'Installed'   ) $cable = $neocaptar->commission_cable   ($cable, $comments);
                else if($cable->status() == 'Damaged'     ) $cable = $neocaptar->un_damage_cable    ($cable, $comments);
                else                                        assert_status($cable,$status);
                break;
            case 'Damaged':
                if     ($cable->status() == 'Commissioned') $cable = $neocaptar->damage_cable       ($cable, $comments);
                else if($cable->status() == 'Retired'     ) $cable = $neocaptar->un_retire_cable    ($cable, $comments);
                else                                        assert_status($cable,$status);
                break;
            case 'Retired':
                if     ($cable->status() == 'Damaged'     ) $cable = $neocaptar->retire_cable       ($cable, $comments);
                else                                        assert_status($cable,$status);
                break;
            default:
                assert_status($cable,$status);
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
