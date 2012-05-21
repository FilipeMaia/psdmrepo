<?php

/**
 * This service will update the specified pinlist and return an updated dictionary
 * of the corresponding cable types, connectors and pinlists.
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

	// Check for proper authorization andf get the current UID
	//
	$authdb = AuthDB::instance();
	$authdb->begin();
	$created_uid = $authdb->authName();
	$authdb->commit();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $pinlist = $neocaptar->find_dict_pinlist_by_id($id);
    if( is_null( $pinlist )) NeoCaptarUtils::report_error('failed to find the specified pinlist');

    $pinlist = $pinlist->connector()->update_pinlist($id,$documentation);
    $cable   = $pinlist->connector()->cable();

    $connectors = array();
	foreach( $cable->connectors() as $connector ) {

		$pinlists = array();
		foreach( $connector->pinlists() as $pinlist ) {
			$pinlists[$pinlist->name()] = array(
				'id'            => $pinlist->id(),
                'documentation' => $pinlist->documentation(),
				'created_time'  => $pinlist->created_time()->toStringShort(),
				'created_uid'   => $pinlist->created_uid()
			);
		}
		$connectors[$connector->name()] = array(
			'id'           => $connector->id(),
			'created_time' => $connector->created_time()->toStringShort(),
			'created_uid'  => $connector->created_uid(),
			'pinlist'      => $pinlists
		);
	}
	$cables = array(
		$cable->name() => array(
			'id'           => $cable->id(),
			'created_time' => $cable->created_time()->toStringShort(),
			'created_uid'  => $cable->created_uid(),
			'connector'    => $connectors
		)
	);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'cable' => $cables ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
