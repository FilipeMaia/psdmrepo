<?php

/**
 * This service will update parameters of the specified cable numbers
 * allocation and return an updated object.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

    $id     = NeoCaptarUtils::get_param_GET('id');
    $first  = NeoCaptarUtils::get_param_GET('first',  false);
    $last   = NeoCaptarUtils::get_param_GET('last',   false);
    $prefix = NeoCaptarUtils::get_param_GET('prefix', false);

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();
        
    $cablenumber = $neocaptar->find_cablenumber_allocation_by_id($id);
    if(is_null($cablenumber)) NeoCaptarUtils::report_error("no such cable number: {$id}");

    if( !is_null($first)) {
        $first = intval($first);
        if( $first == 0 ) NeoCaptarUtils::report_error("illegal value for parameter 'first': must be a non-zero integer");
        if( is_null($last) && ($first >= $cablenumber->last()))
            NeoCaptarUtils::report_error("illegal value for parameters: 'first' must be strictly less than the current value of 'last'");
    }
    if( !is_null($last)) {
        $last = intval($last);
        if( $last == 0 ) NeoCaptarUtils::report_error("illegal value for parameter 'last': must be a non-zero integer");
        if( is_null($first) && ($last <= $cablenumber->first()))
            NeoCaptarUtils::report_error("illegal value for parameters: 'last' must be strictly greater than the current value of 'first'");
    }
    if( !is_null($first) && !is_null($last) && ($first >= $last))
        NeoCaptarUtils::report_error("illegal value for parameters: 'first' must be strictly lesser than 'first'");

    if( !is_null($prefix)) {
        $prefix = strtoupper(trim($prefix));
        if(strlen($prefix) != 2)
            NeoCaptarUtils::report_error("illegal value for parameters: 'prefix' must have exactly 2 symbols");
    }

    $cablenumber_as_array = NeoCaptarUtils::cablenumber2array( $cablenumber->update_self($first, $last, $prefix));

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success(array('cablenumber' => $cablenumber_as_array));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
