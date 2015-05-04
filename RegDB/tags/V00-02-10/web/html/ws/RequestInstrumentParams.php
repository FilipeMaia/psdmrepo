<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

/*
 * This script will process a request for displaying parameters of an instrument.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "instrument identifier can't be empty" );
} else
    die( "no valid instrument identifier" );

function param2json( $param ) {
    return json_encode(
        array (
            "name"  => $param->name(),
            "value" => $param->value(),
            "description" => $param->description()
        )
    );
}

/*
 * Return JSON objects with a list of parameters.
 */
try {
    RegDB::instance()->begin();
    $instrument = RegDB::instance()->find_instrument_by_id( $id ) or die( "no such instrument" );
    $params = $instrument->params();

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $params as $p ) {
      if( $first ) {
          $first = false;
          echo "\n".param2json( $p );
      } else {
          echo ",\n".param2json( $p );
      }
    }
    print <<< HERE
 ] } }
HERE;

    RegDB::instance()->commit();

} catch( RegDBException $e ) { print $e->toHtml(); }

?>
