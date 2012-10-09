<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

function instrument2json( $instrument ) {
    $instrument_url =
        "<a href=\"javascript:view_instrument(".$instrument->id().",'".$instrument->name()."')\">".
        $instrument->name().
        '</a>';
    return json_encode(
        array (
            "instrument"  => $instrument_url,
            "description" => substr( $instrument->description(), 0, 72 )."..."
        )
    );
}

/*
 * Return JSON objects with a list of instruments.
 */
try {
    RegDB::instance()->begin();
    $instruments = RegDB::instance()->instruments();

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $instruments as $i ) {
      if( $first ) {
          $first = false;
          echo "\n".instrument2json( $i );
      } else {
          echo ",\n".instrument2json( $i );
      }
    }
    print <<< HERE
 ] } }
HERE;

    RegDB::instance()->commit();

} catch( RegDBException $e ) { print $e->toHtml(); }

?>
