<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

function experiment2json( $experiment ) {

    $instrument_url =
        "<a href=\"javascript:view_instrument(".$experiment->instrument()->id().",'".$experiment->instrument()->name()."')\">".
        $experiment->instrument()->name().
        '</a>';
    $experiment_url =
        "<a href=\"javascript:view_experiment(".$experiment->id().",'".$experiment->name()."')\">".
        $experiment->name().
        '</a>';
    return json_encode(
        array (
            "instrument"  => $instrument_url,
            "experiment"  => $experiment_url,
            "begin_time"  => $experiment->begin_time()->toStringShort(),
            "end_time"    => $experiment->end_time()->toStringShort(),
            "description" => substr( $experiment->description(), 0, 72 )."..."
        )
    );
}

/*
 * Return JSON objects with a list of experiments.
 */
try {
    RegDB::instance()->begin();
    $experiments = RegDB::instance()->experiments();

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $experiments as $e ) {
      if( $first ) {
          $first = false;
          echo "\n".experiment2json( $e );
      } else {
          echo ",\n".experiment2json( $e );
      }
    }
    print <<< HERE
 ] } }
HERE;

    RegDB::instance()->commit();

} catch( RegDBException $e ) { print $e->toHtml(); }

?>
