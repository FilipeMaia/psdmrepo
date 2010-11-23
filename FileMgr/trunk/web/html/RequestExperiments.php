<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBException;

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned as JSON object (array).
 */
if( !isset( $_GET['type'] )) die( "no valid information type in the request" );
$type = trim( $_GET['type'] );

define( BROWSE_INSTRUMENTS, 1 );
define( BROWSE_EXPERIMENTS, 2 );
define( BROWSE_FILES,       3 );

$instrument = null;
$experiment = null;
$path = null;
switch( $type ) {
    case BROWSE_FILES:
        if( !isset( $_GET['path'] )) die( "no valid path in the request" );
        $path = trim( $_GET['path'] );
    	if( !isset( $_GET['experiment'] )) die( "no valid experiment name in the request" );
        $experiment = trim( $_GET['experiment'] );
    case BROWSE_EXPERIMENTS:
        if( !isset( $_GET['instrument'] )) die( "no valid instrument name in the request" );
        $instrument = trim( $_GET['instrument'] );
    case BROWSE_INSTRUMENTS:
        break;
}

function instr2json( $instr ) {
    return json_encode(
        array (
            "label"      => '<em style="color:#0071bc;"><b>'.$instr->name().'</b></em>',
            "expanded"   => true,
            "type"       => BROWSE_EXPERIMENTS,
            "title"      => "select an experiment",
            "instrument" => $instr->name(),
            "path"       => "/".$instr->name()
        )
    );
}

function exper2json( $exper ) {
    return json_encode(
        array (
            "label"      => '<em style="color:#0071bc;"><b>'.$exper->name().'</b></em>',
            "expanded"   => false,
            "isLeaf"     => true,
            "type"       => BROWSE_FILES,
            "title"      => "click to see file catalogs",
            "experiment" => $exper->name(),
            "exper_id"   => $exper->id(),
            "instrument" => $exper->instrument()->name(),
            "path"       => "/".$exper->instrument()->name()."/".$exper->name()
        )
    );
}

/*
 * Analyze and process the request
 */
try {

    $regdb = new RegDB();
    $regdb->begin();

    // Proceed to the operation
    //
    header( "Content-type: application/json" );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;

    if( $type == BROWSE_INSTRUMENTS ) {
        $instruments = $regdb->instruments();
        foreach( $instruments as $i ) {
            if( $i->is_location()) continue;
            if( $first ) {
                $first = false;
                echo "\n".instr2json( $i );
            } else {
                echo ",\n".instr2json( $i );
            }
        }
    } else if( $type == BROWSE_EXPERIMENTS ) {
        $instrument = $regdb->find_instrument_by_name( $instrument )
            or die("No such instrument");
        $experiments = $regdb->experiments_for_instrument( $instrument->name());
        foreach( $experiments as $e ) {
          if( $first ) {
              $first = false;
              echo "\n".exper2json( $e );
          } else {
              echo ",\n".exper2json( $e );
          }
        }
    } else if( $type == BROWSE_FILES ) {
    	$url = "https://pswww.slac.stanford.edu/ws-auth/irodsws/files";
    	$opts = array(
    	    "timeout" => 1,
    	    "httpauth " => "gapon:newlife2"
    	);
        $response = http_get($url, $opts, $info );
        print $response."<br>";
        print_r($info);
    }
    print <<< HERE
 ] } }
HERE;

    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>
