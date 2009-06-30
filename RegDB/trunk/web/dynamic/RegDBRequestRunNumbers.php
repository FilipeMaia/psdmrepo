<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for displaying run numbers in one of
 * the following scopes depending on input parameters of the script:
 *
 * - if no experiment identifier parameter is passed then a summary run
 *   information for all experiments will be returned
 *
 * - all runs of the requested experiment will be returned otherwise
 */
$exper_id = null;
if( isset( $_GET['exper_id'] )) {
    $exper_id = trim( $_GET['exper_id'] );
    if( $exper_id == '' )
        die( "experiment identifier can't be empty" );
}

function experiment_runs2json( $instrument, $experiment ) {
    $experiment_url =<<<HERE
<a href="javascript:view_run_numbers({$experiment->id()},'{$experiment->name()}')">{$experiment->name()}</a>
HERE;
    $run = $experiment->last_run();
    return json_encode(
        array (
            "instrument"   => $instrument->name(),
            "experiment"   => $experiment_url,
            "last_run_num" => is_null( $run ) ? '' : $run->num(),
            "request_time" => is_null( $run ) ? '' : $run->request_time()->toStringShort()
        )
    );
}

function run2json( $run ) {
HERE;
    return json_encode(
        array (
            "run" =>          $run->num(),
            "request_time" => $run->request_time()->toStringShort()
        )
    );
}
/*
 * Return JSON objects with a list of groups.
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;

    if( is_null( $exper_id )) {
        $instruments = $regdb->instruments();
        foreach( $instruments as $i ) {
            $experiments = $regdb->experiments_for_instrument( $i->name());
            foreach( $experiments as $e ) {
                if( $first ) {
                    $first = false;
                    echo "\n".experiment_runs2json( $i, $e );
                } else {
                    echo ",\n".experiment_runs2json( $i, $e );
                }
            }
        }
    } else {
        $experiment = $regdb->find_experiment_by_id( $exper_id )
            or die( "no such experiment" );
        $runs = $experiment->runs();
        foreach( $runs as $r ) {
            if( $first ) {
                $first = false;
                echo "\n".run2json( $r );
            } else {
                echo ",\n".run2json( $r );
            }
        }
    }
    print <<< HERE
 ] } }
HERE;

    $regdb->commit();

} catch( regdbException $e ) {
    print $e->toHtml();
}

?>
