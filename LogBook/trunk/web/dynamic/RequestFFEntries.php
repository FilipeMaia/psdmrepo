<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for displaying free-form messages
 * in the specified scope of an experiment.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else {
    die( "no valid experiment identifier" );
}
if( isset( $_GET['scope'] )) {
    $scope = trim( $_GET['scope'] );
    if( $scope == '' ) {
        die( "scope can't be empty" );
    } else if( $scope == 'shift' ) {
        if( isset( $_GET['shift_id'] )) {
            $shift_id = trim( $_GET['shift_id'] );
            if( $shift_id == '' ) {
                die( "shift id can't be empty" );
            }
        } else {
            die( "no valid shift id" );
        }
    } else if( $scope == 'run' ) {
        if( isset( $_GET['run_id'] )) {
            $run_id = trim( $_GET['run_id'] );
            if( $run_id == '' )
                die( "run id can't be empty" );
        } else {
            die( "no valid run id" );
        }
    }
} else {
    die( "no valid scope" );
}

function entry2json( $e ) {

    $posted_url =
        $e->insert_time()->toStringShort()." <a href=\"javascript:select_entry(".$e->id().")\">".
        '&gt;&gt;</a> ';
//        '<img src="images/expand.gif" width="10px;" height="15px;" style="color:blue; padding-left:10px;"></a>';
/*
    $posted_url =
        "<a href=\"javascript:select_entry(".$e->id().")\">".
        $e->insert_time()->toStringShort().
        '</a>';
*/
    $tags_url = '';
    $tags = $e->tags();
    foreach( $tags as $t )
        $tags_url .= '<b><em title="'.$t->tag().'">T</em></b>&nbsp;';

    $attachments_url = '';
    $attachments = $e->attachments();
    foreach( $attachments as $a ) {
        $attachments_url .= '<img src="images/attachment.png" title="'.$a->description().'"/>&nbsp;';
    }
    return json_encode(
        array (
            "posted" => $posted_url,
            "author" => $e->author(),
            "relevance_time" => $e->relevance_time()->toStringShort(),
            "message" => substr( $e->content(), 0, 36 ),
            "tags" => $tags_url,
            "attachments" => $attachments_url
        )
    );
}

/*
 * Return JSON objects with a list of experiments.
 */
try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id )
        or die( "no such experiment" );

    if( $scope == 'all' ) {
        $entries = $experiment->entries();
    } else if( $scope == 'experiment' ) {
        $entries = $experiment->entries_of_experiment();
    } else if( $scope == 'shift' ) {
        $entries = $experiment->entries_of_shift( $shift_id );
    } else if( $scope == 'run' ) {
        $entries = $experiment->entries_of_run( $run_id );
    } else {
        die ( 'unsupported scope' );
    }

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $entries as $e ) {
      if( $first ) {
          $first = false;
          echo "\n".entry2json( $e );
      } else {
          echo ",\n".entry2json( $e );
      }
    }
    print <<< HERE
 ] } }
HERE;

    $logbook->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
}
?>
