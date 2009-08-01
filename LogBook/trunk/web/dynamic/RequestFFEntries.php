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

/* Translate an entry into a JASON object. Return the serialized object.
 */
function entry2json( $entry, $format ) {

    $relevance_time_str = is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort();
    $tags = $entry->tags();
    $attachments = $entry->attachments();

    // Produce different output depending on the requested format.
    //
    if( $format == 'detailed' ) {

        $shift_begin_time_str = is_null( $entry->shift_id()) ?
            'n/a' : "<a href=\"javascript:select_shift(".$entry->shift()->id().")\" class=\"lb_link\">".
            $entry->shift()->begin_time()->toStringShort().'</a>';
        $run_number_str = 'n/a';
        if( !is_null( $entry->run_id())) {
            $run = $entry->run();
            $run_number_str = "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$run->num()}</a>";
        }

        // Estimate a number of lines for the message text by counting
        // new lines.
        //
        $message_lines = count( explode( "\n", $entry->content()));
        $message_height = min( 200, 14 + 14*$message_lines );
        $base = 5 + $message_height;

        $extra_lines = max( count( $tags ), count( $attachments ));
        $extra_vspace = $extra_lines == 0 ? 0 :  35 + 20 * $extra_lines;

        $con = new RegDBHtml( 0, 0, 800, 10 + $message_height + $extra_vspace );

        $highlight = true;
        $con->container_1 (   0,   0, "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">{$entry->content()}</pre>", 800, $message_height, $highlight );

        if( $extra_lines != 0 ) {
            $style = 'border: solid 2px #efefef;';
            $highlight = true;
            $con_1 = new RegDBHtml( 0, 0, 240, $extra_vspace, 'relative', $style, $highlight );
            if( count( $tags ) != 0 ) {
                $con_1->label(  10, 5, 'Tag', 80 );
                $base4tags = 25;
                foreach( $tags as $tag ) {
                    $value = $tag->value();
                    $value_str = $value == '' ? '' : ' = <i>'.$value.'</i>';
                    $con_1->value_1(  10, $base4tags, $tag->tag().$value_str);
                    $base4tags = $base4tags + 20;
                }
            }
            $con->container_1( 0, $base, $con_1->html());
            $con_1 = new RegDBHtml( 0, 0, 545, $extra_vspace, 'relative', $style, $highlight );
            if( count( $attachments ) != 0 ) {
                $con_1->label( 10, 5, 'Attachment' )->label( 215, 5, 'Size' )->label( 275, 5, 'Type' );
                $base4attch = 25;
                foreach( $attachments as $attachment ) {
                    $attachment_url = '<a href="ShowAttachment.php?id='.$attachment->id().'" target="_blank" class="lb_link">'.$attachment->description().'</a>';
                    $con_1->value_1(  10, $base4attch, $attachment_url )
                          ->value_1( 215, $base4attch, $attachment->document_size())
                          ->value_1( 275, $base4attch, $attachment->document_type());
                    $base4attch = $base4attch + 20;
                }
            }
            $con->container_1( 250, $base, $con_1->html());
        }
        return json_encode(
            array (
                "event_time" => $entry->insert_time()->toStringShort(),
                "relevance_time" => $relevance_time_str,
                "run" => $run_number_str,
                "shift" => $shift_begin_time_str,
                "author" => $entry->author(),
                "html" => $con->html()
            )
        );

    } else if( $format == 'compact' ) {
        $posted_url =
            " <a href=\"javascript:select_entry({$entry->id()})\">".$entry->insert_time()->toStringShort().'</a> ';

        $shift_begin_time_str = is_null( $entry->shift_id()) ?
            '' : "<a href=\"javascript:select_shift(".$entry->shift()->id().")\" class=\"lb_link\">".
            $entry->shift()->begin_time()->toStringShort().'</a>';
        $run_number_str = '';
        if( !is_null( $entry->run_id())) {
            $run = $entry->run();
            $run_number_str = "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$run->num()}</a>";
        }
        $tags_str = '';
        foreach( $tags as $t ) {
            if( $tags_str == '') $tags_str = $t->tag();
            else                 $tags_str .= "<br>".$t->tag();
        }
        $attachments_str = '';
        foreach( $attachments as $a ) {
            $title = $a->description().', '.$a->document_size().' bytes, document type: '.$a->document_type();
            $attachment_url =
                '<a href="ShowAttachment.php?id='.$a->id().'" target="_blank"'.
                ' title="'.$title.'" class=\"lb_link\">'.substr( $a->description(), 0, 16 ).(strlen( $a->description()) > 16 ? '..' : '').'..</a>';
            if( $attachments_str == '') $attachments_str = $attachment_url;
            else                        $attachments_str .= "<br>".$attachment_url;
        }
        return json_encode(
            array (
                "posted" => $posted_url,
                "author" => substr( $entry->author(), 0, 10 ).(strlen( $entry->author()) > 10 ? '..' : ''),
                "run" => $run_number_str,
                "shift" => $shift_begin_time_str,
                "message" => substr( $entry->content(), 0, 36 ).(strlen( $entry->content()) > 36 ? '..' : ''),
                "tags" => $tags_str,
                "attachments" => $attachments_str
            )
        );
    }
    return null;
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
          echo "\n".entry2json( $e, 'compact' );
      } else {
          echo ",\n".entry2json( $e,  'compact' );
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
