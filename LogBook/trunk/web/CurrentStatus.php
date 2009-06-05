<!--
The page for reporting the current status of the LogBook database.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Current Status of the LogBook Database</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <style>
        #experiment_hdr {
            margin-left:4em;
        }
        #experiment {
            margin-left:4em;
        }
        .hdr_table_cell_1st {
            width:4em;
            font-size:x-large;
        }
        .table_cell_1st {
            color:#0071bc;
            width:6em;
        }
        .table_cell_2d {
            width:2em;
        }
        .table_cell_name {
            background-color:silver;
            width:9em;
        }
        .table_cell_number {
            background-color:silver;
            width:4em;
        }
        .table_cell_time {
            width:9em;
            background-color:silver;
        }
        #runparams {
            margin-left:4em;
        }
        #ffentries {
            margin-left:4em;
        }
        #ffentries_table td {
            vertical-align:top;
        }
        .ffentries_table_cell_contents {
            width:20em;
        }
        </style>
        <p id="title"><b>Experiment Status</b></p>
<?php
require_once('LogBook.inc.php');

function decorate_end_time ( $time ) {
    if( !is_null( $time ))
        return $time->toStringShort();
    return '<span style="color:red;">IN PROGRESS</span>';
}
/* All operations with LogBokk API will be enclose enclosed into this
 * exception catch block.
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( 'FF' ); // find_last_experiment();
    if( !is_null( $experiment )) {

        $e_name       = $experiment->name();
        $e_begin_time = $experiment->begin_time()->toStringShort();
        $e_end_time   = decorate_end_time( $experiment->end_time());

        $e_num_shifts = $experiment->num_shifts();
        $shift = $experiment->find_last_shift();
        if( !is_null( $shift )) {
            $s_leader   = $shift->leader();
            $s_begin_time = $shift->begin_time()->toStringShort();
            $s_end_time   = decorate_end_time( $shift->end_time());
        }

        $e_num_runs = $experiment->num_runs();
        $run = $experiment->find_last_run();
        if( !is_null( $run )) {
            $r_num        = $run->num();
            $r_begin_time = $run->begin_time()->toStringShort();
            $r_end_time   = decorate_end_time( $run->end_time());
        }
        $e_num_entries = $experiment->num_entries();
        $entry = $experiment->find_last_entry();
    }
    echo <<<HERE
<div id="experiment_hdr">
    <table>
        <tbody>
            <tr>
                <td valign="center" class="hdr_table_cell_1st"><b>Status</b></td>
                <td>
                    <table>
                        <tbody>
                            <tr>
                                <td class="table_cell_1st"><b>Name</b></td>
                                <td class="table_cell_name">&nbsp;$e_name&nbsp;</td>
                            </tr>
                            <tr>
                                <td class="table_cell_1st"><b>Begin time</b></td>
                                <td class="table_cell_time">&nbsp;$e_begin_time&nbsp;</td>
                            </tr>
                            <tr>
                                <td class="table_cell_1st"><b>End time</b></td>
                                <td class="table_cell_time">&nbsp;$e_end_time&nbsp;</td>
                            </tr>
                        </tbody>
                    </table>
                </td>
                <td class="table_cell_2d"></td>
                <td>
                    <table>
                        <tbody>
                            <tr>
                                <td class="table_cell_1st"><b># Shifts</b></td>
                                <td class="table_cell_number">&nbsp;$e_num_shifts&nbsp;</td>
                            </tr>
                            <tr>
                                <td class="table_cell_1st"><b># Runs</b></td>
                                <td class="table_cell_number">&nbsp;$e_num_runs&nbsp;</td>
                            </tr>
                            <tr>
                                <td class="table_cell_1st"><b># Messages</b></td>
                                <td class="table_cell_number">&nbsp;$e_num_entries&nbsp;</td>
                            </tr>
                        </tbody>
                    </table>
                </td>
            </tr>
        </tbody>
    </table>
</div>
<br>
<br>
<div id="experiment">
    <table>
        <tbody>
            <tr>
                <td valign="center" class="hdr_table_cell_1st"><b>Shift</b></td>
                <td>
                    <table>
                        <tbody>
                            <tr>
                                <td class="table_cell_1st"><b>Leader</b></td>
                                <td class="table_cell_name">&nbsp;$s_leader&nbsp;</td>
                            </tr>
                            <tr>
                                <td class="table_cell_1st"><b>Begin time</b></td>
                                <td class="table_cell_time">&nbsp;$s_begin_time&nbsp;</td>
                            </tr>
                            <tr>
                                <td class="table_cell_1st"><b>End time</b></td>
                                <td class="table_cell_time">&nbsp;$s_end_time&nbsp;</td>
                            </tr>
                        </tbody>
                    </table>
                </td>
                <td width="48em;"></td>
                <td valign="center" class="hdr_table_cell_1st"><b>Run</b></td>
                <td>
                    <table>
                        <tbody>
                            <tr>
                                <td class="table_cell_1st"><b>Number</b></td>
                                <td class="table_cell_name">&nbsp;$r_num&nbsp;</td>
                            </tr>
                            <tr>
                                <td class="table_cell_1st"><b>Begin time</b></td>
                                <td class="table_cell_time">&nbsp;$r_begin_time&nbsp;</td>
                            </tr>
                            <tr>
                                <td class="table_cell_1st"><b>End time</b></td>
                                <td class="table_cell_time">&nbsp;$r_end_time&nbsp;</td>
                            </tr>
                        </tbody>
                    </table>
                </td>
            </tr>
        </tbody>
    </table>
</div>
<br>
<br>
<br>
<div id="runparams">
    <table>
        <tbody>
            <tr>
                <td valign="center" class="hdr_table_cell_1st"><b>Params</b></td>
                <td>
                    <table cellpadding="3"  border="0">
                        <thead>
                            <th align="left" style="color:#0071bc;">
                                &nbsp;<b>Run Parameter</b>&nbsp;
                            </th>
                            <th align="left" style="color:#0071bc;">
                                &nbsp;<b>Update Time</b>&nbsp;
                            </th>
                            <th align="left" style="color:#0071bc;">
                                &nbsp;<b>Source</b>&nbsp;
                            </th>
                            <th align="left" style="color:#0071bc;">
                                &nbsp;<b>Value</b>&nbsp;
                            </th>
                            <th align="left" style="color:#0071bc;">
                                &nbsp;<b>Description</b>&nbsp;
                            </th>
                        </thead>
                        <tbody>
                            <tr>
                                <td><hr></td>
                                <td><hr></td>
                                <td><hr></td>
                                <td><hr></td>
                                <td><hr></td>
                            </tr>
HERE;
    if( !is_null( $experiment )) {
        $count = 0;
        $params = $experiment->run_params();
        foreach( $params as $p ) {

            $v = $run->get_param_value( $p->name());
            if( !is_null( $v )) {
                $v_updated = $v->updated()->toStringShort();
                $v_source  = $v->source();
                $v_value   = $v->value();
            } else {
                unset( $v_updated );
                unset( $v_source  );
                unset( $v_value   );
            }
            if( $count % 2 == 0 ) $style = 'style="background-color:silver;"';
            else $style = '';
            echo <<<HERE
                            <tr valign="top" {$style}>
                                <td>&nbsp;{$p->name()}&nbsp;</td>
                                <td>&nbsp;{$v_updated}&nbsp;</td>
                                <td>&nbsp;{$v_source}&nbsp;</td>
                                <td>&nbsp;{$v_value}&nbsp;</td>
                                <td>&nbsp;<i>{$p->description()}</i>&nbsp;</td>
                            </tr>
HERE;
                    $count++;
                }
            }
            echo <<<HERE
                            <tr>
                                <td><hr></td>
                                <td><hr></td>
                                <td><hr></td>
                                <td><hr></td>
                                <td><hr></td>
                            </tr>
                        </tbody>
                    </table>
                </td>
            </tr>
        </tbody>
    </table>
</div>
HERE;
echo <<<HERE
<br>
<br>
<div id="ffentries">
    <table cellpadding="3"  border="0">
        <thead>
            <th align="left" style="color:#0071bc;">
                &nbsp;<b>Relevance Time</b>&nbsp;
            </th>
            <th align="left" style="color:#0071bc;">
                &nbsp;<b>Insert Time</b>&nbsp;
            </th>
            <th align="left" style="color:#0071bc;">
                &nbsp;<b>Author</b>&nbsp;
            </th>
            <th align="left" style="color:#0071bc;">
                &nbsp;<b>Content</b>&nbsp;
            </th>
            <th align="left" style="color:#0071bc;">
                &nbsp;<b>Tags</b>&nbsp;
            </th>
            <th align="left" style="color:#0071bc;">
                &nbsp;<b>Attachments</b>&nbsp;
            </th>
        </thead>
        <tbody>
            <tr>
                <td><hr></td>
                <td><hr></td>
                <td><hr></td>
                <td><hr></td>
                <td><hr></td>
                <td><hr></td>
            </tr>
HERE;
    if( !is_null( $experiment )) {
        $count = 0;
        $entries = $experiment->entries();
        foreach( $entries as $e ) {

            $tags_str = '';
            $tags = $e->tags();
            foreach( $tags as $t ) {
                if( $tags_str == '' ) $tags_str = '&nbsp;'.$t->tag();
                else $tags_str = $tags_str.'<br>&nbsp;'.$t->tag();
                if( $t->value() != '' )
                    $tags_str = $tags_str.'='.$t->value();
            }
            $tags_str = $tags_str.'&nbsp;';

            $content_maxlen = 256;
            $content = substr( $e->content(), 0, $content_maxlen);
            if( strlen( $e->content()) > $content_maxlen )
                $content = $content.' ...';

            $attachments_str = '';
            $attachments = $e->attachments();
            foreach( $attachments as $a ) {
                $attachment_ref = '&nbsp;<a href="ShowAttachment.php?id='.$a->id().'">'.$a->description().'</a>&nbsp;';
                if( $attachments_str == '' ) $attachments_str = $attachment_ref;
                else $attachments_str = $attachments_str.'<br>'.$attachment_ref;
            }

            if( $count % 2 == 0 ) $style = 'style="background-color:silver;"';
            else $style = '';
            echo <<<HERE
            <tr valign="top" {$style}>
                <td>&nbsp;{$e->relevance_time()->toStringShort()}&nbsp;</td>
                <td>&nbsp;{$e->insert_time()->toStringShort()}&nbsp;</td>
                <td>&nbsp;{$e->author()}&nbsp;</td>
                <td><div class="ffentries_table_cell_contents"><i>{$content}</i></div></td>
                <td>{$tags_str}</td>
                <td>{$attachments_str}</td>
            </tr>
HERE;
                    $count++;
                }
            }
            echo <<<HERE
            <tr>
                <td><hr></td>
                <td><hr></td>
                <td><hr></td>
                <td><hr></td>
                <td><hr></td>
                <td><hr></td>
            </tr>
        </tbody>
    </table>
</div>
HERE;
            /* Finish the session by commiting the transaction.
             */
            $logbook->commit();

        } catch( LogBookException $e ) {
            print $e->toHtml();
            return;
        }
        ?>
    </body>
</html>