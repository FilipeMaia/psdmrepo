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
        .hdr_table_cell_1st {
             width:3em;
             font-size:x-large;
        }
        .table_cell_1st {
             color:#0071bc;
             width:6em;
        }
        #ffentries {
            margin-left:4em;
        }
        #ffentries_hdr {
            font-size:x-large;
        }
        #ffentries_table {
            margin-left:5em;
        }
        #ffentries_table td {
            vertical-align:top;
        }
        .ffentries_table_cell_contents {
            width:20em;
            /*
            border-left-style:solid;
            border-left-width:thin;
            border-right-style:solid;
            border-right-width:thin;
            */
        }
        </style>
        <?php
        require_once('LogBook.inc.php');

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
                $e_end_time   = $experiment->end_time();
                if( !is_null( $e_end_time ))
                    $e_end_time = $e_end_time->toStringShort();

                $shift = $experiment->find_last_shift();
                if( !is_null( $shift )) {
                    $s_operator   = $shift->operator();
                    $s_begin_time = $shift->begin_time()->toStringShort();
                    $s_end_time   = $shift->end_time();
                    if( !is_null( $s_end_time ))
                        $s_end_time = $s_end_time->toStringShort();
                }

                $run = $experiment->find_last_run();
                if( !is_null( $run )) {
                    $r_num        = $run->num();
                    $r_begin_time = $run->begin_time()->toStringShort();
                    $r_end_time   = $run->end_time();
                    if( !is_null( $r_end_time ))
                        $r_end_time = $r_end_time->toStringShort();
                }
                $entry = $experiment->find_last_entry();
            }
            echo <<<HERE
            <h1>Experiment</h1>
            <table class="table_4">
                <tbody>
                    <tr>
                        <td class="table_cell_1st"><b>Name</b></td>
                        <td>$e_name</td>
                    </tr>
                    <tr>
                        <td class="table_cell_1st"><b>Begin time</b></td>
                        <td>$e_begin_time</td>
                    </tr>
                    <tr>
                        <td class="table_cell_1st"><b>End time</b></td>
                        <td>$e_end_time</td>
                    </tr>
                </tbody>
            </table>
            <br>
            <table class="table_4">
                <tbody>
                    <tr>
                        <td valign="center" class="hdr_table_cell_1st"><b>Shift</b></td>
                        <td>
                            <table>
                                <tbody>
                                    <tr>
                                        <td class="table_cell_1st"><b>Operator</b></td>
                                        <td>$s_operator</td>
                                    </tr>
                                    <tr>
                                        <td class="table_cell_1st"><b>Begin time</b></td>
                                        <td>$s_begin_time</td>
                                    </tr>
                                    <tr>
                                        <td class="table_cell_1st"><b>End time</b></td>
                                        <td>$s_end_time</td>
                                    </tr>
                                </tbody>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td></td>
                    </tr>
                    <tr>
                        <td valign="center" class="hdr_table_cell_1st"><b>Run</b></td>
                        <td>
                            <table>
                                <tbody>
                                    <tr>
                                        <td class="table_cell_1st"><b>Number</b></td>
                                        <td>$r_num</td>
                                    </tr>
                                    <tr>
                                        <td class="table_cell_1st"><b>Begin time</b></td>
                                        <td>$r_begin_time</td>
                                    </tr>
                                    <tr>
                                        <td class="table_cell_1st"><b>End time</b></td>
                                        <td>$r_end_time</td>
                                    </tr>
                                </tbody>
                            </table>
                        </td>
                    </tr>
                </tbody>
            </table>
            <div id="ffentries">
                <p id="ffentries_hdr"><b>Free-Form Entries</b></p>
                <table cellpadding="3"  border="0" id="ffentries_table">
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
                            &nbsp;<b># Attachments</b>&nbsp;
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
                        if( $tags_str == '' ) $tags_str = $t->tag();
                        else $tags_str = $tags_str.',<br>'.$t->tag();
                        if( $t->value() != '' )
                            $tags_str = $tags_str.'='.$t->value();
                    }
                    $content_maxlen = 256;
                    $content = substr( $e->content(), 0, $content_maxlen);
                    if( strlen( $e->content()) > $content_maxlen )
                        $content = $content.' ...';
                    $num_attachments = count( $e->attachments());
                    if( $count % 2 == 0 ) $style = 'style="background-color:silver;"';
                    else $style = '';
                    echo <<<HERE
                        <tr {$style}>
                            <td>&nbsp;{$e->relevance_time()->toStringShort()}&nbsp;</td>
                            <td>&nbsp;{$e->insert_time()->toStringShort()}&nbsp;</td>
                            <td>&nbsp;{$e->author()}&nbsp;</td>
                            <td><div class="ffentries_table_cell_contents"><i>{$content}</i></div></td>
                            <td>&nbsp;{$tags_str}&nbsp;</td>
                            <td>&nbsp;$num_attachments&nbsp;</td>
                        </tr>
HERE;
                    $count++;
                }
            }
            echo <<<HERE
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