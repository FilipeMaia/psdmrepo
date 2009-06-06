<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for listing shifts of an
 * experiment.
 */
if( isset( $_GET['id'] )) {
    if( 1 != sscanf( trim( $_GET['id'] ), "%d", $id ))
        die( "invalid format of the attachment identifier" );
} else
    die( "no valid attachment identifier" );

/* Proceed to the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $attachment = $logbook->find_attachment_by_id( $id )
        or die("no such attachment" );
?>
<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Show Attachment</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <style>
        #attachment {
                margin-left:4em;
            }
            .table_cell_1st {
                color:#0071bc;
                width:6em;
            }
            .table_cell_name {
                background-color:silver;
                width:9em;
            }
            .table_cell_number {
                background-color:silver;
                width:4em;
            }
            #document {
                border:1px solid red;
                width:640px;
                height:480px;
            }
        </style>
        <h1>Attachment : </h1>
        <div id="attachment">
            <table>
                <tbody>
                    <tr>
                        <td class="table_cell_1st">Description</td>
                        <td class="table_cell_name"><?php echo( $attachment->description()); ?></td>
                    </tr>
                    <tr>
                        <td class="table_cell_1st">Type</td>
                        <td class="table_cell_name"><?php echo( $attachment->document_type()); ?></td>
                    </tr>
                    <tr>
                        <td class="table_cell_1st">Size</td>
                        <td class="table_cell_number"><?php echo( $attachment->document_size()); ?> Bytes</td>
                    </tr>

                </tbody>
            </table>
            <br>
            <table>
                <tbody>
                    <tr valign="top">
                        <td class="table_cell_1st">Document</td>
                        <td id="document"><?php echo( $attachment->document()); ?></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>