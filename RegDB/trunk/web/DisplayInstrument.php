<?php

require_once('RegDB.inc.php');

/*
 * This script will process a request for displaying parameters of an instrument.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "instrument identifier can't be empty" );
} else
    die( "no valid instrument identifier" );


/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $instrument = $regdb->find_instrument_by_id( $id )
        or die( "no such instrument" );
?>
<!--
The page for isplaying parameters of an instrument.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Instrument: <?php echo $instrument->name(); ?></title>
    </head>
    <link rel="stylesheet" type="text/css" href="RegDBTest.css" />
    <body>
        <p id="title"><b>Instrument: </b><?php echo $instrument->name(); ?></p>
        <div id="test_container">
            <table border="0" cellpadding="4px" cellspacing="0px" width="100%" align="center">
                <tbody>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                        <td align="center" width="65%"><span class="table_header_cell"><b>Description</b></span></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>Instrument</b></span></td>
                                <td width="60%"><input type="text" value="<?php echo $instrument->name(); ?>"/></td>
                            </table>
                        </td>
                        <td align="left" width="65%" rowspan="3">
                            <textarea style="width:100%; height:100%;"><?php echo $instrument->description(); ?></textarea></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                    </tr>
                </tbody>
            </table>
            <br>
            <p id="title1"><b>Parameters</b></p>
            <div id="test_container1">
                <table border="0" cellpadding="4px" cellspacing="0px" width="100%" align="center">
                    <thead style="color:#0071bc;">
                        <th align="left" style="width:256px;">
                            <b>Name</b></th>
                        <th align="left">
                            <b>Value</b></th>
                        <th align="left">
                            <b>Description</b></th>
                    </thead>
                    <tbody>
                        <tr>
                            <td><hr></td>
                            <td><hr></td>
                            <td><hr></td>
                        </tr>
                        <?php
                        $names = $instrument->param_names();
                        foreach( $names as $name ) {
                            $param = $instrument->find_param_by_name( $name )
                                or die( "inconsistent results from the Registry Database API" );
                            $value = substr( $param->value(), 0, 128 );
                            $description = $param->description();
                            echo <<< HERE
                        <tr>
                            <td align="left" valign="top">
                                <b>{$param->name()}</b></td>
                            <td class="table_cell" valign="top">
                                {$value}</td>
                            <td class="table_cell" style="width:32em;" valign="top">
                                <i>{$description}</i></td>
                        </tr>
                        <tr>
                            <td></td>
                            <td></td>
                            <td></td>
                        </tr>
HERE;
                        }
                        ?>
                    </tbody>
                </table>
            </div>
        </div>
    </body>
</html>
<?php

    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>