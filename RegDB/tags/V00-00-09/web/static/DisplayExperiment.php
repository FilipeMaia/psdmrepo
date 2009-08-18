<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for displaying parameters of an experiment.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else
    die( "no valid experiment identifier" );


/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $experiment = $regdb->find_experiment_by_id( $id )
        or die( "no such experiment" );

    $instrument = $experiment->instrument();
    $group      = $experiment->POSIX_gid();
?>
<!--
The page for isplaying parameters of an experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Experiment: <?php echo $experiment->name(); ?></title>
    </head>
    <link rel="stylesheet" type="text/css" href="RegDBTest.css" />
    <body>
        <p id="title"><b>Experiment: </b><?php echo $experiment->name(); ?></p>
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
                                <td width="40%"><span class="table_header_cell"><b>Experiment</b></span></td>
                                <td width="60%"><input type="text" value="<?php echo $experiment->name(); ?>"/></td>
                            </table>
                        </td>
                        <td align="left" width="65%" rowspan="5">
                            <textarea style="width:100%; height:100%;"><?php echo $experiment->description(); ?></textarea></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>Instrument</b></span></td>
                                <td width="60%"><a href="DisplayInstrument.php?id=<?php echo $instrument->id(); ?>"><?php echo $instrument->name(); ?></a></td>
                            </table>
                        </td>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>Begin Time</b></span></td>
                                <td width="60%"><input type="text" value="<?php echo $experiment->begin_time()->toStringShort(); ?>"/></td>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>End Time</b></span></td>
                                <td width="60%"><input type="text" value="<?php echo $experiment->end_time()->toStringShort(); ?>"/></td>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                        <td align="center" width="65%"><span class="table_header_cell"><b>Contact Info</b></span></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>POSIX Group</b></span></td>
                                <td width="60%"><a href="DisplayGroup.php?name=<?php echo $group; ?>"><?php echo $group; ?></a></td>
                            </table>
                        </td>
                        <td align="left" width="65%" rowspan="2">
                            <textarea style="width:100%; height:100%;"><?php echo $experiment->contact_info(); ?></textarea></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>Leader</b></span></td>
                                <td width="60%"><input type="text" value="<?php echo $experiment->leader_account(); ?>"/></td>
                            </table>
                        </td>
                    </tr>
                    <tr>
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
                        $names = $experiment->param_names();
                        foreach( $names as $name ) {
                            $param = $experiment->find_param_by_name( $name )
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