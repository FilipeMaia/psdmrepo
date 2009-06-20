<!--
The page for creating displaying all experiments.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Display Experiments</title>
    </head>
    <link rel="stylesheet" type="text/css" href="RegDBTest.css" />
    <body>
        <p id="title"><b>Experiments</b></p>
        <div id="test_container">
            <table border="0" cellpadding="4px" cellspacing="0px" width="100%" align="center">
                <thead style="color:#0071bc;">
                    <th align="left" style="width:96px;">
                        <b>Instrument</b></th>
                    <th align="left">
                        <b>Experiment</b></th>
                    <th align="left">
                        <b>Begin Time</b></th>
                    <th align="left">
                        <b>End Time</b></th>
                    <th align="left">
                        <b>Description</b></th>
                </thead>
                <tbody>
                    <tr>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                    </tr>
                    <?php
                    require_once('RegDB.inc.php');
                    try {
                        $regdb = new RegDB();
                        $regdb->begin();
                        $instruments = $regdb->instruments();
                        foreach( $instruments as $i ) {
                            $experiments = $regdb->experiments_for_instrument( $i->name());
                            foreach( $experiments as $e ) {
                                //$description = substr( $e->description(), 0, 128 );
                                $description = $e->description();
                                $begin_time = $e->begin_time()->toStringShort();
                                $end_time = $e->end_time()->toStringShort();
                                echo <<< HERE
                    <tr>
                        <td align="left" valign="top">
                            <a href="DisplayInstrument.php?id={$i->id()}"><b>{$i->name()}</b></a></td>
                        <td align="left" valign="top" style="width:10em;">
                            <a href="DisplayExperiment.php?id={$e->id()}"><b>{$e->name()}</b></a></td>
                        <td  valign="top" style="width:10em;">
                            {$begin_time}</td>
                        <td  valign="top" style="width:10em;">
                            {$end_time}</td>
                        <td  valign="top">
                            <i>{$description}</i></td>
                    </tr>
                    <tr>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                    </tr>
HERE;
                            }
                        }
                        $regdb->commit();

                    } catch ( RegDBException $e ) {
                        print( $e->toHtml());
                    }
                    ?>
                </tbody>
            </table>
        </div>
    </body>
</html>

