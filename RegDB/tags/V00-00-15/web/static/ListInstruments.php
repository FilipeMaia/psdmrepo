<!--
The page for creating displaying all instruments.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Display Instruments</title>
    </head>
    <link rel="stylesheet" type="text/css" href="RegDBTest.css" />
    <body>
        <p id="title"><b>Instruments</b></p>
        <div id="test_container">
            <table border="0" cellpadding="4px" cellspacing="0px" width="100%" align="center">
                <thead style="color:#0071bc;">
                    <th align="left" style="width:96px;">
                        <b>Instrument</b></th>
                    <th align="left">
                        <b>Description</b></th>
                </thead>
                <tbody>
                    <tr>
                        <td><hr></td>
                        <td><hr></td>
                    </tr>
                    <?php
                    require_once('RegDB/RegDB.inc.php');
                    try {
                        $regdb = new RegDB();
                        $regdb->begin();
                        $instruments = $regdb->instruments();
                        foreach( $instruments as $i ) {
                            $description = $i->description();
                            echo <<< HERE
                    <tr>
                        <td align="left" valign="top">
                            <a href="DisplayInstrument.php?id={$i->id()}"><b>{$i->name()}</b></a></td>
                        <td  valign="top">
                            <i>{$description}</i></td>
                    </tr>
                    <tr>
                        <td></td>
                        <td></td>
                    </tr>
HERE;
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

