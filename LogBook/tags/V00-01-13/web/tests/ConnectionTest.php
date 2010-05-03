<?php

/* This sample application will print names of instruments
* and experiments for each instrument.
*/
require_once( 'LogBook/LogBook.inc.php' );

try {

    // Create the top-level API object using default connection parameters.
    // See the constructor of class to get an idea how to pass non-default
    // parameters to the constructor:
    //
    //   File: LogBook.class.php
    //
    $logbook = new LogBook();

    // Begin the database transaction
    //
    $logbook->begin();

    echo "<h1>Instruments and experiments</h1>";

    $instruments = $logbook->instruments();
    foreach( $instruments  as $i ) {

        echo "<br>&nbsp;&nbsp;instrument: <b>'{$i->name()}'</b> description: <b>'{$i->description()}'</b>";

        $experiments = $logbook->experiments_for_instrument ( $i->name());
        foreach( $experiments as $e ) {
            $description = substr( $e->description(), 0, 72 );
            echo "<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|";
            echo "<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-experiment: <b>'{$e->name()}'</b> description: <b>'{$description}...'</b>";
        }
        echo "<br>";
    }

    // Make sure the transaction is commited. This is especially REQUIRED
    // to ensure that all modifications to existing data an/or any new data
    // create during the session are saved in the database.
    //
    $logbook->commit();


    // And here follow exception handlers for three main packages
    // within the API.

} catch( LogBookException $e ) {
    echo $e->toHtml();
} catch( RegDBException $e ) {
    echo $e->toHtml();
} catch( LusiTimeException $e ) {
    echo $e->toHtml();
}
?>
