<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for displaying definitions of
 * run summary parameters of an experiment.
 */
if( isset( $_POST['experiment_name'] ))
    $experiment_name = trim( $_POST['experiment_name'] );
else
    die( "no valid experiment name" );

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("no such experiment");
?>
<!--
The page for displaying definitions of run summary parameters of
an experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Run summary parameters of an experiment</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!------------------------------>
        <h1>Parameter definitions for experiment: <?php echo $experiment->name(); ?></h1>
        <?php
        LogBookTestTable::RunParam( "table_4" )->show( $experiment->run_params());
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>