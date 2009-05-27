<?php

require_once('LogBook.inc.php');

/* Make database connection using default connection
 * parameters.
 */
$logbook = new LogBook();
?>

<!--
The page for reporting the contents of the LogBook database.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Complex read-only test for LogBook PHP API</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!--------------------->
        <h1>All experiments</h1>
        <?php
        LogBookTestTable::Experiment()->show( $logbook->experiments());
        ?>

        <!-------------------------->
        <h1>Selected experiments</h1>
        <?php
        LogBookTestTable::Experiment()->show( $logbook->experiments('WHERE "id" < 5'));
        ?>

        <!--------------------------->
        <h1>Find experiment by id</h1>
        <?php
        LogBookTestTable::Experiment()->show( array($logbook->find_experiment_by_id(6)));
        ?>

        <!----------------------------->
        <h1>Find experiment by name</h1>
        <?php
        LogBookTestTable::Experiment()->show( array($logbook->find_experiment_by_name('H2O')));
        ?>

        <!--------------------------------->
        <h1>All shifts of an experiment</h1>
        <h2>H2O</h2>
        <?php
        $experiment = $logbook->find_experiment_by_name('H2O');
        if( isset( $experiment ))
            LogBookTestTable::Shift()->show( $experiment->shifts());
        ?>

        <!------------------------------------------------------------>
        <h1>Definitions of summary run parameters of an experiment</h1>
        <h2>H2O</h2>
        <?php
        $experiment = $logbook->find_experiment_by_name('H2O');
        if( isset( $experiment ))
            LogBookTestTable::RunParam()->show( $experiment->run_params());
        ?>

        <!------------------------------>
        <h1>All runs of an experiment</h1>
        <h2>H2O</h2>
        <?php
        $experiment = $logbook->find_experiment_by_name('H2O');
        if( isset( $experiment ))
            LogBookTestTable::Run()->show( $experiment->runs());
        ?>

        <!------------------------------>
        <h1>Values of run parameters for all runs of an experiment</h1>
        <h2>H2O</h2>
        <?php
        $experiment = $logbook->find_experiment_by_name('H2O');
        if( isset( $experiment )) {
            $runs = $experiment->runs();
            foreach( $runs as $run ) {
                LogBookTestTable::RunVal()->show(
                    $run->values(),
                    '<h3>Run: '.$run->attr['num'].'</h3>' );
            }
        }
        ?>
    </body>
</html>