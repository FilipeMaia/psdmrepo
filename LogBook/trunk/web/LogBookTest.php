<?php
require_once('LogBook.inc.php');

/* Class for constructing HTML tables of specified configurations,
 * which includes the following parameters:
 * - an array of collumn names
 * - an array of the corresponding (to collumns) keys to be used when
 *   extracting data for table rows.
 */
class Table {
    private $cols;
    private $keys;
    private $class;
    public function __construct( $cols, $keys, $class ) {
        if(count($cols) != count($keys))
            die("illegal parameters to the contsructor");
        $this->cols = $cols;
        $this->keys = $keys;
        $this->class = $class;
    }

    public function begin() {
        echo <<<HERE
<table cellpadding="3"  border="0" class="$this->class">
    <thead style="color:#0071bc;">
HERE;
        foreach($this->cols as $c)
            echo <<<HERE
        <th align="left">
            &nbsp;<b>$c</b>&nbsp;</th>
HERE;
        echo <<<HERE
    </thead>
    <tbody>
        <tr>
HERE;
        foreach($this->cols as $c)
            echo <<<HERE
            <td><hr></td>
HERE;
        echo <<<HERE
        </tr>
HERE;
    }

    public function row($attr) {
        echo <<<HERE
        <tr>
HERE;
        foreach($this->keys as $k) {
            $v = $attr[$k];
            echo <<<HERE
            <td>&nbsp;$v&nbsp;</td>
HERE;
        }
    echo <<<HERE
        </tr>
HERE;
    }

    public function end() {
        echo <<<HERE
    </tbody>
</table>
HERE;
    }

    /*
     * Display a complete table instance from the input array
     */
    public function show( $list, $title=null ) {
        if( !is_null($title))
            echo <<<HERE
$title
HERE;
        $this->begin();
        foreach( $list as $e )
            $this->row($e->attr);
        $this->end();
    }
}

/* Predefined tables
 */
$table_experiments = new Table(
    array("Id", "Name", "Begin Time", "End Time"),
    array("id", "name", "begin_time", "end_time"),
    'table_2' );

$table_shifts = new Table(
    array("Experiment Id", "Begin Time", "End Time", "Shift Leader"),
    array("exper_id",      "begin_time", "end_time", "leader"),
    'table_4' );

$table_run_params = new Table(
    array("Id", "Name",  "Experiment Id", "Type", "Description"),
    array("id", "param", "exper_id",      "type", "descr"),
    'table_4' );

$table_runs = new Table(
    array("Id", "Number",  "Experiment Id", "Begin Time", "End Time"),
    array("id", "num",     "exper_id",      "begin_time", "end_time"),
    'table_4' );

$table_param_values = new Table(
    array("Run Id", "Param Id",  "Source", "Updated", "Value"),
    array("run_id", "param_id",  "source", "updated", "val"),
    'table_6' );

/* Make database connection
 */
$host     = "localhost";
$user     = "gapon";
$password = "";
$database = "logbook";

$logbook = new LogBook( $host, $user, $password, $database );
?>

<!--
The page for creating a new run.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Complex read-only test for LogBook PHP API</title>
    </head>
    <style>
    h1 {
        margin-left:1em;
    }
    h2 {
        margin-left:2em;
    }
    h3 {
        margin-left:4em;
    }
    .table_2 {
        margin-left:2em;
    }
    .table_4 {
        margin-left:4em;
    }
    .table_6 {
        margin-left:6em;
    }
    </style>
    <body>
        <!--------------------->
        <h1>All experiments</h1>
        <?php
        $table_experiments->show( $logbook->experiments());
        ?>

        <!-------------------------->
        <h1>Selected experiments</h1>
        <?php
        $table_experiments->show( $logbook->experiments('WHERE "id" < 5'));
        ?>

        <!--------------------------->
        <h1>Find experiment by id</h1>
        <?php
        $table_experiments->show( array($logbook->find_experiment_by_id(6)));
        ?>

        <!----------------------------->
        <h1>Find experiment by name</h1>
        <?php
        $table_experiments->show( array($logbook->find_experiment_by_name('H2O')));
        ?>

        <!--------------------------------->
        <h1>All shifts of an experiment</h1>
        <h2>H2O</h2>
        <?php
        $experiment = $logbook->find_experiment_by_name('H2O');
        if( isset( $experiment ))
            $table_shifts->show( $experiment->shifts());
        ?>

        <!------------------------------------------------------------>
        <h1>Definitions of summary run parameters of an experiment</h1>
        <h2>H2O</h2>
        <?php
        $experiment = $logbook->find_experiment_by_name('H2O');
        if( isset( $experiment ))
            $table_run_params->show( $experiment->run_params());
        ?>

        <!------------------------------>
        <h1>All runs of an experiment</h1>
        <h2>H2O</h2>
        <?php
        $experiment = $logbook->find_experiment_by_name('H2O');
        if( isset( $experiment ))
            $table_runs->show( $experiment->runs());
        ?>

        <!------------------------------>
        <h1>Values of run parameters for all runs of an experiment</h1>
        <h2>H2O</h2>
        <?php
        $experiment = $logbook->find_experiment_by_name('H2O');
        if( isset( $experiment )) {
            $runs = $experiment->runs();
            foreach( $runs as $run ) {
                $table_param_values->show(
                    $run->values(),
                    '<h3>Run: '.$run->attr['num'].'</h3>' );
            }
        }
        ?>
    </body>
</html>