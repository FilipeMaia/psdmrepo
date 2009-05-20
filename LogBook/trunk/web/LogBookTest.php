<?php

require_once('LogBook.inc.php');

$host     = "localhost";
$user     = "gapon";
$password = "";
$database = "logbook";

$logbook = new LogBook( $host, $user, $password, $database );

echo "#### ALL EXPERIMENTS ####\n\n";
$experiments = $logbook->experiments();
foreach( $experiments as $e ) {
    print_r( $e->attr );
}

echo "#### A SUBSET OF EXPERIMENTS ####\n\n";
$experiments = $logbook->experiments('WHERE "id" < 3');
foreach( $experiments as $e ) {
    print_r( $e->attr );
}

echo "#### FIND EXPERIMENT BY ID ####\n\n";
print_r( $logbook->find_experiment_by_id(6)->attr);

echo "#### FIND EXPERIMENT BY NAME ####\n\n";
print_r( $logbook->find_experiment_by_name('MPI2010')->attr);

echo "#### FIND ALL SHIFTS OF AN EXPERIMENT ####\n\n";
$experiment = $logbook->find_experiment_by_name('MPI2010');
if( isset( $experiment )) {
    $shifts = $experiment->shifts();
    foreach( $shifts as $s ) {
        print_r( $s->attr );
    }
}


$experiment = $logbook->find_experiment_by_name('MPI2010');

echo "#### FIND ALL SUMMARY PARAMETERS OF EXPERIMENT'S RUNS ####\n\n";
if( isset( $experiment )) {
    $runs = $experiment->run_params();
    foreach( $runs as $p ) {
        print_r( $p->attr );
    }
}

echo "#### FIND ALL RUNS OF AN EXPERIMENT ####\n\n";
if( isset( $experiment )) {
    $runs = $experiment->runs();
    foreach( $runs as $r ) {
        print_r( $r->attr );
        echo "     VALUES OF RUN PARAMETERS\n";
        $values = $r->values();
        foreach( $values as $v ) {
            print_r( $v->attr );
        }
    }
}

?>
