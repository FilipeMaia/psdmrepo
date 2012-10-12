<?php

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $experiment = $SVC->regdb()->find_experiment_by_id ($SVC->required_int ('exper_id')) ;
    if (is_null($experiment)) $SVC->abort ('no such experiment') ;
    $SVC->finish (array (
        'exper_id'   => $experiment->id() ,
        'exper_name' => $experiment->name()
    )) ;
}) ;

?>
