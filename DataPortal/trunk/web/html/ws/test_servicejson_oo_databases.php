<?php

/**
 * This is an example of hwo to use the JSON Web services framework
 * in the object-oriented way.
 */
require_once 'dataportal/dataportal.inc.php' ;

class MyService extends \DataPortal\ServiceJSON {

    public function __construct () {
        parent::__construct ('GET') ;
    }

    protected function handler () {

        $exper_id = $this->required_int ('exper_id') ;

        $experiment = $this->regdb()->find_experiment_by_id ($exper_id) ;
        if (is_null($experiment)) $this->abort ('no such experiment') ;

        $this->finish (array (
            'exper_id'   => $experiment->id() ,
            'exper_name' => $experiment->name()
        )) ;
    }
}
$service = new MyService () ;
$service->run () ;

?>
