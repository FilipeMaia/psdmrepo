<?php
class LogBook {
    private $connection;
    private $host;
    private $user;
    private $password;
    private $database;
    public function __construct($host, $user, $password, $database) {
        $this->connection =
            new LogBookConnection(
                $host, $user, $password, $database );
    }
    public function experiments($condition='') {
        $list = array();
        $result = $this->connection->query( 'SELECT * FROM "experiment" '.$condition );
        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookExperiment(
                    $this->connection,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }
    public function find_experiment_by_id( $id ) {
        return $this->find_experiment_by_( 'id='.$id) ;
    }
    public function find_experiment_by_name( $name ) {
        return $this->find_experiment_by_( "name='".$name."'") ;
    }
    private function find_experiment_by_( $condition ) {
        $result = $this->connection->query( 'SELECT * FROM "experiment" WHERE '.$condition );
        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookExperiment(
                $this->connection,
                mysql_fetch_array( $result, MYSQL_ASSOC ));
        return NULL;
    }
    public function create_experiment( $name, $begin_time, $end_time=null ) {
        $sql = "INSERT INTO experiment VALUES(NULL,'".$name
            ."',".$begin_time->to64()
            .",".($end_time==null?'NULL':$end_time->to64()).")";
        echo $sql."\n";
        $result = $this->connection->query( $sql )
            or die ("failed to create new experiment: ".mysql_error());
    }
}
?>
