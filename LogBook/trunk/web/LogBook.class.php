<?php
class LogBook {

    /* Data members
     */
    private $connection;

    /* Constructor
     *
     * Construct the top-level API object using the specified connection
     * parameters. Put null to envorce default values of parameters.
     */
    public function __construct (
        $host     = null,
        $user     = null,
        $password = null,
        $database = null ) {

        $this->connection =
            new LogBookConnection (
                is_null($host)     ? LOGBOOK_DEFAULT_HOST : $host,
                is_null($user)     ? LOGBOOK_DEFAULT_USER : $user,
                is_null($password) ? LOGBOOK_DEFAULT_PASSWORD : $password,
                is_null($database) ? LOGBOOK_DEFAULT_DATABASE : $database);
    }

    /*
     * ==========================
     *   TRANSACTION MANAGEMENT
     * ==========================
     */
    public function begin () {
        $this->connection->begin (); }

    public function commit () {
        $this->connection->commit (); }

    public function rollback () {
        $this->connection->rollback (); }

    /* ===============
     *   EXPERIMENTS
     * ===============
     */
    public function experiments ( $condition = '' ) {

        $list = array();

        $result = $this->connection->query (
            'SELECT * FROM "experiment" '.$condition );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push(
                $list,
                new LogBookExperiment (
                    $this->connection,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function find_experiment_by_id ( $id ) {
        return $this->find_experiment_by_( 'id='.$id) ; }

    public function find_experiment_by_name ( $name ) {
        return $this->find_experiment_by_( "name='".$name."'") ; }

    private function find_experiment_by_ ( $condition ) {

        $result = $this->connection->query(
            'SELECT * FROM "experiment" WHERE '.$condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookExperiment(
                $this->connection,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function create_experiment ( $name, $begin_time, $end_time=null ) {

        /* Verify parameters
         */
        if( !is_null( $end_time ) && !$begin_time->less( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$begin_time."' isn't less than end time '".$end_time."'" );

        $this->connection->query (
            "INSERT INTO experiment VALUES(NULL,'".$name
            ."',".$begin_time->to64()
            .",".( is_null( $end_time ) ? 'NULL' : $end_time->to64()).")" );

        return $this->find_experiment_by_id( '(SELECT LAST_INSERT_ID())' );
    }
}
?>
