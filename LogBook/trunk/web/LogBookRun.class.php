<?php
class LogBookRun {

    /* Data members
     */
    private $connection;
    private $experiment;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }

    /* Accessors
     */
    public function parent () {
        return $this->experiment; }

    public function id () {
        return $this->attr['id']; }

    public function num () {
        return $this->attr['num']; }

    public function exper_id () {
        return $this->attr['exper_id']; }

    public function begin_time () {
        return LogBookTime::from64( $this->attr['begin_time'] ); }

    public function end_time () {
        if( is_null( $this->attr['end_time'] )) return null;
        return LogBookTime::from64( $this->attr['end_time'] ); }

    public function in_interval ( $timestamp ) {
        return LogBookTime::in_interval(
            $timestamp,
            $this->attr['begin_time'],
            $this->attr['end_time'] ); }

    /*
     * =============================
     *   SUMMARY PARAMETERS VALUES
     * =============================
     */
    public function values ( $condition='' ) {

        $list = array();
        $run_id = $this->attr['id'];

        /* Det descriptions of run parameters for the experiment. We need to know
         * type names of the parameters to scedule a request to the corresponding
         * tables.
         */
        $params = $this->experiment->run_params();
        foreach( $params as $p ) {
            $param_id = $p->attr['id'];
            $param = $p->attr['param'];
            $type = $p->attr['type'];
            $extra_condition = $condition == '' ? '' : ' AND '.$condition;
            $result = $this->connection->query(
                'SELECT p.*,v.val FROM run_val AS p, run_val_'.$type.' AS v WHERE p.run_id='.$run_id.
                ' AND p.param_id='.$param_id.
                ' AND p.run_id=v.run_id AND p.param_id=v.param_id'.
                $extra_condition );

            $nrows = mysql_numrows( $result );
            for( $i=0; $i<$nrows; $i++ ) {
                array_push(
                    $list,
                    new LogBookRunVal(
                        $this->connection,
                        $this,
                        mysql_fetch_array( $result, MYSQL_ASSOC )));
            }
        }
        return $list;
    }

    /* Set a value of the specified run parameter
     */
    public function set_param_value ( $param, $value, $source, $updated, $allow_update=false ) {

        /* Find the parameter and get its identifier and its type.
         * Also prepare the value for the specified SQL type.
         */
        $param = $this->experiment->find_run_param_by_name( $param )
            or die( "failed to find the parameter");

        $param_id = $param->attr['id'];
        $type     = $param->attr['type'];

        if( $type=='TEXT') {
            $value4sql = "'".$value."'";
        } else if( $type =='INT' ) {
            if( 1 != sscanf( $value, "%d", $value4sql ))
                die( "not an integer value of the parameter: ".$value );
        } else if( $type =='DOUBLE' ) {
            if( 1 != sscanf( $value, "%lf", $value4sql ))
                die( "not a double precision value of the parameter: ".$value );
        } else {
            /* Treat it as the string */
            $value4sql = "'".$value."'";
        }
        $value_table = 'run_val_'.$type;

        /* Check if its value is already set, and if so - if we're allowed
         * to update it.
         */
        $run_id = $this->attr['id'];

        $result = $this->connection->query(
            'SELECT COUNT(*) "count" FROM run_val AS p WHERE p.run_id='.$run_id.
            ' AND p.param_id='.$param_id );
        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            die( "unexpected size of the result set");
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        if( $attr['count'] > 0 ) {
            if( !$allow_update)
                die( "the value was set before and it's not allowed to be updated" );
            $this->connection->query(
                "UPDATE run_val SET source='".$source."', updated=".$updated->to64().
                ' WHERE run_id='.$run_id.' AND param_id='.$param_id )
                or die( "failed to update the parameter value's bookeeping record" );
            $this->connection->query(
                "UPDATE ".$value_table." SET val=".$value4sql.
                ' WHERE run_id='.$run_id.' AND param_id='.$param_id )
                or die( "failed to update the parameter's value" );
        } else {
            $this->connection->query(
                "INSERT INTO run_val VALUES (".$run_id.",".$param_id.",'".$source."',".$updated->to64().")")
                or die( "failed to set the parameter value's bookeeping record" );
            $this->connection->query(
                "INSERT INTO ".$value_table." VALUES (".$run_id.",".$param_id.",".$value4sql.")")
                or die( "failed to set the parameter's value" );
        }

        /* Fetch the value and the bookkeeping info
         */
        $result = $this->connection->query(
            'SELECT p.*,v.val FROM run_val AS p, '.$value_table.' AS v WHERE p.run_id='.$run_id.
            ' AND p.param_id='.$param_id.
            ' AND p.run_id=v.run_id AND p.param_id=v.param_id'.
            $extra_condition );
        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            die( "unexpected size of the result set");
        return new LogBookRunVal(
            $this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    /* Close the open-ended run
     */
    public function close ( $end_time ) {

        if( !is_null($this->attr['end_time']))
            die( "the run is already closed" );

        /* Verify the value of the parameter
         */
        if( is_null( $end_time ))
            die( "end time can't be null");
        if( 0 != $this->experiment->in_interval( $end_time ))
            die( "incorrect end time - it falls off the experiment's intervall" );

        /* Make the update
         */
        $end_time_64 = LogBookTime::to64from( $end_time );
        $this->connection->query(
            'UPDATE "run" SET end_time='.$end_time_64.
            ' WHERE exper_id='.$this->exper_id().' AND num='.$this->attr['num'] )
            or die ("failed to close the run: ".mysql_error());

        /* Update the current state of the object
         */
        $this->attr['end_time'] = $end_time_64;
        return true;
    }
}
?>
