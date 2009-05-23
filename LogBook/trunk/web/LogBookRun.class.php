<?php
class LogBookRun {
    private $connection;
    private $experiment;
    public function parent() { return $this->experiment; }
    public $attr;
    public function __construct( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }
    public function values( $condition='' ) {

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
    public function set_param_value( $param, $value, $source, $updated, $allow_update=false ) {

        /* Find the parameter and get its identifier and its type.
         * Also prepare the value for the specified SQL type.
         */
        $param = $this->experiment->find_run_param_by_name( $param )
            or die( "failed to find the parameter");

        print_r($param->attr);

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
}
?>
