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
}
?>
