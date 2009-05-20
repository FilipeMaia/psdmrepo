<?php
class LogBookExperiment {
    private $connection;
    public $attr;
    public function __construct( $connection, $attr ) {
        $this->connection = $connection;
        $this->attr = $attr;
    }
    public function shifts($condition='') {
        $list = array();
        $extra_condition = $condition == '' ? '' : 'AND '.$condition;
        $result = $this->connection->query( 'SELECT * FROM "shift" WHERE exper_id='.$this->attr['id'].$extra_condition );
        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookShift(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }
    public function run_params($condition='') {
        $list = array();
        $extra_condition = $condition == '' ? '' : 'AND '.$condition;
        $result = $this->connection->query( 'SELECT * FROM "run_param" WHERE exper_id='.$this->attr['id'].$extra_condition );
        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookRunParam(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }
    public function runs($condition='') {
        $list = array();
        $extra_condition = $condition == '' ? '' : 'AND '.$condition;
        $result = $this->connection->query( 'SELECT * FROM "run" WHERE exper_id='.$this->attr['id'].$extra_condition );
        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookRun(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }
}
?>
