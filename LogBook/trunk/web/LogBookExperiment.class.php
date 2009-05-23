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
    public function create_shift( $leader, $begin_time, $end_time=null ) {
        $sql = 'INSERT INTO "shift" VALUES('.$this->attr['id']
            .",".$begin_time->to64()
            .",".($end_time==null?'NULL':$end_time->to64())
            .",'".$leader."')";
        echo $sql."\n";
        $result = $this->connection->query( $sql )
            or die ("failed to create new shift: ".mysql_error());
        return $this->find_shift_by_begin_time($begin_time);
    }
    public function find_shift_by_begin_time( $begin_time ) {
        return $this->find_shift_by_( "begin_time=".$begin_time) ;
    }
    private function find_shift_by_( $condition=null ) {
        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT * FROM "shift" WHERE exper_id='.
            $this->attr['id'].$extra_condition );
        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookShift(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));
        return NULL;
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
        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
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
    public function find_run_by_id( $id ) {
        return $this->find_run_by_( 'id='.$id) ;
    }
    public function find_run_by_num( $num ) {
        return $this->find_run_by_( "num=".$num) ;
    }
    public function find_last_run() {
        return $this->find_run_by_(
            'id=(SELECT MAX(id) FROM "run" WHERE exper_id='.
            $this->attr['id'].')' );
    }
    private function find_run_by_( $condition=null ) {
        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT * FROM "run" WHERE exper_id='.
            $this->attr['id'].$extra_condition );
        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookRun(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));
        return NULL;
    }
    public function create_run( $num, $begin_time, $end_time=null ) {
        $run_num = $num > 0 ? $num : $this->allocate_run($num);
        $sql = 'INSERT INTO "run" VALUES(NULL,'.$run_num
            .",".$this->attr['id']
            .",".$begin_time->to64()
            .",".($end_time==null?'NULL':$end_time->to64()).")";
        echo $sql."\n";
        $result = $this->connection->query( $sql )
            or die ("failed to create new run: ".mysql_error());
        return $this->find_run_by_id('(SELECT LAST_INSERT_ID())');
    }
    private function allocate_run() {
        $result = $this->connection->query( 'SELECT MAX(num) "num" FROM "run" WHERE exper_id='.$this->attr['id'] );
        $nrows = mysql_numrows( $result );
        if( $nrows == 1 ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            if(isset($attr['num'])) return 1 + $attr['num'];
            return 1;
        }
        return NULL;
    }
    public function find_run_param_by_id( $id ) {
        return $this->find_run_param_by_( 'id='.$id) ;
    }
    public function find_run_param_by_name( $name ) {
        return $this->find_run_param_by_( "param='".$name."'") ;
    }
    private function find_run_param_by_( $condition=null ) {
        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->connection->query( 'SELECT * FROM "run_param" WHERE exper_id='.$this->attr['id'].$extra_condition );
        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookRunParam(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));
        return NULL;
    }
    public function create_run_param( $param, $type, $descr ) {
        $sql = "INSERT INTO \"run_param\" VALUES(NULL,'".$param
            ."',".$this->attr['id']
            .",'".$type."','".$descr."')";
        echo $sql."\n";
        $result = $this->connection->query( $sql )
            or die ("failed to create new run parameter: ".mysql_error());
        return $this->find_run_param_by_id('(SELECT LAST_INSERT_ID())');
    }
}
?>
