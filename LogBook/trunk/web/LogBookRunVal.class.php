<?php
class LogBookRunVal {
    private $connection;
    private $run;
    public function parent() { return $this->run; }
    public $attr;
    public function __construct( $connection, $run, $attr ) {
        $this->connection = $connection;
        $this->run = $run;
        $this->attr = $attr;
    }
}
?>
