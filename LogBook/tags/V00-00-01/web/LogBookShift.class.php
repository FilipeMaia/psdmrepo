<?php
class LogBookShift {
    private $connection;
    private $experiment;
    public $attr;
    public function __construct( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }
}
?>
