<?php

namespace LogBook;

require_once( 'logbook.inc.php' );

class LogBookRunAttr {

    /* Data members
     */
    private $connection;
    private $run;

    private $attr;
    private $val;

    /** Constructor
     */
    public function __construct ( $connection, $run, $attr, $val ) {
        $this->connection = $connection;
        $this->run  = $run;
        $this->attr = $attr;
        $this->val  = $val;
    }

    /* Accessors
     */
    public function run () {
        return $this->run; }

    public function id () {
        return $this->attr['id']; }

    public function class_name () {
        return $this->attr['class']; }

    public function name () {
        return $this->attr['name']; }

    public function type_name () {
        return $this->attr['type']; }

    public function description () {
        return $this->attr['descr']; }

    public function val () {
        return $this->val; }
}
?>
