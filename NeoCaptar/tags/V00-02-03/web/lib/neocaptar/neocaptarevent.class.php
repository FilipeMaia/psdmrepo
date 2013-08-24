<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarEvent is a base class for history events.
 *
 * @author gapon
 */
class NeoCaptarEvent {

   /* Data members
     */
    private $connection;
    private $scope;
    private $scope_id;
    private $attr;

    /* Constructor
     */
    public function __construct ($connection,$scope, $scope_id, $attr) {
        $this->connection = $connection;
        $this->scope      = $scope;
        $this->scope_id   = $scope_id;
        $this->attr       = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function id         () { return $this->attr['id']; }
    public function scope      () { return $this->scope; }
    public function scope_id   () { return $this->scope_id; }
    public function event_time () { return LusiTime::from64( $this->attr['event_time'] ); }
    public function event_uid  () { return $this->attr['event_uid']; }
    public function event      () { return $this->attr['event']; }

    public function comments () {
        $list = array();
        $result = $this->connection->query("SELECT comment FROM {$this->connection->database}.{$this->scope()}_history_comments WHERE {$this->scope()}_history_id={$this->id()}");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push(
                $list,
                $attr['comment']);
        }
        return $list;
    }
}
?>
