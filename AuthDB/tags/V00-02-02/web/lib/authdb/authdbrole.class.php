<?php

namespace AuthDB;

require_once( 'authdb.inc.php' );

/**
 * Class AuthDBRole is an abstraction for roles
 *
 * @author gapon
 */
class AuthDBRole {

   /* Data members
     */
    private $authdb;

    public $attr;

    /* Constructor
     */
    public function __construct ( $authdb, $attr ) {
        $this->authdb = $authdb;
        $this->attr = $attr;
    }

    public function parent () {
        return $this->authdb; }

    public function id () {
        return $this->attr['id']; }

    public function name () {
        return $this->attr['name']; }

    public function application () {
        return $this->attr['app']; }

    public function player () {
        $player = $this->attr['user'];
        $group = '';
        $user  = $player;
        if(( strlen($player) >=4 ) && ( substr( $player, 0, 4 ) == "gid:" )) {
            $group = substr( $player, 4 );
            $user = '';
        }
        return array( "group" => $group, "user" => $user );
    }

    /**
     * Get an experiment identifier (if any).
     *
     * Return 'null' if there is none.
     *
     * @return iteger or null
     */
    public function exper_id () {
        $exp_id = $this->attr['exp_id'];
        return $exp_id == '' ? null : $exp_id;
    }
}
?>
