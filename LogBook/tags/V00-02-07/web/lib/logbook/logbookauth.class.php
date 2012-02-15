<?php

namespace LogBook;

require_once( 'logbook.inc.php' );
require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;

/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class LogBookAuth provides an interface to the Authorization Service
 *
 * @author gapon
 */
class LogBookAuth {

    /* Data members
     */
    private $authdb;

    private static $instance = null;

    public static function instance() {
        if( is_null( LogBookAuth::$instance )) LogBookAuth::$instance = new LogBookAuth();
        return LogBookAuth::$instance;
    }

    public function __construct () {
        $this->authdb = new AuthDB();
    }

    public function authName() {
        return $this->authdb->authName(); //return $_SERVER['REMOTE_USER'];
    }

    public function authType() {
        return $this->authdb->authName(); //return $_SERVER['AUTH_TYPE'];
    }

    public function isAuthenticated() {
        return LogBookAuth::instance()->authName() != '';
    }

    public function canRead( $exper_id ) {
        return $this->can( $exper_id, 'read' );
    }

    public function canPostNewMessages( $exper_id ) {
        return $this->can( $exper_id, 'post' );
    }

    public function canEditMessages( $exper_id ) {
        return $this->can( $exper_id, 'edit' );
    }

    public function canDeleteMessages( $exper_id ) {
        return $this->can( $exper_id, 'delete' );
    }

    public function canManageShifts( $exper_id ) {
        return $this->can( $exper_id, 'manage_shifts' );
    }

    private function can( $exper_id, $priv ) {
        if( !$this->isAuthenticated()) return false;
        $this->authdb->begin();
        return $this->authdb->hasPrivilege(
            LogBookAuth::instance()->authName(), $exper_id, 'LogBook', $priv );
    }

    public static function reporErrorHtml( $message, $link=null ) {
        $suggested_back_link =
            is_null($link) ?
            'the <b>BACK</b> button of your browser' :
            '<a href="'.$link.'">here</a>';
        return <<<HERE
<center>
  <br>
  <br>
  <div style="background-color:#f0f0f0; border:solid 2px red; max-width:640px;">
    <h1 style="color:red;">Authorization Error</h1>
    <div style="height:2px; background-color:red;"></div>
    <p>{$message}</p>
    <p>Click {$suggested_back_link} to return to the previous context</p>
  </div>
</center>
HERE;
    }

}
?>
