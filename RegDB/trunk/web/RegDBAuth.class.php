<?php

require_once('AuthDB/AuthDB.inc.php');

/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class RegDBAuth provides an interface to the Authorization Service
 *
 * @author gapon
 */
class RegDBAuth {

    /* Data members
     */
    private $authdb;

    private static $instance = null;

    public static function instance() {
        if( is_null( RegDBAuth::$instance )) RegDBAuth::$instance = new RegDBAuth();
        return RegDBAuth::$instance;
    }

    public function __construct () {
        $this->authdb = new AuthDB();
    }

    public function authName() {
        return $_SERVER['REMOTE_USER'];
    }

    public function authType() {
        return $_SERVER['AUTH_TYPE'];
    }

    public function isAuthenticated() {
        return RegDBAuth::instance()->authName() != '';
    }

    public function canRead() {
    	// Anyone who's been authenticated can read the contents of
    	// this database.
    	//
        return $this->isAuthenticated();
    }

    public function canEdit() {
        if( !$this->isAuthenticated()) return false;
        $this->authdb->begin();
        return $this->authdb->hasPrivilege(
            RegDBAuth::instance()->authName(), null, 'RegDB', 'edit' );
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
