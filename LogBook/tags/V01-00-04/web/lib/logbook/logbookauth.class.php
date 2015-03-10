<?php

namespace LogBook;

require_once( 'logbook.inc.php' );
require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;

/**
 * Class LogBookAuth provides an interface to the Authorization Service
 *
 * @author gapon
 */
class LogBookAuth {

    private static $instance = null;

    public static function instance() {
        if( is_null( LogBookAuth::$instance )) LogBookAuth::$instance = new LogBookAuth();
        return LogBookAuth::$instance;
    }

    public function authName        () { return AuthDb::instance()->authName(); }
    public function authType        () { return AuthDb::instance()->authName(); }
    public function isAuthenticated () { return LogBookAuth::instance()->authName() != ''; }

    public function canRead            ($exper_id, $user=null) { return $this->can($exper_id, $user, 'read'); }
    public function canPostNewMessages ($exper_id, $user=null) { return $this->can($exper_id, $user, 'post'); }
    public function canEditMessages    ($exper_id, $user=null) { return $this->can($exper_id, $user, 'edit'); }
    public function canDeleteMessages  ($exper_id, $user=null) { return $this->can($exper_id, $user, 'delete'); }
    public function canManageShifts    ($exper_id, $user=null) { return $this->can($exper_id, $user, 'manage_shifts' ); }

    private function can( $exper_id, $user, $priv ) {
        if( !$this->isAuthenticated()) return false;
        AuthDb::instance()->begin();
        return AuthDb::instance()->hasPrivilege(
            is_null($user) ? LogBookAuth::instance()->authName() : trim($user), $exper_id, 'LogBook', $priv );
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
