<?php

namespace RegDB;

require_once( 'regdb.inc.php' );
require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;

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
        return $this->authdb->authName(); // $_SERVER['REMOTE_USER'];
    }

    public function authType() {
        return $this->authdb->authType(); // $_SERVER['AUTH_TYPE'];
    }

    public function isAuthenticated() {
        return $this->authdb->isAuthenticated(); // RegDBAuth::instance()->authName() != '';
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

    /* Check if the curent user is allowed to manage POSIX group of
     * the specified experiment.
     * 
     * NOTE: The current implementation of the method requires
     * the experiment to exist. NO complain will be made if it
     * doen't, the 'false' will be returned.
     */
    public function canManageLDAPGroupOf( $name ) {

        if( !$this->isAuthenticated()) return false;

    	/* Find experiment in order to get its identifier.
    	 */
    	$regdb = new RegDB();
    	$regdb->begin();
    	$experiment = $regdb->find_experiment_by_unique_name( $name );
    	if( is_null( $experiment )) return false;
    	/*
    		throw new RegDBException (
            	__METHOD__,
            	"no such experiement: ".$name );
        */
        $this->authdb->begin();
        return $this->authdb->hasPrivilege(
            RegDBAuth::instance()->authName(), $experiment->id(), 'LDAP', 'manage_groups' );
    }

    /* Check if the curent user is allowed to manage the specified POSIX group.
     * 
     * This operation will go through all registered experiments and find out
     * thoses in which this group group is configured as the primary one.
     * Then the authorization record will be pulled for those experiments.
     * 
     * NOTE: The 'facilities' will be excluded from the search.
     */
    public function canManageLDAPGroup( $name ) {
  
        if( !$this->isAuthenticated()) return false;

        $this->authdb->begin();

        /* Check if a user is allowed to manage any groups.
         * If so then unconditionally proceed with the authorization.
         */
        if( $this->authdb->hasPrivilege(
            RegDBAuth::instance()->authName(),
            null, /* exper_id */
            'LDAP',
            'manage_groups' )) return true;

        /* Go through all experiments. Skip 'facilities'.
    	 */
        $regdb = new RegDB();
    	$regdb->begin();
    	foreach( $regdb->experiments() as $experiment ) {    	
    		if( $experiment->is_facility()) continue;
    		if( $experiment->POSIX_gid() == $name ) {
                if( $this->authdb->hasPrivilege(
                    RegDBAuth::instance()->authName(),
                    $experiment->id(),
                    'LDAP',
                    'manage_groups' )) return true;
    		}
    	}
    	return false;
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
