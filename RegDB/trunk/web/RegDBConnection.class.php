<?php

/**
 * Class RegDBConnection encapsulates operations with the database
 */
class RegDBConnection {

    /* Parameters of the object
     */
    private $host;
    private $user;
    private $password;
    private $database;
    private $ldap_host;

    /* Current state of the object
     */
    private $link;
    private $ldap_ds;
    private $in_transaction = false;

    /**
     * Constructor
     *
     * The constructor won't make any actual connection attempts. This will be deffered
     * to operations dealing with queries, transactions, etc.
     *
     * @param string $host
     * @param string $user
     * @param string $password
     * @param string $database 
     */
    public function __construct ( $host, $user, $password, $database, $ldap_host ) {
        $this->host      = $host;
        $this->user      = $user;
        $this->password  = $password;
        $this->database  = $database;
        $this->ldap_host = $ldap_host;
    }

    /**
     * Destructor
     */
    public function __destruct () {
        if( isset( $this->link    )) mysql_close( $this->link );
        if( isset( $this->ldap_ds )) ldap_close( $this->ldap_ds );
    }

    /*
     * ================================
     *   MySQL TRANSACTION MANAGEMENT
     * ================================
     */
    public function begin () {
        $this->connect();
        if( $this->in_transaction ) return;
        $this->transaction( 'BEGIN' );
        $this->in_transaction = true;
    }

    public function commit () {
        $this->connect();
        if( !$this->in_transaction ) return;
        $this->transaction( 'COMMIT' );
        $this->in_transaction = false;
    }

    public function rollback () {
        $this->connect();
        if( !$this->in_transaction ) return;
        $this->transaction( 'ROLLBACK' );
        $this->in_transaction = false;
    }

    private function transaction ( $transition ) {
        if( !mysql_query( $transition ))
            throw new RegDBException (
                __METHOD__,
                "MySQL error: ".mysql_error().', in query: '.$transition );
    }

    /* =================
     *   MySQL QUERIES
     * =================
     */
    public function query ( $sql ) {

        $this->connect();

        if( !$this->in_transaction )
            throw new RegDBException (
                __METHOD__, "no active transaction" );

        $result = mysql_query( $sql );
        if( !$result )
            throw new RegDBException (
                __METHOD__,
                "MySQL error: ".mysql_error().', in query: '.$sql );
        return $result;
    }

    /* ================
     *   LDAP QUERIES
     * ================
     */
    public function posix_groups ( ) {

        $this->connect();

        $list = array();

        $sr = ldap_search( $this->ldap_ds, "ou=Group,dc=reg,o=slac", "cn=*" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $info = ldap_get_entries( $this->ldap_ds, $sr );

        $num_groups = $info["count"];
        for( $i = 0; $i < $num_groups; $i++ )
            array_push( $list, $info[$i]["cn"][0] );

        return $list;
    }

    public function is_known_posix_group ( $name ) {

        $this->connect();

        $trim_name = trim( $name );
        if( $trim_name == '' )
            throw new RegDBException (
                __METHOD__,
               "group name can't be empty" );

        $sr = ldap_search( $this->ldap_ds, "ou=Group,dc=reg,o=slac", "cn=$trim_name" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $info = ldap_get_entries( $this->ldap_ds, $sr );

        $num_groups = $info["count"];
        if( $num_groups == 0 ) return false;
        if( $num_groups != 1 )
            throw new RegDBException (
                __METHOD__,
               "inconsistent result set returned from LDAP server" );
        return true;
    }

    public function is_member_of_posix_group ( $group, $uid ) {

        $this->connect();

        $trim_uid = trim( $uid );

        $members = $this->posix_group_members( $group );
        foreach( $members as $member )
            if( $member['uid'] == $trim_uid  ) return true;
        return false;
    }

    public function posix_group_members ( $name ) {

        $this->connect();

        $trim_name = trim( $name );
        if( $trim_name == '' )
            throw new RegDBException (
                __METHOD__,
               "group name can't be empty" );

        $sr = ldap_search( $this->ldap_ds, "ou=Group,dc=reg,o=slac", "cn=$trim_name" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $info = ldap_get_entries( $this->ldap_ds, $sr );

        $num_groups = $info["count"];
        if( $num_groups == 0 )
            throw new RegDBException (
                __METHOD__,
               "no such group known to LDAP server" );
        if( $num_groups != 1 )
            throw new RegDBException (
                __METHOD__,
               "inconsistent result set returned from LDAP server" );

        $result = array();
        for( $i = 0; $i < $info[0]["memberuid"]["count"]; $i++ ) {
            $uid = $info[0]["memberuid"][$i];
            $full_name = "";
            $passwd = posix_getpwnam( $uid );
            if( $passwd ) $full_name = $passwd["gecos"];
            array_push(
                $result,
                array(
                    "uid" => $uid,
                    "gecos" => $full_name,
                    "email" => $uid.'@slac.stanford.edu' ));
        }
        return $result;
    }

    /**
     * Make a connection if this hasn't been done yet.
     */
    private function connect () {
        if( !isset( $this->link )) {

            /* Connect to MySQL server
             */
            $this->link = mysql_connect( $this->host, $this->user, $this->password );
            if( !$this->link )
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error().", in function: mysql_connect" );

            if( !mysql_select_db( $this->database ))
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error().", in function: mysql_select_db" );

            $sql = "SET SESSION SQL_MODE='ANSI'";
            if( !mysql_query( $sql ))
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error().', in query: '.$sql );

            $sql = "SET SESSION AUTOCOMMIT=0";
            if( !mysql_query( $sql ))
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error().', in query: '.$sql );

            /* Connect to LDAP server
             */
            $this->ldap_ds = ldap_connect( $this->ldap_host );
            if( !$this->ldap_ds )
                throw new RegDBException (
                    __METHOD__,
                    "failed to connect to LDAP server: ".$this->ldap_host );

            if( !ldap_bind( $this->ldap_ds ))
                throw new RegDBException (
                    __METHOD__,
                   "failed to bind to LDAP server due to: ".ldap_error( $this->ldap_ds ));
        }
    }
}

/* ==========================
 * UNIT TEST FOR LDAP METHODS
 * ==========================
 *

require_once( "RegDB.inc.php");

$conn = new RegDBConnection (
    REGDB_DEFAULT_HOST,
    REGDB_DEFAULT_USER,
    REGDB_DEFAULT_PASSWORD,
    REGDB_DEFAULT_DATABASE,
    REGDB_DEFAULT_LDAP_HOST );

try {
    $groups = $conn->posix_groups();
    foreach( $groups as $group ) {
        $members = $conn->posix_group_members( $group );
        echo "<br><hr>Group: <b>" . $group."</b>";
        echo "<br>&nbsp;&nbsp;Members: ";
        foreach( $members as $member )
            echo "<br>&nbsp;&nbsp;&nbsp;&nbsp;<b>".$member["uid"]."</b>&nbsp;".$member["gecos"];
    }

    echo "<br><hr>";
    $test_groups = array_merge( $groups, array( "aaa" ));
    foreach( $test_groups as $group ) {
        if( $conn->is_known_posix_group( $group ))
            echo "<br>Group <b>$group</b> is known";
        else
            echo "<br>Group <b>$group</b> is NOT known";
    }
 } catch ( RegDBException $e ) {
    print( e.toHtml());
}
*/
?>
