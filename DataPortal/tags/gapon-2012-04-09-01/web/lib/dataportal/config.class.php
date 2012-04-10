<?php

namespace DataPortal;

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;

use LusiTime\LusiTime;

/**
 * Class Logger encapsulates operations with the logging database
 */
class Config {

	// ---------------------------------------------------
    // --- SIMPLIFIED INTERFACE AND ITS IMPLEMENTATION ---
    // ---------------------------------------------------

    private static $instance = null;

    /**
     * Singleton to simplify certain operations.
     *
     * @return unknown_type
     */
    public static function instance() {
        if( is_null( Config::$instance )) Config::$instance =
        	new Config (
        		PORTAL_DEFAULT_HOST,
				PORTAL_DEFAULT_USER,
				PORTAL_DEFAULT_PASSWORD,
				PORTAL_DEFAULT_DATABASE );
        return Config::$instance;
    }
	
    /* Parameters of the object
     */
    private $host;
    private $user;
    private $password;
    public  $database;

    /* Current state of the object
     */
    private $link;
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
    public function __construct ( $host, $user, $password, $database ) {
        $this->host      = $host;
        $this->user      = $user;
        $this->password  = $password;
        $this->database  = $database;
    }

    /**
     * Destructor
     */
    public function __destruct () {

    	// Do not close this connection from here because it might be shared
    	// with other instances of the class (also from other APIs).
    	//
        //if( isset( $this->link )) mysql_close( $this->link );
    }

    /* ===============================================
     *   METHODS FOR MANAGING MIGRATION SUBSCRIPTIONS
     * ===============================================
     */

    /* Subscribe the current user for e-mail notifications on delayed migrations
     * if the flag value is set to TRUE. Unsubscribe otherwise.
     */
    public function subscribe4migration_if ( $subscribe, $subscriber, $address ) {

    	$this->connect();

    	$authdb = AuthDB::instance();
    	$authdb->begin();

    	$subscriber_str = $this->escape_string( trim( $subscriber ));
    	$address_str    = $this->escape_string( trim( $address ));
    	$by             = $this->escape_string( trim( $authdb->authName()));
    	$now            = LusiTime::now();
    	$host_str       = $this->escape_string( trim( $authdb->authRemoteAddr()));

    	$this->query(
			$subscribe ?
    		"INSERT INTO subscriber_migration VALUES (NULL,'{$subscriber_str}','{$address_str}','{$by}',{$now->to64()},'{$host_str}')" :
    		"DELETE FROM subscriber_migration WHERE subscriber='{$subscriber_str}' AND address='{$address_str}'"
		);

		$url = ($_SERVER[HTTPS] ? "https://" : "http://" ).$_SERVER['SERVER_NAME'].'/apps-dev/portal/DataMigrationMonitor';

        if( $subscribe )                
        	$this->do_notify(
        		$address,
        		"*** SUBSCRIBED ***",
				<<<HERE
                             ** ATTENTION **

The message was sent by the automated notification system because this e-mail
has been just registered to recieve alerts on data migration delays within
PCDS DAQ and OFFLINE.

The registration has been requested by:

  '{$authdb->authName()}' @ {$authdb->authRemoteAddr()} [ {$now->toStringShort()} ]

To unsubscribe from this service, please use the Data Migration Monitor app:

  {$url}

HERE
       		);
        else
        	$this->do_notify(
        		$address,
        		"*** UNSUBSCRIBED ***",
				<<<HERE
                             ** ATTENTION **

The message was sent by the automated notification system because this e-mail
has been just unregistered from recieving alerts on data migration delays within
PCDS DAQ and OFFLINE.

The change has been requested by:

  '{$authdb->authName()}' @ {$authdb->authRemoteAddr()} [ {$now->toStringShort()} ]

To subscribe back to this service, please use the Data Migration Monitor app:

  {$url}

HERE
			);
    }

    /* Check if the current user is subscribed for e-mail notifications on delayed
     * migrations, and if so return an array with the details of the subscription.
     * Return null otherwise.
     *
     *   Key             | Type            | Description
     *   ----------------+-----------------+------------------------------------------------------------------------------
     *   id              | unsigned number | unique identifier of the record in the database
     *   subscriber      | string          | user account of a person subscribed
     *   address         | string          | e-mail address of the subscriber
     *   subscribed_by   | LusiTime        | user account of a person who requested the subscription 
     *   subscribed_time | LusiTime        | time when the subscription was made 
     *   subscribed_host | string          | host (IP address or DNS name) name from which the operation was requested
     *
     */
    public function check_if_subscribed4migration ( $subscriber, $address ) {

    	$this->connect();

    	$subscriber_str = $this->escape_string( trim( $subscriber ));
    	$address_str    = $this->escape_string( trim( $address ));
    	$result = $this->query(
    		"SELECT * FROM subscriber_migration WHERE subscriber='{$subscriber_str}' AND address='{$address_str}'"
    	);
    	$nrows = mysql_numrows( $result );
    	if( !$nrows ) return null;
    	if( $nrows != 1 )
			throw new DataPortalException (
				__METHOD__,
				"duplicate entries for migration subscriber: {$subscriber} ({$address}) in database. Database can be corrupted." );
		$row = mysql_fetch_array( $result, MYSQL_ASSOC );
       	$row['subscribed_time'] = LusiTime::from64($row['subscribed_time']);
       	return $row;
    }

    /* Get all known subscribers for delayed migration notifications.
     * 
     * The method will return an array (a list) of entries similar to
     * the ones reported by the previous method.
     */
    public function get_all_subscribed4migration () {
	    $list = array();
   		$this->connect();
    	$result = $this->query( "SELECT * FROM subscriber_migration" );
    	$nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
        	$row = mysql_fetch_array( $result, MYSQL_ASSOC );
        	$row['subscribed_time'] = LusiTime::from64($row['subscribed_time']);
        	array_push ( $list, $row );
        }
        return $list;
    }

    public function do_notify( $address, $subject, $body ) {
        $tmpfname = tempnam("/tmp", "webportal");
        $handle = fopen( $tmpfname, "w" );
        fwrite( $handle, $body );
        fclose( $handle );

        shell_exec( "cat {$tmpfname} | mail -s '{$subject}' {$address} -- -F 'LCLS Data Migration Monitor'" );

        // Delete the file only after piping its contents to the mailer command.
        // Otherwise its contents will be lost before we use it.
        //
        unlink( $tmpfname );
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
        if( !mysql_query( $transition, $this->link ))
            throw new DataPortalException (
                __METHOD__,
                "MySQL error: ".mysql_error( $this->link ).', in query: '.$transition );
    }

    /* =================
     *   MySQL QUERIES
     * =================
     */
    public function query ( $sql ) {

        $this->connect();

        if( !$this->in_transaction )
            throw new DataPortalException (
                __METHOD__, "no active transaction" );

        $result = mysql_query( $sql, $this->link );
        if( !$result )
            throw new DataPortalException (
                __METHOD__,
                "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );
        return $result;
    }

    /* ==========================================================
     *   MISC. OPERATIONS REQUIREING DATABASE SERVER CONNECTION
     * ==========================================================
     */
    public function escape_string( $text ) {
        return mysql_real_escape_string( $text, $this->link );  }
 
    /**
     * Make a connection if this hasn't been done yet.
     */
    private function connect () {
        if( !isset( $this->link )) {

            /* Connect to MySQL server and register the shutdown function
             * to clean up after self by doing 'ROLLBACK'. This would allow
             * to use so called 'persistent' MySQL connections.
             * 
             * NOTE: using the 'persistent' connection. This connection won't be
             * closed by 'mysql_close()'.
             */
        	$new_link = false; // true;
            $this->link = mysql_pconnect( $this->host, $this->user, $this->password, $new_link );
            if( !$this->link )
                throw new DataPortalException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_connect" );

            if( !mysql_select_db( $this->database, $this->link ))
                throw new DataPortalException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_select_db" );

            $sql = "SET SESSION SQL_MODE='ANSI'";
            if( !mysql_query( $sql, $this->link ))
                throw new DataPortalException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );

            $sql = "SET SESSION AUTOCOMMIT=0";
            if( !mysql_query( $sql, $this->link ))
                throw new DataPortalException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );

            register_shutdown_function( array( $this, "rollback" ));
        }
    }
}

/* =======================
 * UNIT TEST FOR THE CLASS
 * =======================
 *
try {
	$user    = 'gapon';
	$address = $user.'@slac.stanford.edu';
	$config  = Config::instance();
	$config->begin();

	$config->subscribe4migration_if (
		is_null( $config->check_if_subscribed4migration ( $user, $address )),
		$user,
		$address );

	foreach( $config->get_all_subscribed4migration() as $entry ) {
		print(
			'<br>id              : <b>'.$entry['id'].'</b>'.
		    '<br>subscriber      : <b>'.$entry['subscriber'].'</b>'.
		    '<br>address         : <b>'.$entry['address'].'</b>'.
		    '<br>subscribed_by   : <b>'.$entry['subscribed_by'].'</b>'.
		    '<br>subscribed_time : <b>'.$entry['subscribed_time']->toStringShort().'</b>'.
		    '<br>subscribed_host : <b>'.$entry['subscribed_host'].'</b>'
		);
	}
	$config->commit();

} catch ( Exception           $e ) { print $e; }
  catch ( DataPortalException $e ) { print $e->toHtml(); }
*/
?>
