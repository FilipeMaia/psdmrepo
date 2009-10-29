<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for displaying parameters of a run.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "run identifier can't be empty" );
} else
    die( "no valid run identifier" );

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $run = $logbook->find_run_by_id( $id )
        or die( "no such run" );

    $experiment = $run->parent();
    $instrument = $experiment->instrument();

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to access any information about the experiment' ));
        exit;
    }

    // Proceed to the operation
    //
    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    // Grid-like markup
    //
    define(C0,  0);
    define(C1,150);
    define(C2,250);
    define(C3,350);
    define(C4,450);
    define(C5,550);
    define(C6,650);

    define(R0,  0);
    define(R1, 20);
    define(R2, 50);
    define(R3, 70);
    define(R4,100);
    define(R5,120);
    define(R6,140);
    define(R7,170);
    define(R8,200);
    define(R9,220);
    define(RA,240);
    define(RB,260);
    define(RC,280);
    define(RD,310);
    define(RE,340);

    $con = new RegDBHtml( 0, 0, 800, 360 );
    $con->label(C0,R0,'Type'    )->value(C0,R1,'Colliding Beams')
        ->label(C1,R0,'Solenoid')->value(C1,R1,'4597')
        ->label(C2,R0,'#L1Acc'  )->value(C2,R1,'5184775')
        ->label(C3,R0,'#Evts'   )->value(C3,R1,'1243007')
        ->label(C4,R0,'QA'      )->value(C4,R1,'')

        ->label(C0,R2,'Config' )
        ->label(C1,R2,'TrigMask:'   )->value(C2,R2,'33ffffff')
        ->label(C3,R2,'CrateMask:'  )->value(C4,R2,'4fff77ff')
        ->label(C5,R2,'ConfigKey:'  )->value(C6,R2,'206d')
        ->label(C1,R3,'OprMode:'    )->value(C2,R3,'YES')
        ->label(C4,R3,'ConfigAlias:')->value(C5,R3,'PHYSICS_TEST')

        ->label(C0,R4,'Beam Energy<br>and<br>Current')
        ->label(C1,R4,'LER:'        )->value(C2,R4,'3.1127')
        ->label(C3,R4,'HER:'        )->value(C4,R4,'8.0517')
        ->label(C5,R4,'CM:'         )->value(C6,R4,'10.0125')
        ->label(C1,R5,'LER:'        )->value(C2,R5,'1980 mA')
        ->label(C3,R5,'HER:'        )->value(C4,R5,'1440 mA')
        ->label(C1,R6,'LER Pattern:')->value(C2,R6,'by2_t34_3gap')
        ->label(C4,R6,'HER Pattern:')->value(C5,R6,'by2_t34_3gap')

        ->label(C0,R7,'PEP (x.99)')
        ->label(C1,R7,'InitLum:'  )->value(C2,R7,'6650.0 (6583.5)')
        ->label(C4,R7,'Lumin:'    )->value(C5,R7,'11286.0 (11173.1) nb<SUP>-1</SUP>')

        ->label(C0,R8,'L3 Vars')
        ->label(C1,R8,'Deliv Lum:')->value(C2,   R8,'11280.5 nb<SUP>-1</SUP>')
        ->label(C4,R8,'Rec Lum:'  )->value(C5,   R8,'11104.0 nb<SUP>-1</SUP>')
        ->label(C1,R9,'Time:'     )->value(C2,   R9,'enabled = 1668.1 s')
                                   ->value(C3+30,R9,'paused = 2.9 s')
                                   ->value(C4+50,R9,'elapsed = 1671 s')
                                   ->value(C5+70,R9,'dead = 1.56 %')
        ->label(C1,RA,'# CycTrig:' )->value(C2,   RA,'1642')
        ->label(C4,RA,'# LumTrig:' )->value(C5,   RA,'100919')
        ->label(C1,RB,'# HadATrig:')->value(C2,   RB,'79785 (A/L = 0.791 &plusmn; 0.004)')
        ->label(C4,RB,'# HadBTrig:')->value(C5,   RB,'42822 (B/L = 0.424 &plusmn; 0.002)')
        ->label(C1,RC,'Trigger Rates:')->label(C2,RC,'&lt;L1&gt;')->value(C2+35,RC,'= 3108.2 Hz')
                                       ->label(C4,RC,'&lt;L3&gt;')->value(C4+35,RC,'= 745.2 Hz')

        ->label(C0,RD,'On?')
        ->label(C1,RD,'Deliv Lum:')->value(C2,RD,'Svt,Dch,Drc,Emc,Ifr,Emt,Glt,L3,Dcz')

        ->label(C0,RE,'Why Ended?' )->value(C1,RE,'BEAMLOSS')
        ->label(C2,RE,'Total Events:'  )->value(C3,RE,'16484358')
        ->label(C4,RE,'Luminosity:')->value(C5,RE,'124688 x .99 = 123441.12');

    echo $con->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>