<?php

require_once( 'logbook/logbook.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

/*
 * This script will process a request for listing shifts of an
 * experiment.
 */
if( isset( $_GET['id'] )) {
    if( 1 != sscanf( trim( $_GET['id'] ), "%d", $id ))
        die( "invalid format of the attachment identifier" );
} else {
    die( "no valid attachment identifier" );
}
$preview = isset( $_GET['preview'] );

/*
 * Read a PDF file, extract the first page and translate it into
 * a PNG thumbnail whose geometry won't exceed specified limits.
 * 
 * NOTE: The algorithm uses Imagemagic, not GD!
 */
function pdf2png( $document, $maxwidth, $maxheight ) {

    // TODO: Doing it via an intermediate file because the current
    //       implementation of Imagemagic's readImageBlob() simply
    //         doesn't work. Revisit this issues in the later versions
    //       of the library.

    $tmpfname = tempnam( "/tmp/logbook", "" ).".pdf";

    $handle = fopen($tmpfname, "w");
    fwrite($handle, $document);
    fclose($handle);

    $im = new Imagick();
    $im->readImage( $tmpfname );

    unlink($tmpfname);
    
    // Reverse iteration to the very first image in a sequence
    //
    while( $im->hasPreviousImage()) $im->previousImage();

    // Rescale the image and translate it into JPEG.
    //
    $bestfit = true;
    $im->thumbnailImage( $maxwidth, $maxheight, $bestfit );
    $im->setImageFormat( 'png');

    // Note that we're converting the real image into a GD object
    // in order for this function to be compatible with the rest
    // of the algorithm.

    $thumb = imagecreatefromstring( $im->getimageblob());
    return $thumb;
}

/*
 * Save back to the database by starting PHP buffering and capturing its contents
 * before flushing the buffer back to the caller. Apparentlly this is the only way to get
 * the image into a binary string in the GD library.
*/
function display_and_cache( $thumb, $attachment ) {

    header( "Content-type: image/png" );
    header( "Cache-Control: max-age=259200, must-revalidate" );    // keep in browser's cache for 30 days
    
    ob_start();    // begin buffering the output

    imagepng( $thumb );        // dump the image into PHP buffer; no output yet for the caller
    imagedestroy( $thumb );

    $stringdata = ob_get_contents();    // read from buffer

    ob_end_flush();    // flush & delete buffer. NOTE: do not use ob_end_clean() because
                       // no output will be deliverd to the caller.        

    $attachment->update_document_preview( $stringdata );
}

/* Proceed to the operation
 */
try {
    LogBook::instance()->begin();

    $attachment = LogBook::instance()->find_attachment_by_id( $id )  or die("no such attachment" );
    $experiment = $attachment->parent()->parent();
    $instrument = $experiment->instrument();

    /* Check for the authorization
     */
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to access any information about the experiment',
            '../index.php?action=select_experiment'.
                '&instr_id='.$instrument->id().
                '&instr_name='.$instrument->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name()));
        exit;
    }


    /* Proceed to the operation.
     */
    $type = $attachment->document_type();
    if( $preview ) {

        /* Check if the thumbnail already exists in the database, If not
         * then generate and save it there before displaying.
         */
        $document_preview = $attachment->document_preview();            
        if( $document_preview != '' ) {

            $thumb = imagecreatefromstring( $document_preview );

            header( "Content-type: image/png" );
            header( "Cache-Control: max-age=259200, must-revalidate" );    // keep in browser's cache for 30 days

            imagepng( $thumb );
            imagedestroy( $thumb );

        } else {

            $document = $attachment->document();

            // Preview to be generated and cached in the database for all suported
            // data formats for which a conversion onto PDF is available and make a sense.

            $thumb = null;

            $imagetype = 'image/';
            if( substr($type, 0, strlen( $imagetype )) == $imagetype ) {

                // External pre-processing for TIIF & BMP. These formats aren't supported by the GD
                // library, so we need to turn them into something else before generating a thumbnail.

                $subtype = substr( $type, strlen( $imagetype ));
                if(( $subtype == 'tiff' ) || ( $subtype ==  'bmp' )) {

                    $filename = tempnam( '/tmp', '' );
                    $filename_subtype = $filename.'.'.$subtype;
                    $filename_png = $filename.'.png';

                    $file_subtype = fopen( $filename_subtype, 'wb');
                    fwrite( $file_subtype, $document );
                    fclose( $file_subtype );

                    exec( "convert {$filename_subtype} {$filename_png}");

                    $file_png = fopen( $filename_png, 'rb');
                    $document = fread( $file_png, filesize( $filename_png ));
                    fclose( $file_png );
                    
                    unlink( $filename );
                    unlink( $filename_subtype );
                    unlink( $filename_png );
                }

                // Generate

                $original = imagecreatefromstring( $document );
                $oldwidth = imagesx( $original );
                $oldheight = imagesy( $original );

                // Do not rescale the image if its height doesn't exceed the limit. Just retranslate
                // it into PNG.

                $width = $oldwidth;
                $height = $oldheight;
                if( $height > 240 ) {
                    $width = round( $oldwidth * ( 240.0 / $oldheight ));
                    $height = 240;
                }
                $thumb = imagecreatetruecolor( $width, $height );
                imagecopyresampled($thumb, $original, 0, 0, 0, 0, $width, $height, $oldwidth, $oldheight );

            } else if(( $type == 'application/pdf' ) || ( $type == 'application/x-pdf' )) {
                $thumb = pdf2png( $document, 800, 640 );
            } else {
                $thumb = imagecreatefrompng( 'images/NoPreview.png' );
            }
            display_and_cache( $thumb, $attachment );
        }

    } else {

        $document = $attachment->document();

        header( "Content-type: {$type}" );
        header( "Cache-Control: max-age=259200, must-revalidate" );    // keep in browser's cache for 30 days

        echo( $document );
    }
    LogBook::instance()->commit();

} catch( LogBookException $e ) { print $e->toHtml(); }

?>