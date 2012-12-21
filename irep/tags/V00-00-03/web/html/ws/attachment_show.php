<?php

/**
 * This service will return the specified attachment or its preview.
 * 
 * Parameters:
 * 
 *   <scope> <id> [<preview>]
 * 
 * Where:
 * 
 *   <scope> := { <equipment> | <equipment_model> | <model> }
 *   <id>    : an equipment ID for <equipment_model>
 *             an attachment ID for <equipment> and <model>
 */

require_once 'dataportal/dataportal.inc.php' ;

/*
 * Read a PDF file, extract the first page and translate it into
 * a PNG thumbnail whose geometry won't exceed specified limits.
 * 
 * NOTE: The algorithm uses Imagemagic, not GD!
 */
function pdf2png ($document, $maxwidth, $maxheight) {

    // TODO: Doing it via an intermediate file because the current
    //       implementation of Imagemagic's readImageBlob() simply
    //	     doesn't work. Revisit this issues in the later versions
    //       of the library.

    $tmpfname = tempnam( "/tmp/irep", "" ).".pdf" ;

    $handle = fopen($tmpfname, "w" );
    fwrite($handle, $document) ;
    fclose($handle) ;

    $im = new Imagick() ;
    $im->readImage($tmpfname) ;

    unlink($tmpfname) ;
    
    // Reverse iteration to the very first image in a sequence
    //
    while ($im->hasPreviousImage()) $im->previousImage() ;

    // Rescale the image and translate it into JPEG.
    //
    $bestfit = true ;
    $im->thumbnailImage($maxwidth, $maxheight, $bestfit) ;
    $im->setImageFormat('png') ;

    // Note that we're converting the real image into a GD object
    // in order for this function to be compatible with the rest
    // of the algorithm.

    $thumb = imagecreatefromstring($im->getimageblob()) ;
    return $thumb ;
}

/*
 * Save back to the database by starting PHP buffering and capturing its contents
 * before flushing the buffer back to the caller. Apparentlly this is the only way to get
 * the image into a binary string in the GD library.
*/
function display_and_cache ($thumb, $attachment=null) {

    header("Content-type: image/png") ;

    // TODO: Enable caching after done with debugging the code
    //
    //  header("Cache-Control: max-age=259200, must-revalidate") ;  // keep in browser's cache for 30 days
    //
    header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
    header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past
    
    ob_clean();
    ob_start() ;            // begin buffering the output

    imagepng($thumb) ;      // dump the image into PHP buffer; no output yet for the caller
    imagedestroy($thumb) ;

    $stringdata = ob_get_contents() ;   // read from buffer

    ob_end_flush() ;    // flush & delete buffer. NOTE: do not use ob_end_clean() because
                        // no output will be deliverd to the caller.        

    if ($attachment) $attachment->update_document_preview($stringdata) ;
}

\DataPortal\Service::run_handler ('GET', function ($SVC) {

    $scope   = strtolower($SVC->required_str ('scope')) ;
    $id      =            $SVC->required_int ('id') ;
    $preview =            $SVC->optional_flag('preview') ;

    $attachment = null ;
    switch ($scope) {
        case 'equipment':
            $attachment = $SVC->irep()->find_equipment_attachment_by_id($id) ;
            if (is_null($attachment)) $SVC->abort("no attachment found for ID: {$id}", 400) ;
            break ;
        case 'equipment_model':
            $equipment = $SVC->irep()->find_equipment_by_id($id) ;
            if (is_null($equipment)) $SVC->abort("no equipment found for ID: {$id}", 400) ;
            $manufacturer = $SVC->irep()->find_manufacturer_by_name($equipment->manufacturer()) ;
            if (!is_null($manufacturer)) {
                $model = $manufacturer->find_model_by_name($equipment->model()) ;
                if (!is_null($model)) {
                    $attachment = $model->default_attachment() ;
                }
            }
            if (is_null($attachment)) {
                $thumb = imagecreatefrompng('../img/NoImageAvailable150.png') ;
                imagealphablending($thumb, false) ;
                imagesavealpha($thumb, true) ;
                display_and_cache($thumb) ;
                $SVC->finish() ;
            }
            break ;
        case 'model':
            $attachment = $SVC->irep()->find_model_attachment_by_id($id) ;
            if (is_null($attachment)) $SVC->abort("no model attachment found for ID: {$id}", 400) ;
            break ;
        default:
            $SVC->abort("unsupported scope: '{$scope}'", 400) ;
    }

    $type = $attachment->document_type() ;
    if ($preview) {

        /* Check if the thumbnail already exists in the database, If not
         * then generate and save it there before displaying.
         */
        $document_preview = $attachment->document_preview() ;            
        if ($document_preview != '') {

            $thumb = imagecreatefromstring ($document_preview) ;
            imagealphablending($thumb, false) ;
            imagesavealpha($thumb, true) ;
            display_and_cache($thumb) ;
            $SVC->finish() ;

        } else {

            $document = $attachment->document( );

            // Preview to be generated and cached in the database for all suported
            // data formats for which a conversion onto PDF is available and make a sense.

            $thumb = null ;

            $imagetype = 'image/' ;
            if (substr($type, 0, strlen($imagetype)) == $imagetype) {

                // External pre-processing for TIIF & BMP. These formats aren't supported by the GD
                // library, so we need to turn them into something else before generating a thumbnail.

                $subtype = substr($type, strlen($imagetype)) ;
                if (($subtype == 'tiff') || ($subtype ==  'bmp')) {

                    $filename = tempnam('/tmp/irep', '') ;
                    $filename_subtype = $filename.'.'.$subtype ;
                    $filename_png = $filename.'.png' ;

                    $file_subtype = fopen($filename_subtype, 'wb') ;
                    fwrite($file_subtype, $document) ;
                    fclose($file_subtype) ;

                    exec("convert {$filename_subtype} {$filename_png}") ;

                    $file_png = fopen($filename_png, 'rb') ;
                    $document = fread($file_png, filesize($filename_png)) ;
                    fclose($file_png) ;
                    
                    unlink($filename) ;
                    unlink($filename_subtype) ;
                    unlink($filename_png) ;
                }

                // Generate

                $original = imagecreatefromstring($document) ;
                $oldwidth = imagesx($original) ;
                $oldheight = imagesy($original) ;

                // Do not rescale the image if its height doesn't exceed the limit. Just retranslate
                // it into PNG.

                $width = $oldwidth ;
                $height = $oldheight ;
                if ($height > 120) {
                    $width = round($oldwidth * (120.0 / $oldheight)) ;
                    $height = 120 ;
                }
                $thumb = imagecreatetruecolor($width, $height) ;
                imagealphablending($thumb, false) ;
                imagesavealpha($thumb, true) ;
                imagecopyresampled($thumb, $original, 0, 0, 0, 0, $width, $height, $oldwidth, $oldheight) ;

            } else if (($type == 'application/pdf') || ($type == 'application/x-pdf')) {
                $thumb = pdf2png($document, 800, 640) ;
            } else {
                $thumb = imagecreatefrompng('../img/NoImageAvailable150.png') ;
                imagealphablending($thumb, false) ;
                imagesavealpha($thumb, true) ;
            }
            display_and_cache($thumb, $attachment) ;
        }

    } else {

        $document = $attachment->document() ;

        header("Content-type: {$type}");

        // TODO: Enable caching after done with debugging the code
        //
        //  header("Cache-Control: max-age=259200, must-revalidate") ;  // keep in browser's cache for 30 days
        //
        header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
        header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

        ob_clean();
        ob_start() ;            // begin buffering the output

        echo ($document) ;

        ob_end_flush() ;    // flush & delete buffer. NOTE: do not use ob_end_clean() because
                            // no output will be deliverd to the caller.        
    }
    $SVC->finish() ;
}) ;

?>
