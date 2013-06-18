<?php

/**
 * This service will obtain and return extended status of files found
 * in the request.
 */
require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $input_files = json_decode($SVC->required_str('files')) ;
    if ( is_null ($input_files)) $SVC->abort("failed to translate parameter 'files' as a JSON object") ;
    if (!is_array($input_files)) $SVC->abort("expected array as parameter 'files'") ;

    // Group files by experiment and file type
    //
    $files_grouped = array () ;
    foreach ($input_files as $triplet) {

        if (!is_array($triplet))
            $SVC->abort("expected array as a file description triplet in parameter 'files'") ;

        if (count($triplet) != 3)
            $SVC->abort("exactly 3 elements expected in file description triplets: (experiment_id,file_type,file_name)") ;

        $exper_id = intval($triplet[0]) ;
        if (!$exper_id)
            $SVC->abort("invalid experiment identifier found in the request") ;

        if (!array_key_exists($exper_id, $files_grouped)) {

            $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
            if (is_null($experiment))
                $SVC->abort("unknown experiment identifier {$exper_id} found in the request") ;

            $files_grouped[$exper_id] = array (
                'experiment' => $experiment ,
                'files'      => array ()
            ) ;
        }

        $file_type = strtolower($triplet[1]) ;
        if (!$file_type) $SVC->abort("invalid file type '{$file_type}' found in the request") ;
        if (!array_key_exists($file_type, $files_grouped[$exper_id]['files'])) {
            $files_grouped[$exper_id]['files'][$file_type] = array () ;
        }
        array_push (
            $files_grouped[$exper_id]['files'][$file_type] ,
            $triplet[2]
        ) ;
    }

    // Process the request according to the way the files were groupped
    // above.
    //
    $files_extended = array () ;
    foreach ($files_grouped as $exper_id => $files_by_experiment) {

        $experiment = $files_by_experiment['experiment'] ;

        foreach ($files_by_experiment['files'] as $file_type => $files) {


            // Obtain all files of the given type for the specified instrumet & experiment
            // from the migration database.
            //
            $data_migration_files = array () ;
            foreach ($experiment->regdb_experiment()->data_migration_files($file_type) as $file) {
                $data_migration_files[$file->name()] = $file ;
            }

            // Contact iRODS File Manager and obtain all files of the given type
            // for the specified instrument & experiment.
            //
            $files_in_irods = array (
                'lustre-resc' => array () ,
                'hpss-resc'   => array ()
            ) ;
            $file_to_size = array() ;
            foreach ($SVC->irodsdb()->runs($experiment->instrument()->name(), $experiment->name(), $file_type) as $run) {
                foreach ($run->files as $file) {
                    if (!array_key_exists($file->resource, $files_in_irods))
                        $SVC->abort("implementation error: unknown storage resource reported by iRODS: '{$file->resource}'") ;
                    array_push (
                        $files_in_irods[$file->resource] ,
                        $file->name
                    );
                    $file_to_size[$file->name] = intval($file->size) ;
                }
            }
            foreach ($files as $file_name) {

                // Check if this file status in the File Manager and migration database.
                // Compute status flags accordingly.
                //
                $flags = array() ;
                if (array_key_exists      ($file_name, $data_migration_files))          array_push($flags,'IN_MIGRATION_DATABASE') ;
                if (False !== array_search($file_name, $files_in_irods['lustre-resc'])) array_push($flags,'DISK') ;
                if (False !== array_search($file_name, $files_in_irods['hpss-resc'  ])) array_push($flags,'HPSS') ;

                array_push (
                    $files_extended ,
                    array (
                        array ($exper_id,$file_type,$file_name) ,
                        $flags ,
                        array_key_exists($file_name, $file_to_size) ? $file_to_size[$file_name] : 0
                    )
                ) ;
            }
        }
    }
    $SVC->finish(array ('files_extended' => $files_extended)) ;
}) ;

?>
