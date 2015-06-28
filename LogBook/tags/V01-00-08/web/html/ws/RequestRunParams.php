<?php

/**
 * Return the parameters and attributes of the run.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;

function extract_params4run ($instr_name, $run) {

    /* Get values for existing run parameters
     */
    $return_dict = true ;
    $values = $run->values('', $return_dict);


    /* Normal processing for common and instrument-specific parameters
     */
    $used_names = array() ; // Remember names of parameters displayed
                            // in common and instrument-specific sections.
                            // We're going to use this information later to put
                            // remaining parameters into the footer.
    $params = array() ;

    foreach (array('HEADER', $instr_name) as $area) {

        foreach (LogBook\LogBookUtils::$sections[$area] as $s) {

            $section = array (
                'section' => $s['SECTION'] ,
                'title'   => $s['TITLE'] ,
                'params'  => array()
            ) ;
            foreach ($s['PARAMS'] as $p) {

                $name  = $p['name'] ;
                $value = array_key_exists($name, $values) ? $values[$name]->value() : '&lt; no data &gt;' ;
                $descr = $p['descr'] ;

                if (array_key_exists($name, $used_names)) continue;

                array_push (
                    $section['params'] ,
                    array (
                        'name'  => $name ,
                        'value' => $value ,
                        'descr' => $descr)) ;

                $used_names[$name] = True ;
            }
            array_push($params, $section) ;
        }
    }

    /* Special processing for experiment-specific parameters  not found
     * in the dictionary.
     */
    $section = array (
        'section' => 'FOOTER' ,
        'title'   => 'Additional Parameters' ,
        'params'  => array()
    ) ;
    foreach ($values as $p) {

        $name  = $p->name() ;
        $value = $p->value() ;

        $descr = $name ;

        $param = $run->parent()->find_run_param_by_name($name) ;
        if ($param) {
            $param_descr = $param->description() ;
            if ($param_descr !== '') $descr = $param_descr ;
        }
                        
        if (array_key_exists($name, $used_names)) continue;

        array_push (
            $section['params'] ,
            array (
                'name'  => $name ,
                'value' => $value ,
                'descr' => $descr)) ;

        $used_names[$name] = True ;
    }
    array_push($params, $section) ;

    /* Display per-run attributes in a separate section
     */
    foreach ($run->attr_classes() as $class_name) {

        $title = array_key_exists($class_name, LogBook\LogBookUtils::$attribute_sections) ? LogBook\LogBookUtils::$attribute_sections[$class_name] : $class_name ;

        $section = array (
            'section' => $class_name ,
            'title'   => $title ,
            'params'  => array()
        ) ;
        foreach ($run->attributes($class_name) as $attr) {
            array_push (
                $section['params'] ,
                array (
                    'name'  => $attr->name() ,
                    'value' => $attr->val() ,
                    'descr' => $attr->description())) ;
        }
        array_push($params, $section) ;
    }
    return $params ;
}

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $run_id = $SVC->required_int('run_id') ;

    $run        = $SVC->logbook()->find_run_by_id($run_id) or $SVC->abort('no such run') ;
    $experiment = $run->parent() ;

    if (!$SVC->logbookauth()->canRead($experiment->id()))
        $SVC->abort('You are not authorized to access any information about the experiment') ;

    $SVC->finish (array (
        'params' => \extract_params4run ($experiment->instrument()->name(), $run))) ;
}) ;

?>
