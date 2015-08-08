define ([] ,

function () {

    var _DEFINITION_CTIME =
        'The parameter (if set) will tell the File Manager to ignore actual ' +
        'file creation timestamps of files which are older ' +
        'than the specified value of the parameter and to use the value of the ' +
        'parameter when calculating the expiration dates of the files. Note that ' +
        'this will not affect the real timestamp of the file in a file system ' +
        'neither in the Experiment Portal Web interface.' ;
    var _DOCUMENT_CTIME =
        'An optional override for file creation timestamps of files \n' +
        'which are older than the value of the parameter. It could be \n' +
        'used to adjust the expiration dates of the files. The parameter \n' +
        'will not affect the real timestamp of the file in a file system \n' +
        'neither in the Experiment Portal Web interface. \n' +
        'Value must adhere to this format: YYYY-MM-DD. ' ;

    var _DEFINITION_RETENTION =
        'The maximum duration (retention) of stay for ' +
        'a file in this type of storage. Files will be considered <b>expired</b> ' +
        'on a day which is the specified (or default) number of months after ' +
        'they (the files) are created (unless the CTIME override is used). ' +
        'An empty value of the parameter means that no specific limit is imposed ' +
        'on a duration of time the files can be kept in this type of storage.' ;
    var _DOCUMENT_RETENTION =
        'The maximum duration (retention) of stay for files of this \n' +
        'storage class. Files will be considered as expired after \n' +
        'the specified number of months after their actual creation \n' +
        'or CTIME overriden time (is used). \n' +
        'An empty value of the parameter means that no specific limit is imposed ' +
        'on a duration of time the files can be kept in this type of storage.' ;

    var _DEFINITION_QUOTA =
        'The storage quota allocated for each experiment. ' +
        'An empty value of the parameter means that no specific limit is imposed ' +
        'on the amount of data which can be kept in this type of storage.' ;
    var _DOCUMENT_QUOTA =
        'The storage quota allocated for an experiment. \n' +
        'An empty value of the parameter means that no specific limit is imposed \n' +
        'on the amount of data which can be kept in this type of storage.' ;

    var STORAGE_CLASS = [
        {   name: 'SHORT-TERM' ,
            parameters: [
                {   name:  'ctime' ,
                    title: 'CTIME override' ,
                    units: 'yyyy-mm-dd' ,
                    definition: _DEFINITION_CTIME ,
                    document:   _DOCUMENT_CTIME ,
                    if_not_set_then: 'actual file creation time' ,
                    input: {
                        size: '11' ,
                        value: '' ,
                        on_validate: function (str) {
                            return str ;
                        }
                    }
                } ,
                {   name:  'retention' ,
                    title: 'retention' ,
                    units: 'number of months' ,
                    definition: _DEFINITION_RETENTION ,
                    document:   _DOCUMENT_RETENTION ,
                    if_not_set_then: 'no limit' ,
                    input: {
                        size: '2' ,
                        value: '6' ,
                        on_validate: function (str) {
                            var val = parseInt(str) ;
                            return val < 1 ? 1 : val ;
                        }
                    }
                }
            ]
        } ,
        {   name: 'MEDIUM-TERM' ,
            parameters: [
                {   name:  'ctime' ,
                    title: 'CTIME override' ,
                    units: 'yyyy-mm-dd' ,
                    definition: _DEFINITION_CTIME ,
                    document:   _DOCUMENT_CTIME ,
                    if_not_set_then: 'actual file creation time' ,
                    input: {
                        size: '11' ,
                        value: '' ,
                        on_validate: function (str) {
                            return str ;
                        }
                    }
                } ,
                {   name: 'retention' ,
                    title: 'retention' ,
                    units: 'number of months' ,
                    definition: _DEFINITION_RETENTION ,
                    document:   _DOCUMENT_RETENTION ,
                    if_not_set_then: 'no limit' ,
                    input: {
                        size: '2' ,
                        value: '24' ,
                        on_validate: function (str) {
                            var val = parseInt(str) ;
                            return val < 1 ? 1 : val ;
                        }
                    }
                } ,
                {   name:  'quota' ,
                    title: 'quota' ,
                    units: 'GB' ,
                    definition: _DEFINITION_QUOTA ,
                    document:   _DOCUMENT_QUOTA ,
                    if_not_set_then: 'no limit' ,
                    input: {
                        size: '4' ,
                        value: '10000' ,
                        on_validate: function (str) {
                            var val = parseInt(str) ;
                            return val < 1000 ? 1000 : val ;
                        }
                    }
                }
            ]
        }
    ] ;
    function DOCUMENT_METHOD (str) {
        return 'data="'+str+'"' ;
    }
    return {
        DOCUMENT_METHOD: DOCUMENT_METHOD ,
        STORAGE_CLASS:   STORAGE_CLASS
    } ;
}) ;