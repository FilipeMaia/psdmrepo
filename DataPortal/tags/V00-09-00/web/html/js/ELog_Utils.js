define ([] ,

function () {

    /* 
     * The namespace for various utility functions and parameters of e-Log.
     */
    var ELog_Utils = new function () {

        var that = this ;

        this.max_num_tags  = 3 ;

        /**
         * Load all known tags for the specified experiment.
         * 
         * The result can be teported in three ways:
         * - by setting a layout of the specified 'target4tags' container (a JSON object is expected)
         * - by calling the 'on_success()' function and passing a list of tags as a parameter
         * - or by both methods
         * 
         * Errors are reported either to the standard reporting channel of Fwk
         * or (if provided) via the failure handler function 'on_failure()'.
         */
        this.load_tags_and_authors = function (exper_id, target4tags, on_success, on_failure) {
            Fwk.web_service_GET (
                '../logbook/ws/RequestUsedTagsAndAuthors.php' ,
                {id: exper_id} ,
                function (data) {

                    if (target4tags) {

                        var select_tag_html = '<option></option>' ;
                        for (var i in data.Tags) select_tag_html += '<option>'+data.Tags[i]+'</option>' ;
                        var html = '' ;
                        for (var i = 0; i < that.max_num_tags; i++) html +=
'<div style="width: 100%;">' +
'  <select id="library-'+i+'" name="'+i+'" >'+select_tag_html+'</select>'+
'  <input type="text"   class="tag-name" id="tag-name-'+i+'"  name="tag_name_'+i+'"  value="" size=16 title="type new tag here or select a known one from the left" />' +
'  <input type="hidden"                  id="tag-value-'+i+'" name="tag_value_'+i+'" value="" />' +
'</div>' ;
                        target4tags.html(html) ;

                        for (var i = 0; i < that.max_num_tags; i++) {
                            target4tags.find('#library-'+i).change(function () {
                                var idx = this.name ;
                                var tag = $(this).val() ;
                                target4tags.find('#tag-name-'+idx    ).val(tag) ;
                                target4tags.find('#library-'+idx).attr('selectedIndex', 0) ;
                            }) ;
                        }
                    }
                    if (on_success) { on_success(data.Tags, data.Authors) ; }
                } ,
                function (msg) {
                    if (on_failure) on_failure(msg) ;
                    else            Fwk.report_error(msg) ; 
                }
            ) ;
        } ;


        this.load_shifts = function (exper_id, target4shifts, on_success, on_failure) {
            Fwk.web_service_GET (
                '../logbook/ws/RequestShifts.php' ,
                {id: exper_id} ,
                function (data) {

                    if (target4shifts) {
                        var html = '<option value="0" ></option>' ;
                        for (var i in data.ResultSet.Result) {
                            var shift = data.ResultSet.Result[i] ;
                            html += '<option value="'+shift.id+'" >'+shift.begin_time+'</option>' ;
                        }
                        target4shifts.html(html) ;

                    }
                    if (on_success) { on_success(data.ResultSet.Result) ; }
                } ,
                function (msg) {
                    if (on_failure) on_failure(msg) ;
                    else            Fwk.report_error(msg) ; 
                }
            ) ;
        } ;
    } ;

    return ELog_Utils ;
}) ;