define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk' ,
    'portal/ELog_Utils'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk ,
    ELog_Utils) {

    cssloader.load('../portal/css/ELog_Post.css') ;

    /**
     * The application for posting messages to the experimental e-Log
     *
     * @returns {ELog_Post}
     */
    function ELog_Post (experiment, access_list, post_onsuccess) {

        var that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            this.init() ;
            if (!this.experiment.is_facility) {
                this.load_shifts() ;
            }
            this.load_tags() ;
        } ;

        this.on_deactivate = function() {
            this.init() ;
        } ;

        this.on_update = function () {
            if (this.active) {
                this.init() ;
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.experiment     = experiment ;
        this.access_list    = access_list ;
        this.post_onsuccess = post_onsuccess || null ;

        // --------------------
        // Own data and methods
        // --------------------

        this.is_initialized = false ;

        this.wa = null ;

        if (!this.experiment.is_facility) {
            this.runnum = null ;
            this.shift  = null ;
        }

        this.form              = null ;
        this.form_scope        = null ;
        this.form_run_num      = null ;
        this.form_shift_id     = null ;
        this.form_message_text = null ;
        this.form_attachments  = null ;
        this.form_tags         = null ;

        this.submit_button = null ;
        this.reset_button  = null ;

        this.max_file_size = 25000000 ;

        this.init = function () {

            if (this.is_initialized) return ;
            this.is_initialized = true ;

            this.container.html('<div id="elog-post"></div>') ;
            this.wa = this.container.find('div#elog-post') ;

            if (!this.access_list.elog.post_messages) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var html =
'<div style="float:left;">' ;
            if (!this.experiment.is_facility) html +=
'  <div style="float:left; font-weight:bold; padding-top:5px;">Run number:</div>' +
'  <div style="float:left; margin-left:5px;">' +
'    <input type="text" id="runnum" value="" size=4 />' +
'  </div>' +
'  <div class="info" style="float:left; margin-left:5px; padding-top:5px;">(optional)</div>' +
'  <div style="float:left; font-weight:bold; margin-left:20px; padding-top:5px;">Shift:</div>' +
'  <div style="float:left; margin-left:5px;">' +
'    <select id="shift" ></select>' +
'  </div>' +
'  <div class="info" style="float:left; margin-left:5px; padding-top:5px;">(optional)</div>' +
'  <div style="clear:both;"></div>' ;
            html +=
'  <form id="form" enctype="multipart/form-data" action="../logbook/ws/message_new.php" method="post">' +
'    <input type="hidden" name="id" value="'+this.experiment.id+'" />' +
'    <input type="hidden" name="scope" value="" />' +
'    <input type="hidden" name="run_num" value="" />' +
'    <input type="hidden" name="shift_id" value="" />' +
'    <input type="hidden" name="MAX_FILE_SIZE" value="'+this.max_file_size+'" />' +
'    <input type="hidden" name="num_tags" value="'+ELog_Utils.max_num_tags+'" />' +
'    <input type="hidden" name="onsuccess" value="" />' +
'    <input type="hidden" name="relevance_time" value="" />' +
'    <input type="hidden" name="post2instrument" value="" />' +
'    <input type="hidden" name="post2sds" value="" />' +
'    <textarea name="message_text" rows="12" cols="64" style="padding:4px; margin-top:5px;" title="The first line of the message body will be used as its subject"></textarea>' +
'    <div style="margin-top: 10px;">' +
'      <div style="float:left;">' +
'        <div style="font-weight:bold;">Author:</div>' +
'        <input type="text" name="author_account" value="'+this.access_list.user.uid+'" size=32 style="padding:2px; margin-top:5px; width:100%;" />' +
'        <div style="margin-top:20px;"> ' +
'          <div style="font-weight:bold;">Tags:</div>' +
'          <div style="margin-top:5px;" id="tags"></div>' +
'        </div>' +
'      </div>' +
'      <div style="float:left; margin-left:30px;"> ' +
'        <div style="font-weight:bold;">Attachments:</div>' +
'        <div id="attachments" style="margin-top:5px;">' +
'          <div>' +
'            <input type="file"   name="file2attach_0 />' +
'            <input type="hidden" name="file2attach_0" value="" />' +
'          </div>' +
'        </div>' +
'      </div>' +
'      <div style="clear:both;"></div>' +
'    </div>' +
'  </form>' +
'</div>' +
'<div style="float:left; margin-left:20px; padding-top:30px;">' +
'  <div id="buttons" >' +
'    <button class="control-button" name="submit" >Post</button>' +
'    <button class="control-button" name="reset" style="margin-left:5px;">Reset form</button>' +
'  </div>' + (this.experiment.is_facility ? '' :
'  <div id="extra_options" >' +
'    <input name="post2instrument" type="checkbox" title="also post a copy of the message to the instrument e-log" />' +
'    <span>Post a copy to the instrument e-Log</span>' +
'    <br>' +
'    <input name="post2sds" type="checkbox" title="also post a copy of the message to the Sample Delivery System e-log" />' +
'    <span>Post a copy to the Sample Delivery System e-Log</span>' +
'  </div>') +
'</div>' +
'<div style="clear:both;"></div>' ;
            this.wa.html(html) ;

            if (!this.experiment.is_facility) {
                this.runnum = this.wa.find('input#runnum') ;
                this.shift  = this.wa.find('select#shift') ;
            }
            this.form = this.wa.find('form#form') ;

            this.form_scope          = this.form.find('input[name="scope"]') ;
            this.form_run_num        = this.form.find('input[name="run_num"]') ;
            this.form_shift_id       = this.form.find('input[name="shift_id"]') ;
            this.post2instrument     = this.form.find('input[name="post2instrument"]') ;
            this.post2sds            = this.form.find('input[name="post2sds"]') ;
            this.form_message_text   = this.form.find('textarea[name="message_text"]') ;
            this.form_author_account = this.form.find('input[name="uthor_account"]') ;
            this.form_tags           = this.form.find('#tags') ;
            this.form_attachments    = this.form.find('#attachments') ;

            this.form_attachments.find('input:file[name="file2attach_0"]').change(function () {
                that.post_add_attachment() ;
            }) ;

            this.submit_button = this.wa.find('button[name="submit"]').button() ;
            this.submit_button.click(function () {

                var urlbase = window.location.href ;
                var idx = urlbase.indexOf('?') ;
                if( idx > 0 ) urlbase = urlbase.substring(0, idx) ;

                if (that.form_message_text.val() === '') {
                    Fwk.report_error('Can not post the empty message. Please put some text into the message box.') ;
                    return ;
                }

                that.form_scope.val('experiment') ;

                if (!that.experiment.is_facility) {
                    var runnum = 0 ;
                    var str = that.runnum.val() ;
                    if (str) {
                        runnum = parseInt(str) ;
                        if (!runnum) {
                            Fwk.report_error('Failed to parse the run number. Please, correct or clean the field.') ;
                            return ;
                        }
                        that.form_scope.val('run') ;
                        that.form_run_num.val(runnum) ;
                    } else {
                        that.form_run_num.val('') ;
                    }

                    var shift_id = parseInt(that.shift.val()) ;
                    if (shift_id) {
                        that.form_scope.val('shift') ;
                        that.form_shift_id.val(shift_id) ;
                    } else {
                        that.form_shift_id.val('') ;
                    }

                    if (runnum && shift_id) {
                        Fwk.report_error('Run number and shift are mutually exclusive options. Please, chose either one or another.') ;
                        return ;
                    }

                    that.post2instrument.val(that.wa.find('#extra_options').children('input[name="post2instrument"]').attr('checked') ? 1 : 0) ;
                    that.post2sds       .val(that.wa.find('#extra_options').children('input[name="post2sds"]')       .attr('checked') ? 1 : 0) ;
                }

                /* Submit the new message using the JQuery AJAX POST plug-in,
                 * which also allow uploading files w/o reloading the current page.
                 *
                 * NOTE: We aren't refreshing the list of messages because we're relying on
                 *       the live display.
                 */
                that.form.ajaxSubmit ({
                    success: function(data) {
                        if (data.Status !== 'success') {
                            Fwk.report_error(data.Message) ;
                            return ;
                        }
                        that.post_reset() ;

                        // If the parent provided a call back then tell the parent
                        // that we have a new message.
                        //
                        if (that.post_onsuccess) that.post_onsuccess() ;
                    },
                    error: function () {
                        Fwk.report_error('The request can not go through due a failure to contact the server.') ;
                    } ,
                    dataType: 'json'
                }) ;
            }) ;

            this.reset_button = this.wa.find('button[name="reset"]').button() ;
            this.reset_button.click(function () {
                that.post_reset() ;
            }) ;

            if (!this.experiment.is_facility) {
                this.load_shifts() ;
            }
            this.load_tags() ;
            this.post_reset() ;
        };
        this.simple_post4experiment = function (text2post) {
            this.post_reset() ;
            this.form_message_text.val(text2post) ;
        };

        this.post_reset = function () {

            if (!this.experiment.is_facility) {
                this.load_shifts() ;
            }
            this.load_tags() ;

            if (!this.experiment.is_facility) {
                this.runnum.val('') ;
                this.shift.val(0) ;
                this.post2instrument.val(0) ;
                this.post2sds.val(0) ;
            }
            this.form_scope.val('') ;
            this.form_message_text.val('') ;
            this.form_run_num.val('') ;
            this.form_author_account.val(this.access_list.user.uid) ;
            this.form_tags.find('.tag-name').val('') ;
            this.form_attachments.html (
'<div>' +
'  <input type="file"   name="file2attach_0" />' +
'  <input type="hidden" name="file2attach_0" value="" />' +
'</div>'
            ) ;
            this.form_attachments.find('input:file[name="file2attach_0"]').change(function () { that.post_add_attachment() ; }) ;
            if (!this.experiment.is_facility) {
                this.wa.find('#extra_options').children('input[name="post2instrument"]').removeAttr('checked') ;
                this.wa.find('#extra_options').children('input[name="post2sds"]')       .removeAttr('checked') ;
            }
        } ;

        this.post_add_attachment = function () {
            var num = this.form_attachments.find('div').size() ;
            this.form_attachments.append (
'<div>' +
' <input type="file"   name="file2attach_'+num+'" />' +
' <input type="hidden" name="file2attach_'+num+'" value="" />' +
'</div>'
            ) ;
            this.form_attachments.find('input:file[name="file2attach_'+num+'"]').change(function () { that.post_add_attachment() ; }) ;
        } ;

        if (!this.experiment.is_facility) {
            this.load_shifts = function () {
                ELog_Utils.load_shifts (
                    this.experiment.id ,
                    this.shift
                ) ;
            } ;
        }
        this.load_tags = function () {
            ELog_Utils.load_tags_and_authors (
                this.experiment.id ,
                this.form_tags
            ) ;
        } ;
    }
    Class.define_class (ELog_Post, FwkApplication, {}, {}) ;

    return ELog_Post ;
}) ;
