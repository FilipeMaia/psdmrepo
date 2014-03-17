function ELog_RunBody (parent, message) {

    var that = this ;
   
    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    StackRowBody.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    this.parent = parent ;
    this.experiment = parent.experiment ;
    this.access_list = parent.access_list ;
    this.message = message ;

    // ----------------------------
    // Static variables & functions
    // ----------------------------

    this.run_url = function () {
        var idx = window.location.href.indexOf('?') ;
        var url = (idx < 0 ? window.location.href : window.location.href.substr(0, idx))+'?exper_id='+this.experiment.id+'&app=elog:search&params=run:'+message.run_num;
        var html = '<a href="'+url+'" target="_blank" title="Click to open in a separate tab, or cut and paste to incorporate into another document as a link."><img src="../portal/img/link.png"></img></a>' ;
        return html ;
    }
    // ------------------------------------------------
    // Override event handler defined in thw base class
    // ------------------------------------------------

    this.is_rendered = false ;

    this.m_cont = null ;
    this.ctrl_cont = null ;
    this.message_cont = null ;
    this.dialogs_cont = null ;
    this.form_cont = null ;
    this.form_attachments_cont = null ;

    this.render = function () {

        if (this.is_rendered) return ;
        this.is_rendered = true ;

        var html =
'<div class="m-cont">' ;

        //////////////////// CONTROL BUTTONS ////////////////
        
        html +=
'  <div class="ctrl">' +
'    <div class="url-cont">'+this.run_url()+'</div>' +
'    <div class="button-cont"><button class="control-button" name="print"  title="print this run info">P</button></div>' +
'    <div class="button-cont"><button class="control-button" name="reply"  title="post a message which will be associated with this run"><b>&crarr;</b></button></div>' +
'    <div class="button-cont-last"></div>' +
'  </div>' ;

        //////////////////// PARAMETERS AND ATTRIBUTES OF THIS RUN ////////////////

        html +=
'  <div class="message">Loading...</div>' +
'  <div class="dialogs"></div>' ;

        html +=
'</div>' ;

        //////////////////// RENDER AND SETUP HANDLERS ////////////////
        
        this.container.html(html) ;
        
        this.m_cont = this.container.children('.m-cont') ;

        this.ctrl_cont = this.m_cont.children('.ctrl') ;
        this.ctrl_cont.children('.button-cont').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'reply': that.reply() ; break ;
                default:
                    console.log('ELog_RunBody.render(): this operation is not supported: '+op) ; break ;
            }
        }) ;

        this.message_cont = this.m_cont.children('.message') ;
        this.load_parameters() ;

        this.dialogs_cont = this.m_cont.children('.dialogs') ;
    } ;

    this.parameters_view = null ;

    this.load_parameters = function () {
        Fwk.web_service_GET (
            '../logbook/ws/RequestRunParams.php' ,
            {run_id: this.message.run_id} ,
            function (data) {
                that.parameters_view = new StackOfRows() ;
                for (var i in data.params) {
                    var section = data.params[i] ;
                    var section_body_html =
'<table><thead>' ;
                    for (var num = section.params.length, j = 0; j < num; j++) {
                        var extra_class = j === num - 1 ? 'table_cell_bottom' : '' ;
                        var param = section.params[j];
                        section_body_html +=
'  <tr>' +
'    <td class="table_cell table_cell_left  '+extra_class+'" style="font-weight:normal; font-style:italic;" >'+param.descr+'</td>' +
'    <td class="table_cell                  '+extra_class+'" style="color:maroon"                           >'+param.value+'</td>' +
'    <td class="table_cell table_cell_right '+extra_class+'"                                                >'+param.name +'</td>' +
'  </tr>' ;
                    }
                    section_body_html +=
'</thead></table>' ;
                    that.parameters_view.add_row({
                        title: '<b>'+section.title+'</b>' ,
                        body:  section_body_html}) ;
                }
                that.parameters_view.display(that.message_cont) ;
            } ,
            function (msg) {
                Fwk.report_error(msg) ;
            });
    } ;
    
    this.reply = function () {
        this.ctrl_cont.css('display', 'none') ;
        this.message_cont.css('display', 'none') ;
        var html =
'<div style="color:maroon;">Compose message. Note the total limit of <b>25 MB</b> for attachments.</div>' +
'<div style="float:left; margin-right:20px;">' +
'  <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/NewFFEntry4portalJSON.php" method="post">' +
'    <input type="hidden" name="id" value="'+this.experiment.id+'" />' +
'    <input type="hidden" name="scope" value="run" />' +
'    <input type="hidden" name="message_id" value="" />' +
'    <input type="hidden" name="run_id" value="'+this.message.run_id+'" />' +
'    <input type="hidden" name="shift_id" value="" />' +
'    <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'    <input type="hidden" name="num_tags" value="0" />' +
'    <input type="hidden" name="onsuccess" value="" />' +
'    <input type="hidden" name="relevance_time" value="" />' +
'    <textarea name="message_text" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>' +
'    <div style="margin-top: 10px;">' +
'      <div style="float:left;">' + 
'        <div style="font-weight:bold;">Author:</div>' +
'        <input type="text" name="author_account" value="'+this.access_list.user.uid+'" size=32 style="padding:2px; margin-top:5px;" />' +
'      </div>' +
'      <div style="float:left; margin-left:20px;">' + 
'        <div style="font-weight:bold;">Attachments:</div>' +
'        <div class="attachments">' +
'          <div>' +
'            <input type="file" name="file2attach_0" />' +
'            <input type="hidden" name="file2attach_0" value=""/ >' +
'          </div>' +
'        </div>' +
'      </div>' +
'      <div style="clear:both;"></div>' +
'    </div>' +
'  </form>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the message">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.reply_post() ;   break ;
                case 'cancel': that.reply_cancel() ; break ;
                default:
                    console.log('ELog_RunBody.reply(): this operation is not supported: '+op) ; break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        this.form_attachments_cont = this.form_cont.find('.attachments') ;
        this.form_attachments_cont.find('input:file[name="file2attach_0"]').change(function () {
            that.add_attachment() ;
        }) ;
    } ;

    this.reply_post = function () {
        if (this.form_cont.find('textarea[name="message_text"]').val() === '') {
            Fwk.report_error('Can not post the empty message. Please put some text into the message box.') ;
            return ;
        }
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post the reply w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.Status != 'success') {
                    Fwk.report_error(data.Message) ; return ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                }

                // NOTE: we don't bother to do anything because the live message veiwer
                // is supposed to automatically refresh itself and pick up any new
                // messages posted here or anywhere.

                that.reply_cancel() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;
    this.add_attachment = function () {
        var num = this.form_attachments_cont.find('div').size() ;
        this.form_attachments_cont.append (
'<div>' +
' <input type="file"   name="file2attach_'+num+'" />' +
' <input type="hidden" name="file2attach_'+num+'" value="" />' +
'</div>'
        ) ;
        this.form_attachments_cont.find('input:file[name="file2attach_'+num+'"]').change(function () { that.add_attachment() ; }) ;
    } ;
    this.reply_cancel = function () {
        this.ctrl_cont.css('display', 'block') ;
        this.message_cont.css('display', 'block') ;
        this.dialogs_cont.html('') ;
    } ;
}
define_class (ELog_RunBody, StackRowBody, {}, {}) ;

function ELog_MessageBody (parent, message) {

    var that = this ;
   
    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    StackRowBody.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    this.parent = parent ;
    this.experiment = parent.experiment ;
    this.access_list = parent.access_list ;
    this.message = message ;

    // ----------------------------
    // Static variables & functions
    // ----------------------------

    this.message_url = function () {
        var idx = window.location.href.indexOf('?') ;
        var url = (idx < 0 ? window.location.href : window.location.href.substr(0, idx))+'?exper_id='+this.experiment.id+'&app=elog:search&params=message:'+this.message.id;
        var html = '<a href="'+url+'" target="_blank" title="Click to open in a separate tab, or cut and paste to incorporate into another document as a link."><img src="../portal/img/link.png"></img></a>' ;
        return html ;
    }

    // ------------------------------------------------
    // Override event handler defined in thw base class
    // ------------------------------------------------

    this.is_rendered = false ;

    this.m_cont = null ;
    this.ctrl_cont = null ;
    this.message_cont = null ;
    this.attachments_cont = null ;
    this.tags_cont = null ;
    this.dialogs_cont = null ;
    this.form_cont = null ;
    this.form_attachments_cont = null ;

    this.children_viewer = null ;

    this.render = function () {

        if (this.is_rendered) return ;
        this.is_rendered = true ;

        var id = this.message.id ;

        var html =
'<div class="m-cont">' ;


        //////////////////// CONTROL BUTTONS ////////////////
        
        html +=
'  <div class="ctrl view">' +
'    <div class="url-cont">'+this.message_url()+'</div>' ;
        if (this.parent.deleted) {
            ;
        } else {
            if (this.message.deleted) html +=
'    <div class="button-cont"><button class="control-button" name="undelete"    title="undelete this message to allow othe operations" >undelete</button></div>' ;
            else {
                html +=
'    <div class="button-cont"><button class="control-button" name="print"        title="print this message and all its children if any">P</button></div>' +
'    <div class="button-cont"><button class="control-button" name="delete"       title="delete this message and all its children if any" style="color:red;">X</button></div>' ;
                if (!this.message.parent_id) html +=
'    <div class="button-cont"><button class="control-button" name="tags"         title="add more tags to the message">+ tags</button></div>' ;
             html +=
'    <div class="button-cont"><button class="control-button" name="attachments"  title="add more attachments to the message">+ attach</button></div>' ;
             if (this.access_list.elog.edit_messages) {
                    if (!this.message.parent_id) {
                        if (this.message.run_id) html +=
'    <div class="button-cont"><button class="control-button" name="dettach"     title="attach this message to a run" >- run</button></div>' ;
                        else html +=
'    <div class="button-cont"><button class="control-button" name="attach"      title="dettach this message from the run" >+ run</button></div>' ;
                    }
                    html +=
'    <div class="button-cont"><button class="control-button" name="edit"        title="edit the message text" >E</button></div>' ;
                }
                html +=
'    <div class="button-cont"><button class="control-button" name="reply"        title="reply to the message"><b>&crarr;</b></button></div>' ;
            }
        }
        html +=
'    <div class="button-cont-last"></div>' +
'  </div>' ;


        //////////////////// MESSAGE TEXT ////////////////

        html +=
'  <div class="message view"></div>' ;


        //////////////////// ATTACHMENTS ////////////////

        if (this.message.attachments_num) {
            html +=
'  <div class="attachments view">' ;
            var title = 'click to open the attachment in a separate tab' ;
            for (var i in this.message.attachments) {
                var attach = this.message.attachments[i] ;
                var url = '<a href="../logbook/attachments/'+attach.id+'/'+attach.description+'" target="_blank" title="'+title+'"><img src="../logbook/attachments/preview/'+attach.id+'"></a>' ;
                html +=
'    <div class="attach">' +
'      <div class="preview">' +
'        <a href="../logbook/attachments/'+attach.id+'/'+attach.description+'" target="_blank" title="'+title+'"><img src="../logbook/attachments/preview/'+attach.id+'"></a>' +
'      </div>' +
'      <div class="attrname">file:</div><div class="attrval">'+attach.description+'</div><div class="attrend"></div>' +
'      <div class="attrname">type:</div><div class="attrval">'+attach.type+'</div><div class="attrend"></div>' +
'      <div class="attrname">size:</div><div class="attrval">'+attach.size+'</div><div class="attrend"></div>' + 
'</div>' ;
            }
            html +=
'    <div class="attach-last"></div>' +
'  </div>' ;
        }


        //////////////////// TAGS ////////////////

        if (this.message.tags_num) {
            html +=
'  <div class="tags view"><b>keywords</b>: ' ;
            for (var i in this.message.tags) {
                var tag = this.message.tags[i] ;
                html += '<span>'+tag.tag+'&nbsp;</span>' ;
            }
            html +=
'  </div>' ;
        }


        //////////////////// CHILDREN ////////////////

        if (this.message.children_num) {
            html +=
'  <div class="children view"></div>' ;
        }

        html +=
'  <div class="dialogs edit"></div>' ;
'</div>' ;



        //////////////////// RENDER AND SETUP HANDLERS ////////////////
        
        this.container.html(html) ;
        
        this.m_cont = this.container.children('.m-cont') ;

        this.ctrl_cont = this.m_cont.children('.ctrl') ;
        this.ctrl_cont.children('.button-cont').children('.control-button') .button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'reply':       that.reply() ; break ;
                case 'edit':        that.edit_message() ; break ;
                case 'tags':        that.edit_tags() ; break ;
                case 'attachments': that.edit_attachments() ; break ;
                case 'attach':
                case 'dettach':     that.edit_run(op) ; break ;
                case 'delete':
                case 'undelete':    that.delete_message(op) ; break ;
                default:
                    console.log('ELog_MessageBody.render(): this operation is not supported: '+op) ; break ;
            }
        }) ;

        this.message_cont = this.m_cont.children('.message') ;
        this.message_cont.html(this.message.html1) ;

        this.attachments_cont = this.m_cont.children('.attachments') ;
        this.tags_cont = this.m_cont.children('.tags') ;

        if (this.message.children_num) {
            var hidden_header = true ;
            var instant_expand = true ;
            this.children_viewer = new ELog_MessageViewer (
                this ,
                this.m_cont.children('.children') ,
                {   hidden_header  : hidden_header ,
                    instant_expand : instant_expand ,
                    deleted        : this.message.deleted || this.parent.deleted
                }
            ) ;
            this.children_viewer.load(this.message.children) ;
        }

        this.dialogs_cont = this.m_cont.children('.dialogs') ;

        this.enable_view() ;
    } ;

    /*
     * Switcher between viewing and editing modes.
     * 
     * Note that there is a couple of wrapper functions defined after this one.
     * Those are meant to provide a more explicit interface for th erest of the code.
     * 
     * @param {boolean} view
     * @returns {undefined}
     */
    this.view_vs_edit = function (view) {
        if (view) {
            this.dialogs_cont.html('') ;
            this.m_cont.parent().removeClass('editing') ;
        } else {
            this.m_cont.parent().addClass('editing') ;
        }
        this.m_cont.children('.view').css('display', view ? 'block' : 'none') ;
        this.m_cont.children('.edit').css('display', view ? 'none'  : 'block') ;
    } ;

    this.enable_view = function () { this.view_vs_edit(true) ; }
    this.enable_edit = function () { this.view_vs_edit(false) ; }


    /**
     * Create a dialog for replying to the current message.
     * 
     * @returns {undefined}
     */
    this.reply = function () {

        this.enable_edit() ;

        var html =
'<div style="float:left; margin-right:20px; padding-top:5px;">' +
'  <div style="color:maroon;">Compose message. Note the total limit of <b>25 MB</b> for attachments.</div>' +
'  <div style="margin-top:20px; padding-left:10px;">' +
'    <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/NewFFEntry4portalJSON.php" method="post">' +
'      <input type="hidden" name="id" value="'+this.experiment.id+'" />' +
'      <input type="hidden" name="scope" value="message" />' +
'      <input type="hidden" name="message_id" value="'+this.message.id+'" />' +
'      <input type="hidden" name="run_id" value="" />' +
'      <input type="hidden" name="shift_id" value="" />' +
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'      <input type="hidden" name="num_tags" value="0" />' +
'      <input type="hidden" name="onsuccess" value="" />' +
'      <input type="hidden" name="relevance_time" value="" />' +
'      <textarea name="message_text" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>' +
'      <div style="margin-top: 10px;">' +
'        <div style="float:left;">' + 
'          <div style="font-weight:bold;">Author:</div>' +
'          <input type="text" name="author_account" value="'+this.access_list.user.uid+'" size=32 style="padding:2px; margin-top:5px;" />' +
'        </div>' +
'        <div style="float:left; margin-left:20px;">' + 
'          <div style="font-weight:bold;">Attachments:</div>' +
'          <div class="attachments">' +
'            <div>' +
'              <input type="file" name="file2attach_0" />' +
'              <input type="hidden" name="file2attach_0" value=""/ >' +
'            </div>' +
'          </div>' +
'        </div>' +
'        <div style="clear:both;"></div>' +
'      </div>' +
'      <input type="hidden" name="return_parent" value="1" />' +
'    </form>' +
'  </div>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the message">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.reply_post() ;  break ;
                case 'cancel': that.enable_view() ; break ;
                default: 
                    console.log('ELog_MessageBody.reply(): this operation is not supported: '+op) ;
                    that.enable_view() ;
                    break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        this.form_attachments_cont = this.form_cont.find('.attachments') ;
        this.form_attachments_cont.find('input:file[name="file2attach_0"]').change(function () { that.add_attachment() ; }) ;
    } ;

    this.reply_post = function () {
        if (this.form_cont.find('textarea[name="message_text"]').val() === '') {
            Fwk.report_error('Can not post the empty message. Please put some text into the message box.') ;
            return ;
        }
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post the reply w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.Status != 'success') {
                    Fwk.report_error(data.Message) ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                    return ;
                }

                var new_message = data.Entry ;
                that.parent.update_row(that, new_message) ;

                that.enable_view() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;
    
    /**
     * Create a dialog for editing the message text and adding more attachments.
     * 
     * @returns {undefined}
     */
    this.edit_message = function () {

        this.enable_edit() ;

        var html =
'<div style="float:left; margin-right:20px; padding-top:5px;">' +
'  <div style="color:maroon;">Edit message text. Note the total limit of <b>25 MB</b> for attachments.</div>' +
'  <div style="margin-top:20px; padding-left:10px;" >' +
'    <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/UpdateFFEntry4portalJSON.php" method="post">' +
'      <input type="hidden" name="id" value="'+this.message.id+'" />' +
'      <input type="hidden" name="content_type" value="TEXT" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'      <input type="hidden" name="onsuccess" value="" />' +
'      <textarea name="content" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>'+
'      <div style="font-weight:bold; margin-top: 10px;">Extra attachments:</div>'+
'      <div class="attachments">' +
'        <div>' +
'          <input type="file" name="file2attach_0" />' +
'          <input type="hidden" name="file2attach_0" value=""/ >' +
'        </div>' +
'      </div>' +
'    </form>' +
'  </div>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the modifications">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.edit_message_post() ;  break ;
                case 'cancel': that.enable_view() ; break ;
                default: 
                    console.log('ELog_MessageBody.edit_message(): this operation is not supported: '+op) ;
                    that.enable_view() ;
                    break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        this.form_attachments_cont = this.form_cont.find('.attachments') ;
        this.form_attachments_cont.find('input:file[name="file2attach_0"]').change(function () { that.add_attachment() ; }) ;

        this.form_cont.find('textarea[name="content"]').val(this.message.content) ;
    } ;
    this.edit_message_post = function () {

        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.Status != 'success') {
                    Fwk.report_error(data.Message) ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                    return ;
                }

                // Refresh the current message tree

                var new_message = data.Entry ;
                that.parent.update_row(that, new_message) ;

                that.enable_view() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;

    /**
     * Create a dialog for adding more tags to the current message.
     * 
     * @returns {undefined}
     */
    this.edit_tags = function () {

        this.enable_edit() ;

        var html =
'<div style="float:left; margin-right:20px; padding-top:5px;">' +
'  <div style="color:maroon;">Select additional tags or add the new ones.</div>' +
'  <div style="margin-top:20px; padding-left:10px;" >' +
'    <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/ExtendFFEntry4portalJSON.php" method="post">' +
'      <input type="hidden" name="message_id" value="'+this.message.id+'" />' +
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'      <input type="hidden" name="num_tags" value="'+ELog_Utils.max_num_tags+'" />' +
'      <input type="hidden" name="onsuccess" value="" />' +
'      <div style="font-weight:bold;">Tags:</div>' +
'      <div class="tags"></div>' +
'    </form>' +
'  </div>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the modifications">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.edit_tags_post() ;  break ;
                case 'cancel': that.enable_view() ; break ;
                default: 
                    console.log('ELog_MessageBody.edit_tags(): this operation is not supported: '+op) ;
                    that.enable_view() ;
                    break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        ELog_Utils.load_tags (
            this.experiment.id ,
            this.form_cont.find('.tags') ,
            function (tags) {
                // TODO: enable all inputs on the page which can't be used before
                // the tags information is ready.
                ;
            } ,
            function (msg)  {
                Fwk.report_error(msg) ;
            }
        ) ;
    } ;
    this.edit_tags_post = function () {

        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.Status != 'success') {
                    Fwk.report_error(data.Message) ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                    return ;
                }

                // TODO: Refresh the current message.

                that.enable_view() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;

    /**
     * Create a dialog for adding more attachments the current message.
     * 
     * @returns {undefined}
     */
    this.edit_attachments = function () {

        this.enable_edit() ;

        var html =
'<div style="float:left; margin-right:20px; padding-top:5px;">' +
'  <div style="color:maroon;">Select attachments to upload. Note the total limit of <b>25 MB</b>.</div>' +
'  <div style="margin-top:20px; padding-left:10px;" >' +
'    <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/ExtendFFEntry4portalJSON.php" method="post">' +
'      <input type="hidden" name="message_id" value="'+this.message.id+'" />' +
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'      <input type="hidden" name="num_tags" value="0" />' +
'      <input type="hidden" name="onsuccess" value="" />' +
'      <div style="font-weight:bold;">Attachments:</div>' +
'      <div class="attachments">' +
'        <div>' +
'          <input type="file" name="file2attach_0" />' +
'          <input type="hidden" name="file2attach_0" value=""/ >' +
'        </div>' +
'      </div>' +
'    </form>' +
'  </div>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the modifications">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.edit_attachments_post() ;  break ;
                case 'cancel': that.enable_view() ; break ;
                default: 
                    console.log('ELog_MessageBody.edit_attachments(): this operation is not supported: '+op) ;
                    that.enable_view() ;
                    break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        this.form_attachments_cont = this.form_cont.find('.attachments') ;
        this.form_attachments_cont.find('input:file[name="file2attach_0"]').change(function () { that.add_attachment() ; }) ;
    } ;
    this.edit_attachments_post = function () {

        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.Status != 'success') {
                    Fwk.report_error(data.Message) ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                    return ;
                }

                // TODO: Refresh the current message.

                that.enable_view() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;

    this.edit_run = function (operation) {
        var params = {id: this.message.id, scope: 'run'} ;
        switch (operation) {

            case 'attach':

                Fwk.form_dialog (
                    'Attaching message to a run' ,
                    '<b>Run<b> <input type="text" size=2 val=""/>' ,
                    function (form) {
                        var run_num = parseInt(form.find('input').val()) ;
                        if (!run_num) {
                            Fwk.report_error('Illegal run number. Please correct or cancel the operation.' ) ;
                            return false ;  // keep the dialog open
                        }
                        params.run_num = run_num ;
                        that.edit_run_submit(params) ;

                        return true ;  // close the dialog
                    }
                ) ;
                break ;

            case 'dettach':

                Fwk.ask_yes_no (
                    'Dettaching message from the run' ,
                    'Are you sure you want to dettach this message from run: <b>'+this.message.run_num+'</b>' ,
                    function () {
                        that.edit_run_submit(params) ;
                    }
                ) ;
                break ;
        }
    } ;
    this.edit_run_submit = function (params) {
        Fwk.web_service_GET (
            '../logbook/ws/MoveFFEntry4portalJSON.php' ,
            params ,
            function (data) {
                // TODO: tell the container to update the current message tree.
            } ,
            function (msg) {
                Fwk.report_error(msg) ;
            }
        ) ;
    } ;

    this.delete_message = function (operation) {
        switch (operation) {
            case 'undelete':
                this.delete_message_submit(operation, '../logbook/ws/UndeleteFFEntry4portalJSON.php') ;
                break ;
            case 'delete':
                Fwk.ask_yes_no (
                    'Information Deletion Warning' ,
                    '<span style="color:red;">You have requested to delete the selected message. Are you sure?</span>' ,
                    function () {
                        that.delete_message_submit(operation, '../logbook/ws/DeleteFFEntry4portalJSON.php') ;
                    }
                ) ;
                break ;
        }
    } ;
    this.delete_message_submit = function (operation, url) {
        Fwk.web_service_GET (
            url ,
            {id: this.message.id} ,
            function (data) {
                switch (operation) {
                    case 'undelete': that.parent.undelete_row(that) ; break ;
                    case 'delete': that.parent.delete_row(that, data.deleted_time, data.deleted_by) ; break ;
                }
            } ,
            function (msg) {
                Fwk.report_error(msg) ;
            }
        ) ;
    } ;

    /**
     * This method is used to add more attachments to the form found at
     * the following editing dialogs:
     * 
     * - posting a reply to the message
     * - editing the message
     * - adding more attachments to the message
     * 
     * @returns {undefined}
     */
    this.add_attachment = function () {
        var num = this.form_attachments_cont.find('div').size() ;
        this.form_attachments_cont.append (
'<div>' +
' <input type="file"   name="file2attach_'+num+'" />' +
' <input type="hidden" name="file2attach_'+num+'" value="" />' +
'</div>'
        ) ;
        this.form_attachments_cont.find('input:file[name="file2attach_'+num+'"]').change(function () { that.add_attachment() ; }) ;
    } ;
}
define_class (ELog_MessageBody, StackRowBody, {}, {}) ;


function ELog_MessageViewer (parent, cont, options) {

    // -- parameters

    this.parent = parent ;
    this.experiment = parent.experiment ;
    this.access_list = parent.access_list ;
    this.cont = cont ;

    // -- options

    this.hidden_header  = false ;
    this.instant_expand = false ;
    this.deleted        = false ;
    if (options) {
        for (var opt in options) {
            var val = options[opt] ;
            this[opt] = val ? true : false ;
        }   
    }

    this.messages = [] ;
    var hdr = [
        {id: 'posted',   title: 'Posted',    width: 150} ,
        {id: 'run',      title: 'Run',       width:  34, align: 'right'} ,
        {id: 'duration', title: 'Length',    width:  55, align: 'right', style: 'color:maroon;'} ,
        {id: 'attach',   title: '&nbsp',     width:  16} ,
        {id: 'tag',      title: '&nbsp;',    width:  16} ,
        {id: 'child',    title: '&nbsp;',    width:  20} ,
        {id: 'subj',     title: 'Subject',   width: 520} ,
        {id: '>'} ,
        {id: 'id',       title: 'MessageId', width:  70} ,
        {id: 'author',   title: 'Author',    width:  90}
    ] ;
    this.table = new StackOfRows (
        hdr ,
        [] ,
        {   hidden_header: this.hidden_header ,
            effect_on_insert: function (hdr_cont) {
               hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
            }
        }
    ) ;

    /**
     * @function - overloading the function of the base class Widget
     */
    this.render = function () {
        this.table.display(this.container) ;
        if (this.instant_expand) this.table.expand_or_collapse(true) ;
    } ;

    this.num_rows = function () { return this.table.num_rows() ; } ;

    this._num_runs = 0 ;
    this._min_run = 0 ;
    this._max_run = 0 ;

    this.num_runs = function () { return this._num_runs; }
    this.min_run  = function () { return this._min_run; }
    this.max_run  = function () { return this._max_run; }


    /**
     * Produce a data object to be fed into the StackOfRows as a row
     * 
     * @param object m 
     * @returns array of data objects
     */
    this.message2row = function (m) {
        var row = {
            title: {
                posted: '<b>'+m.ymd+'</b>&nbsp;&nbsp;'+m.hms ,
                run: m.run_num ? '<div class="m-run">'+m.run_num+'</div>' : '&nbsp;' ,
                author: m.author ,
                subj:  '&nbsp;' ,
                id: '&nbsp;' ,
                attach: '&nbsp;' ,
                child: '&nbsp;' ,
                tag: '&nbsp;' ,
                duration: '&nbsp;'
            } ,
            body: ''
        } ;
        if (m.is_run) {
            row.title.run = '<div class="m-run">'+m.run_num+'</div>' ;
            row.title.subj = this.run2subj(m) ;
            if (m.type !== 'begin_run') { row.title.duration = m.duration1 ; }
            row.body = new ELog_RunBody(this, m) ;
            row.color_theme = 'stack-theme-green' ;
            row.block_common_expand = true ;
        } else {
            row.title.subj = m.deleted ? '<span class="m-subj-deleted">'+m.subject+'</span>'+' <span class="m-subj-notes">deleted by <b>'+m.deleted_by+'</b> [ '+m.deleted_time+' ]</span>': m.subject ;
            row.title.id = '<div class="m-id">'+m.id+'</div>' ;
            if (m.attachments_num) row.title.attach = '<img src="../logbook/images/attachment.png" height="18">' ;
            if (m.children_num) row.title.child = '<sup><b>&crarr;</b></sup>' ;
            if (m.tags_num) row.title.tag = '<sup><b>T</b></sup>' ;
            row.body = new ELog_MessageBody(this, m) ;
        }
        return row ;
    } ;
    this.run2subj = function (m) {
        switch (m.type) {
            case 'run'       : return '<b>stop</b>';
            case 'end_run'   : return '<b>stop</b>' ;
            case 'begin_run' : return '<b>start</b>' ;
        }
        return '' ;
    } ;

    /**
     * Reload internal message store and redisplay the list of messages.
     * 
     * NOTE: The method will make a local deep copy of the input messages.
     * 
     * @param Array messages
     * @returns {undefined}
     */
    this.load = function (messages) {

        this.messages = jQuery.extend(true, [], messages) ;
        this.messages.reverse() ;

        this._num_runs = 0 ;
        this._min_run = 0 ;
        this._max_run = 0 ;

        var rows = [] ;
        for (var i in this.messages) {
            var m = this.messages[i] ;
            if (typeof m === 'string') {
                m = eval('('+m+')') ;
                this.messages[i] = m ;
            }
            rows.push(this.message2row(m)) ;
            if (m.is_run) {
                this._num_runs++ ;
                this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
            }
        }
        this.table.set_rows(rows) ;
        this.display(this.cont) ;
    } ;

    /**
     * Insert new messages in front of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.update = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages in front of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will always get to the front.

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this.messages.length) this.messages.splice(0, 0, m) ;
                else                      this.messages.push(m) ;
                this.table.insert_front(this.message2row(m)) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }
            }
        }
    } ;

    /**
     * Append messages at the bottom of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.append = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages iat the very end of the of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will alwats get to the front.

            new_messages.reverse() ;    // need this to append newst message first

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this.messages.length) this.messages.splice(0, 0, m) ;
                else                      this.messages.push(m) ;
                this.table.append(this.message2row(m)) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }
            }
        }
    } ;

    /**
     * This handler is meant to be used by row (bodies) to report that their
     * content has been updated.
     * 
     * The method will update its local data storage (for the message content)
     * and redisplay the message tree.
     * 
     * @param StackRow old_row
     * @param Array new_message
     * @returns {undefined}
     */
    this.update_row = function (old_row, new_message) {
        if (typeof new_message === 'string') new_message = eval('('+new_message+')') ;
        var id = old_row.message.id ;
        for (var i in this.messages) {
            var m = this.messages[i] ;
            if (id === m.id) {
                this.messages[i] = new_message ;
                this.table.update_row (
                    old_row.row_id ,
                    this.message2row(new_message)) ;
                return ;
            }
        }
    } ;

    this.undelete_row = function (row) {
        var id = row.message.id ;
        for (var i in this.messages) {
            var m = this.messages[i] ;
            if (id === m.id) {
                m.deleted = 0 ;
                m.deleted_time = '' ;
                m.deleted_by = '' ;
                this.table.update_row (
                    row.row_id ,
                    this.message2row(m)) ;
                return ;
            }
        }
    } ;

    this.delete_row = function (row, deleted_time, deleted_by) {
        var id = row.message.id ;
        for (var i in this.messages) {
            var m = this.messages[i] ;
            if (id === m.id) {
                m.deleted = 1 ;
                m.deleted_time = deleted_time ;
                m.deleted_by = deleted_by ;
                this.table.update_row (
                    row.row_id ,
                    this.message2row(m)) ;
                return ;
            }
        }
    } ;

    this.focus_at_message = function (message_id) {
        for (var i in this.table.rows) {
            var row = this.table.rows[i] ;
            if (row.data_object.body.message.id == message_id) {
                var id = row.id ;
                this.table.expand_or_collapse_row(id, true, $('#fwk-center')) ;
                return ;
            }
        }
        console.log('ELog_MessageViewer.focus_at_message('+message_id+') the message was not found in StackOfRows') ;
    } ;

    this.focus_at_run = function (run_num) {
        for (var i in this.table.rows) {
            var row = this.table.rows[i] ;
            if (row.data_object.body.message.is_run && row.data_object.body.message.run_num == run_num) {
                var id = row.id ;
                this.table.expand_or_collapse_row(id, true, $('#fwk-center')) ;
                return ;
            }
        }
        console.log('ELog_MessageViewer.focus_at_run('+run_num+') the message was not found in StackOfRows') ;
    } ;

    // Must be the last call. Otherwise the widget won't be able to see
    // functon 'render()' defined above in this code.

    this.display(this.cont) ;
}
define_class (ELog_MessageViewer, Widget, {}, {}) ;
