function p_appl_dictionary () {

    var that = this ;

    this.when_done = null ;

    /* -------------------------------------------------------------------------
     *   Data structures and methods to be used/called by external users
     *
     *   select(context, when_done)
     *      select a specific context
     *
     *   select_default()
     *      select default context as implemented in the object
     *
     *   if_ready2giveup(handler2call)
     *      check if the object's state allows to be released, and if so then
     *      call the specified function. Otherwise just ignore it. Normally
     *      this operation is used as a safeguard preventing releasing
     *      an interface focus if there is on-going unfinished editing
     *      within one of the interfaces associated with the object.
     *
     * -------------------------------------------------------------------------
     */
    this.name      = 'dictionary' ;
    this.full_name = 'Dictionary' ;
    this.context   = '' ;
    this.default_context = 'manufacturers' ;

    this.select = function (context) {
        that.context = context ;
        this.init() ;
    } ;
    this.select_default = function () {
        this.init() ;
        if (this.context == '') this.context = this.default_context ;
    } ;
    this.if_ready2giveup = function (handler2call) {
        this.init() ;
        handler2call() ;
    } ;

    /* ----------------------------------------
     *   Internal methods and data structures
     * ----------------------------------------
     */
    this.initialized = false ;
    this.init = function () {
        if (this.initialized) return ;
        this.initialized = true ;

        this.init_manufacturer() ;
        this.init_location() ;
        this.init_status() ;
    } ;
    this.can_manage = function () {
        return global_current_user.has_dict_priv ;
    } ;

    /* ----------------------------
     *   Manufacturers and Models
     * ----------------------------
     */
    this.manufacturer = null ;
    this.manufacturers = function () {
        return this.manufacturer ;
    } ;

    this.init_manufacturer = function () {
        $('#dictionary-manufacturers-reload').button().click(function () { that.manufacturer_load() ; }) ;

        var manufacturer2add = $('#dictionary-manufacturers').find('input[name="manufacturer2add"]') ;
        manufacturer2add.
            keyup(function (e) {
                var name = $(this).val() ;
                if (name == '') { return ; }
                if (e.keyCode == 13) { that.manufacturer_create(name) ; $(this).val('') ; return ; }}) ;

        var model2add = $('#dictionary-manufacturers').find('input[name="model2add"]') ;
        model2add.
            keyup(function (e) {
                var name = $(this).val() ;
                if (name == '') { return ; }
                if (e.keyCode == 13) { that.model_create(name) ; $(this).val('') ; return ; }}) ;

        this.manufacturer_load() ;
    } ;
    
    this.table_manufacturer = null ;

    this.manufacturer_display = function () {
        var elem = $('#dictionary-manufacturers-manufacturers') ;
        var hdr = [] ;
        if (this.can_manage()) hdr.push (
            {   name: 'DELETE', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        elem.find('.dict-manufacturer-delete').
                            button().
                            click(function () {
                                var id = this.name ;
                                that.manufacturer_delete(id) ;
                            }) ;
                    }
                }
            }) ;
        hdr.push (
            {   name: 'manufacturer', selectable: true ,
                type: {
                    select_action : function (manufacturer_name) {
                        that.model_display() ;
                    }
                }
            } ,
            {   name: 'created', hideable: true } ,
            {   name: 'by user', hideable: true } ,
            {   name: 'documentation url', hideable: true, sorted: false } ,
            {   name: 'in use', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        elem.find('.dict-manufacturer-search').
                            button().
                            click(function () {
                                var id = this.name ;
                                global_search_equipment_by_manufacturer(id) ;
                            }) ;
                    }
                }
            }
        ) ;
        var rows = [] ;
        for (var i in this.manufacturer) {
            var m = this.manufacturer[i] ;
            var row = [] ;
            if (this.can_manage()) row.push (
                Button_HTML('X', {
                    name:    m.id,
                    classes: 'dict-manufacturer-delete',
                    title:   'delete this manufacturer from the list' })) ;
            row.push(
                m.name ,
                m.created_time ,
                m.created_uid ,
                m.url ,
                Button_HTML('search', {
                    name:    m.id,
                    classes: 'dict-manufacturer-search',
                    title:   'search for all equipment of this manufacturer' })
            ) ;
            rows.push(row) ;
        }
        this.table_manufacturer = new Table (
            'dictionary-manufacturers-manufacturers', hdr, rows ,
            {selected_col: this.can_manage() ? 1 : 0},
            config.handler('dict', 'table_manufacturer')
        ) ;
        this.table_manufacturer.display() ;
        
        this.model_display(); 
        
        if (this.can_manage()) {
            var input = $('#dictionary-manufacturers').find('input[name="model2add"]') ;
            if (this.table_manufacturer.selected_object() == null)
                input.attr('disabled', 'disabled') ;
            else
                input.removeAttr('disabled') ;
        }
    } ;

    this.table_model = null ;

    this.model_display = function () {
        var elem = $('#dictionary-manufacturers-models') ;
        var hdr = [] ;
        if (this.can_manage()) hdr.push (
            {   name: 'DELETE', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        elem.find('.dict-model-delete').
                            button().
                            click(function () {
                                var id = this.name ;
                                that.model_delete(id) ; }) ;
                        elem.find('.dict-model-search').
                            button().
                            click(function () {
                                var id = this.name ;
                                global_search_equipment_by_model(id) ; }) ;
                      }}}) ;
        hdr.push (
            {   name: 'model' } ,
            {   name: 'created', hideable: true } ,
            {   name: 'by user', hideable: true } ,
            {   name: 'documentation url', hideable: true, sorted: false } ,
            {   name: 'image', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        elem.find('.dict-model-image-delete').
                            button().
                            click(function () {
                                var attachment_id = this.name ;
                                that.model_image_delete(attachment_id) ; }) ;
                        elem.find('.dict-model-image-upload').
                            button().
                            click(function () {
                                var model_id = this.name ;
                                that.model_image_upload(model_id) ; }) ;
                    }}} ,
            {   name: 'in use', hideable: true, sorted: false }
        ) ;
        var rows = [] ;
        
        var manufacturer_name = this.table_manufacturer.selected_object() ;
        if (manufacturer_name != null) {
            for (var i in this.manufacturer) {
                var manufacturer = this.manufacturer[i] ;
                if (manufacturer.name == manufacturer_name) {
                    for (var j in manufacturer.model) {
                        var model = manufacturer.model[j] ;
                        var row = [] ;
                        if (this.can_manage()) row.push (
                            Button_HTML('X', {
                                name:    model.id,
                                classes: 'dict-model-delete',
                                title:   'delete this model from the list' })) ;
                        var images_html =
                            this.can_manage() ?
                                model.default_attachment.is_available ?
                                    '<div style="float:left;">' +
                                      Button_HTML('delete', {
                                          name:    model.default_attachment.id,
                                          classes: 'dict-model-image-delete',
                                          title:   'delete this image' }) +
                                    '</div>' +
                                    '<div style="float:left; margin-left:10px;">' +
                                    '  <a class="link" href="../irep/model_attachments/'+model.default_attachment.id+'/file" target="_blank" title="click on the image to open/download a full size image in a separate tab"><img src="../irep/model_attachments/preview/'+model.default_attachment.id+'" /></a>' +
                                    '</div>' +
                                    '<div style="clear:both;"></div>'
                                :
                                    '<div style="float:left;">' +
                                      Button_HTML('upload', {
                                          name:    model.id,
                                          classes: 'dict-model-image-upload visible',
                                          title:   'upload image' }) +
                                    '</div>'+
                                    '<div class="hidden" style="float:left; margin-left:5px;" id="dict-model-image-upload-'+model.id+'">' +
                                    '  <form enctype="multipart/form-data" action="../irep/ws/model_image_upload.php" method="post">' +
                                    '    <input type="hidden" name="model_id" value="'+model.id+'" />' +
                                    '    <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
                                    '    <input type="file" name="file2attach" onchange="dict.model_image_submit('+model.id+')" />' +
                                    '    <input type="hidden" name="file2attach" value="" />' +
                                    '  </form>' +
                                    '</div>' +
                                    '<div style="clear:both;"></div>'
                            :
                                model.default_attachment.is_available ?
                                    '<a class="link" href="../irep/model_attachments/'+model.default_attachment.id+'/file" target="_blank" title="click on the image to open/download a full size image in a separate tab"><img src="../irep/model_attachments/preview/'+model.default_attachment.id+'" /></a>'
                                :
                                    ''
                            ;

                        row.push(
                            model.name ,
                            model.created_time ,
                            model.created_uid ,
                            model.url ,
                            images_html ,
                            Button_HTML('search', {
                                name:    model.id,
                                classes: 'dict-model-search',
                                title:   'search for all equipment of this model' })
                        ) ;
                        rows.push(row) ;
                    }
                    break ;
                }
            }
        }
        this.table_model = new Table (
            'dictionary-manufacturers-models', hdr, rows ,
            {},
            config.handler('dict', 'table_model')
        ) ;
        this.table_model.display() ;
    } ;

    this.manufacturer_load = function () {
        this.manufacturer_action (
            '../irep/ws/manufacturer_get.php' ,
            {}
        ) ;
    } ;
    this.manufacturer_create = function (name) {
        this.manufacturer_action (
            '../irep/ws/manufacturer_new.php' ,
            {name: name}
        ) ;
    } ;
    this.manufacturer_delete = function (id) {
        ask_yes_no (
            'Confirm Manufacturer Deletion' ,
            'Are you sure you want to delete this manufacturer from the Dictionary?' ,
            function () {
                that.manufacturer_action (
                    '../irep/ws/manufacturer_delete.php' ,
                    {id: id}
                ) ;
            }
        ) ;
    } ;
    this.model_create = function (name) {
        this.model_action (
            '../irep/ws/model_new.php' ,
            {manufacturer_name: this.table_manufacturer.selected_object(), model_name: name}
        ) ;
    } ;
    this.model_delete = function (id) {
        ask_yes_no (
            'Confirm Model Deletion' ,
            'Are you sure you want to delete this model from the Dictionary?' ,
            function () {
                that.model_action (
                    '../irep/ws/model_delete.php' ,
                    {id: id}
                ) ;
            }
        ) ;
    } ;
    this.model_image_upload = function (model_id) {
        var elem = $('#dict-model-image-upload-'+model_id) ;
        var button = $('button.dict-model-image-upload[name="'+model_id+'"]') ;
        elem.removeClass('hidden').addClass('visible') ;
        button.removeClass('visible').addClass('hidden') ;
    } ;
    this.model_image_submit = function (model_id) {
        var form = $('#dict-model-image-upload-'+model_id).find('form') ;
        form.ajaxSubmit({
            success: function(data) {
                if (data.status != 'success') {
                    report_error(data.message) ;
                } else {
                    that.manufacturer = data.manufacturer ;
                }
                that.model_display() ;
            } ,
            error: function() {
                report_error('failed to contact the server in order to upload the image') ;
            } ,
            dataType: 'json'
        }) ;
    } ;
    this.model_image_delete = function (attachment_id) {
        this.model_action (
            '../irep/ws/model_image_delete.php' ,
            {id: attachment_id}
        ) ;
    } ;
    this.manufacturer_action = function (url, params) {
        web_service_GET(url, params, function (data) {
            that.manufacturer = data.manufacturer ;
            that.manufacturer_display() ;
        }) ;
    } ;
    this.model_action = function (url, params) {
        web_service_GET(url, params, function (data) {
            that.manufacturer = data.manufacturer ;
            that.model_display() ;
        }) ;
    } ;

    /* ---------
     * Locations
     * ---------
     */

    this.location = null ;
    this.locations = function () {
        return this.location ;
    } ;

    this.init_location = function () {
        $('#dictionary-locations-reload').button().click(function () { that.location_load() ; }) ;

        var location2add = $('#dictionary-locations').find('input[name="location2add"]') ;
        location2add.
            keyup(function (e) {
                var name = $(this).val() ;
                if (name == '') { return ; }
                if (e.keyCode == 13) { that.location_create(name) ; $(this).val('') ; return ; }}) ;

        var room2add = $('#dictionary-locations').find('input[name="room2add"]') ;
        room2add.
            keyup(function (e) {
                var name = $(this).val() ;
                if (name == '') { return ; }
                if (e.keyCode == 13) { that.room_create(name) ; $(this).val('') ; return ; }}) ;

        this.location_load() ;
    } ;

    this.table_location = null ;

    this.location_display = function () {
        var elem = $('#dictionary-locations-locations') ;
        var hdr = [] ;
        if (this.can_manage()) hdr.push (
            {   name: 'DELETE', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        elem.find('.dict-location-delete').
                            button().
                            click(function () {
                                var id = this.name ;
                                that.location_delete(id) ; }) ;
                            elem.find('.dict-location-search').
                                button().
                                click(function () {
                                    var id = this.name ;
                                    global_search_equipment_by_location(id) ; }) ;
                      }}
            }) ;
        hdr.push (
            {   name: 'location', selectable: true ,
                type: {
                    select_action : function (location_name) {
                        that.room_display() ;
                    }
                }
            } ,
            {   name: 'created', hideable: true } ,
            {   name: 'by user', hideable: true } ,
            {   name: 'in use', hideable: true, sorted: false }
        ) ;
        var rows = [] ;
        for (var i in this.location) {
            var m = this.location[i] ;
            var row = [] ;
            if (this.can_manage()) row.push (
                Button_HTML('X', {
                    name:    m.id,
                    classes: 'dict-location-delete',
                    title:   'delete this location from the list' })) ;
            row.push(
                m.name ,
                m.created_time ,
                m.created_uid ,
                Button_HTML('search', {
                    name:    m.id,
                    classes: 'dict-location-search',
                    title:   'search for all equipment of this location' })
            ) ;
            rows.push(row) ;
        }
        this.table_location = new Table (
            'dictionary-locations-locations', hdr, rows ,
            {selected_col: this.can_manage() ? 1 : 0},
            config.handler('dict', 'table_location')
        ) ;
        this.table_location.display() ;
        
        this.room_display(); 
        
        if (this.can_manage()) {
            var input = $('#dictionary-locations').find('input[name="room2add"]') ;
            if (this.table_location.selected_object() == null)
                input.attr('disabled', 'disabled') ;
            else
                input.removeAttr('disabled') ;
        }
    } ;

    this.table_room = null ;

    this.room_display = function () {
        var elem = $('#dictionary-locations-rooms') ;
        var hdr = [] ;
        if (this.can_manage()) hdr.push (
            {   name: 'DELETE', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        elem.find('.dict-room-delete').
                            button().
                            click(function () {
                                var id = this.name ;
                                that.room_delete(id) ; }) ;
                        elem.find('.dict-room-search').
                            button().
                            click(function () {
                                var id = this.name ;
                                global_search_equipment_by_room(id) ; }) ;
                      }}}) ;
        hdr.push (
            {   name: 'room' } ,
            {   name: 'created', hideable: true } ,
            {   name: 'by user', hideable: true } ,
            {   name: 'in use', hideable: true, sorted: false }
        ) ;
        var rows = [] ;
 
        var location_name = this.table_location.selected_object() ;
        if (location_name != null) {
            for (var i in this.location) {
                var location = this.location[i] ;
                if (location.name == location_name) {
                    for (var j in location.room) {
                        var room = location.room[j] ;
                        var row = [] ;
                        if (this.can_manage()) row.push (
                            Button_HTML('X', {
                                name:    room.id,
                                classes: 'dict-room-delete',
                                title:   'delete this room from the list' })) ;
                        row.push(
                            room.name ,
                            room.created_time ,
                            room.created_uid ,
                            Button_HTML('search', {
                                name:    room.id,
                                classes: 'dict-room-search',
                                title:   'search for all equipment of this room' })
                        ) ;
                        rows.push(row) ;
                    }
                    break ;
                }
            }
        }
        this.table_room = new Table (
            'dictionary-locations-rooms', hdr, rows ,
            {},
            config.handler('dict', 'table_room')
        ) ;
        this.table_room.display() ;
    } ;
    this.location_load = function () {
        this.location_action (
            '../irep/ws/location_get.php' ,
            {}
        ) ;
    } ;
    this.location_create = function (name) {
        this.location_action (
            '../irep/ws/location_new.php' ,
            {name: name}
        ) ;
    } ;
    this.location_delete = function (id) {
        ask_yes_no (
            'Confirm Location Deletion' ,
            'Are you sure you want to delete this location from the Dictionary?' ,
            function () {
                that.location_action (
                    '../irep/ws/location_delete.php' ,
                    {id: id}
                ) ;
            }
        ) ;
    } ;
    this.location_action = function (url, params) {
        web_service_GET(url, params, function (data) {
            that.location = data.location ;
            that.location_display() ;
        }) ;
    } ;
    this.room_create = function (name) {
        this.room_action (
            '../irep/ws/room_new.php' ,
            {location_name: this.table_location.selected_object(), room_name: name}
        ) ;
    } ;
    this.room_delete = function (id) {
        ask_yes_no (
            'Confirm Room Deletion' ,
            'Are you sure you want to delete this room from the Dictionary?' ,
            function () {
                that.room_action (
                    '../irep/ws/room_delete.php' ,
                    {id: id}
                ) ;
            }
        ) ;
    } ;
    this.room_action = function (url, params) {
        web_service_GET(url, params, function (data) {
            that.location = data.location ;
            that.room_display() ;
        }) ;
    } ;

    /* -------------------------
     * Statuses and sub-statuses
     * -------------------------
     */

    this.status = null ;
    this.statuses = function () {
        return this.status ;
    } ;

    this.init_status = function () {
        $('#dictionary-statuses-reload').button().click(function () { that.status_load() ; }) ;

        var status2add = $('#dictionary-statuses').find('input[name="status2add"]') ;
        status2add.
            keyup(function (e) {
                var name = $(this).val() ;
                if (name == '') { return ; }
                if (e.keyCode == 13) { that.status_create(name) ; $(this).val('') ; return ; }}) ;

        var status22add = $('#dictionary-statuses').find('input[name="status22add"]') ;
        status22add.
            keyup(function (e) {
                var name = $(this).val() ;
                if (name == '') { return ; }
                if (e.keyCode == 13) { that.status2_create(name) ; $(this).val('') ; return ; }}) ;

        this.status_load() ;
    } ;
    
    this.table_status = null ;

    this.status_display = function () {
        var elem = $('#dictionary-statuses-statuses') ;
        var hdr = [] ;
        if (this.can_manage()) hdr.push (
            {   name: 'DELETE', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        elem.find('.dict-status-delete').
                            button().
                            click(function () {
                                var id = this.name ;
                                that.status_delete(id) ;
                            }) ;
                    }
                }
            }) ;
        hdr.push (
            {   name: 'status', selectable: true ,
                type: {
                    select_action : function (status_name) {
                        that.status2_display() ;
                    }
                }
            } ,
            {   name: 'created', hideable: true } ,
            {   name: 'by user', hideable: true } ,
            {   name: 'in use', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        elem.find('.dict-status-search').
                            button().
                            click(function () {
                                var id = this.name ;
                                global_search_equipment_by_status(id) ;
                            }) ;
                    }
                }
            }
        ) ;
        var rows = [] ;
        for (var i in this.status) {
            var status = this.status[i] ;
            var row = [] ;
            if (this.can_manage()) row.push (
                status.is_locked ? '&nbsp;' :
                Button_HTML('X', {
                    name:    status.id,
                    classes: 'dict-status-delete',
                    title:   'delete this status from the list' })) ;
            row.push(
                status.name ,
                status.created_time ,
                status.created_uid ,
                Button_HTML('search', {
                    name:    status.id,
                    classes: 'dict-status-search',
                    title:   'search for all equipment of this status' })
            ) ;
            rows.push(row) ;
        }
        this.table_status = new Table (
            'dictionary-statuses-statuses', hdr, rows ,
            {selected_col: this.can_manage() ? 1 : 0},
            config.handler('dict', 'table_status')
        ) ;
        this.table_status.display() ;
        
        this.status2_display(); 
        
        if (this.can_manage()) {
            var input = $('#dictionary-statuses').find('input[name="status22add"]') ;
            if (this.table_status.selected_object() == null)
                input.attr('disabled', 'disabled') ;
            else
                input.removeAttr('disabled') ;
        }
    } ;

    this.table_status2 = null ;

    this.status2_display = function () {
        var elem = $('#dictionary-statuses-statuses2') ;
        var hdr = [] ;
        if (this.can_manage()) hdr.push (
            {   name: 'DELETE', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        elem.find('.dict-status2-delete').
                            button().
                            click(function () {
                                var id = this.name ;
                                that.status2_delete(id) ; }) ;
                        elem.find('.dict-status2-search').
                            button().
                            click(function () {
                                var id = this.name ;
                                global_search_equipment_by_status2(id) ; }) ;
                      }}}) ;
        hdr.push (
            {   name: 'sub-status', style: 'font-weight:bold;' } ,
            {   name: 'created', hideable: true } ,
            {   name: 'by user', hideable: true } ,
            {   name: 'in use', hideable: true, sorted: false }
        ) ;
        var rows = [] ;
        
        var status_name = this.table_status.selected_object() ;
        if (status_name != null) {
            for (var i in this.status) {
                var status = this.status[i] ;
                if (status.name == status_name) {
                    for (var j in status.status2) {
                        var status2 = status.status2[j] ;
                        var row = [] ;
                        if (this.can_manage()) row.push (
                            status2.is_locked ? '&nbsp;' :
                            Button_HTML('X', {
                                name:    status2.id,
                                classes: 'dict-status2-delete',
                                title:   'delete this sub-status from the list' })) ;
                        row.push(
                            status2.name ,
                            status2.created_time ,
                            status2.created_uid ,
                            Button_HTML('search', {
                                name:    status2.id,
                                classes: 'dict-status2-search',
                                title:   'search for all equipment of this status and sub-status' })
                        ) ;
                        rows.push(row) ;
                    }
                    break ;
                }
            }
        }
        this.table_status2 = new Table (
            'dictionary-statuses-statuses2', hdr, rows ,
            {},
            config.handler('dict', 'table_status2')
        ) ;
        this.table_status2.display() ;
    } ;

    this.status_load = function () {
        this.status_action (
            '../irep/ws/status_get.php' ,
            {}
        ) ;
    } ;
    this.status_create = function (name) {
        this.status_action (
            '../irep/ws/status_new.php' ,
            {status: name}
        ) ;
    } ;
    this.status_delete = function (id) {
        ask_yes_no (
            'Confirm Status Deletion' ,
            'Are you sure you want to delete this status and all its sub-statuses from the Dictionary?' ,
            function () {
                that.status_action (
                    '../irep/ws/status_delete.php' ,
                    {scope:'status', id: id}
                ) ;
            }
        ) ;
    } ;
    this.status2_create = function (name) {
        this.status2_action (
            '../irep/ws/status_new.php' ,
            {status: this.table_status.selected_object(), status2: name}
        ) ;
    } ;
    this.status2_delete = function (id) {
        ask_yes_no (
            'Confirm Sub-Status Deletion' ,
            'Are you sure you want to delete this sub-status from the Dictionary?' ,
            function () {
                that.status2_action (
                    '../irep/ws/status_delete.php' ,
                    {scope:'status2', id: id}
                ) ;
            }
        ) ;
    } ;
    this.status_action = function (url, params) {
        web_service_GET(url, params, function (data) {
            that.status = data.cable_status ;
            that.status_display() ;
        }) ;
    } ;
    this.status2_action = function (url, params) {
        web_service_GET(url, params, function (data) {
            that.status = data.cable_status ;
            that.status2_display() ;
        }) ;
    } ;
}
var dict = new p_appl_dictionary();
