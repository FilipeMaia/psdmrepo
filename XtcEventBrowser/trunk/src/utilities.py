class PyanaOptions( object ):
    def __init__( self ):
        pass


    def getOptString(self, options_string) :
        """
        parse the option string,
        return a tuple of N, item(s), i.e. item or list of items
        """
        if options_string is None:
            return 0, None

        options = options_string.split(" ")

        if len(options)==0 :
            print "option %s has no items!" % options_string
            return 0, None

        elif len(options)==1 :
            print "option %s has one item" % options_string

            if ( options_string == "" or
                 options_string == "None" or
                 options_string == "No" ) :
                return 0, None

            return 1, options[0]

        elif len(options)>1 :
            print "option %s has %d items" % (options_string, len(options))
            return len(options), options


    def getOptInt(self, options_string):
        if options_string is None: return None

        N, opt = self.getOptString(options_string)
        if N is 1:
            return 1, int(opt)
        if N > 1 :
            items = []
            for item in opt :
                items.append( int(item) )
            return N, items


    def getOptInteger(self, options_string):
        if options_string is None: return None

        if options_string == "" : return None
        return int(options_string)


    def getOptBoolean(self, options_string):
        if options_string is None: return None

        opt = options_string
        if opt == "False" or opt == "0" or opt == "No" or opt == "" : return 1, False
        if opt == "True" or opt == "1" or opt == "Yes" : return 1, True
        else :
            print "utilities.py: cannot parse option ", opt
            return 1, None

    def getOptBooleans(self, options_string):
        if options_string is None: return None

        N, opt_list = self.getOptStrings(options_string)
        if N == 0 : return 0, None

        
        opts = []
        for opt in optlist :
            opts.append( self.getOptBoolean(opt) )

        return N, opts


