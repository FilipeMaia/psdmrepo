from psmon import app, config, client


class PublishError(Exception):
    """
    Class for exceptions thrown by PublishManager instances.
    """
    pass


class PublishManager(object):
    """
    Class for handling the functionality of the publish module.

    A single instance of this class is created on import of the module. It can 
    then be accessed via a series of static functions defined in the module.
    """
    def __init__(
            self,
            port=config.APP_PORT,
            bufsize=config.APP_BUFFER,
            local=config.APP_LOCAL,
            renderer=config.APP_CLIENT,
            rate=config.APP_RATE,
            recv_limit=config.APP_RECV_LIMIT
        ):
        self.port = port
        self.bufsize = bufsize
        self.local = local
        self.renderer = renderer
        self.rate = rate
        self.recv_limit = recv_limit
        self.daemon = False
        self.disable = False
        self.publisher = app.ZMQPublisher()
        self.reset_listener = app.ZMQListener(self.publisher.comm_socket)
        self.plot_opts = app.PlotInfo()
        self.active_clients = {}

    @property
    def initialized(self):
        return self.publisher.initialized

    def send(self, topic, data):
        """
        Publishes 'data' to the passed topic of the ZMQPublisher.
        """
        if not self.initialized and not self.disable:
            try:
                from mpi4py import MPI
                if MPI.COMM_WORLD.Get_rank() == 0:
                    self.init()
                else:
                    raise PublishError('Cannot send messages on a non-rank-zero MPI process without explicitly calling publish.init')
            except ImportError:
                self.init()

        if self.local:
            if topic in self.active_clients:
                if not self.active_clients[topic].is_alive():
                    self.create_client(topic)
            else:
                self.create_client(topic)

        self.publisher.send(topic, data)

    def create_client(self, topic):
        """
        Spawns a local client listening to the specified topic.
        """
        client_opts = app.ClientInfo(
            self.publisher.data_endpoint,
            self.publisher.comm_endpoint,
            self.bufsize,
            self.rate,
            self.recv_limit,
            topic,
            self.renderer
        )
        self.active_clients[topic] = client.spawn_process(client_opts, self.plot_opts, self.daemon, True)

    def init(self):
        """
        Initializes the underlying ZMQPublisher and ZMQListerner.
        """
        self.publisher.initialize(self.port, self.bufsize, self.local)
        self.reset_listener.start()
        # turn off further autoconnect attempts
        self.disable = True

    def add_plot_opts(self, **kwargs):
        """
        Pass plot options as keyword args to the underlying PlotInfo object

        Returns a list of attributes that is was not able to set.
        """
        failed = []
        for name, value in kwargs.iteritems():
            if hasattr(self.plot_opts, name):
                setattr(self.plot_opts, name, value)
            else:
                failed.append(name)

        return failed

    def reset(self):
        """
        Resets the publish module manager to an unitialized state. All socket 
        connections are close and the port bindings released.
        """
        pass
