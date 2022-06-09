

class BaseWidget:
    # this need to be reset in the subclass
    possible_backends = None
    
    def __init__(self, plot_data=None, backend=None, **backend_kwargs):
        # every widgets must prepare a dict "plot_data" in the init
        print('yep')
        self.plot_data = plot_data
        self.backend = backend
        
        # delegated to one of the plotter
        self.do_plot(backend, **backend_kwargs)
        
    def do_plot(self, backend, **backend_kwargs):
        assert backend in self.possible_backends, f'Backend {backend} not supported for this widget'    
        plotter = self.possible_backends[backend]()
        plotter.do_plot(self.plot_data, **backend_kwargs)

    @classmethod
    def register_backend(cls, backend_plotter):
        cls.possible_backends[backend_plotter.backend] = backend_plotter


class BackendPlotter():
    backend = ''
    
    @classmethod
    def register(cls, widget_cls):
        #~ print('BackendPlotter.register', print(cls), isinstance(cls, BackendPlotter))
        widget_cls.register_backend(cls)

