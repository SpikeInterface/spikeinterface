from .method_list import matching_methods

from .main import find_spikes_from_templates


# generic class for template engine
class BaseTemplateMatchingEngine:
    default_params = {}
    
    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        """This function runs before loops"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        """This function serializes kwargs to distribute them to workers"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def unserialize_in_worker(cls, recording, kwargs):
        """This function unserializes kwargs in workers"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def get_margin(cls, recording, kwargs):
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def main_function(cls, traces, method_kwargs):
        """This function returns the number of samples for the chunk margins"""
        # need to be implemented in subclass
        raise NotImplementedError
