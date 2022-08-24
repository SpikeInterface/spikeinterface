import neo


def get_reader(raw_class, **neo_kwargs):
    neoIOclass = eval('neo.rawio.' + raw_class)
    neo_reader = neoIOclass(**neo_kwargs)
    neo_reader.parse_header()
    
    return neo_reader


def get_streams(raw_class, **neo_kwargs):
    neo_reader = get_reader(raw_class, **neo_kwargs)
    
    stream_channels = neo_reader.header['signal_streams']
    stream_names = list(stream_channels['name'])
    stream_ids = list(stream_channels['id'])
    
    return stream_names, stream_ids


def get_num_blocks(raw_class, **neo_kwargs):
    neo_reader = get_reader(raw_class, **neo_kwargs)
    return neo_reader.block_count()
