from spikeinterface.widgets.utils import get_some_colors


def test_get_some_colors():
    
    keys = ['a', 'b', 'c', 'd']
    
    colors = get_some_colors(keys, color_engine='auto')
    # print(colors)

    colors = get_some_colors(keys, color_engine='distinctipy')
    # print(colors)

    colors = get_some_colors(keys, color_engine='matplotlib', shuffle=None)
    # print(colors)
    colors = get_some_colors(keys, color_engine='matplotlib', shuffle=False)
    colors = get_some_colors(keys, color_engine='matplotlib', shuffle=True)

    colors = get_some_colors(keys, color_engine='colorsys')
    # print(colors)
    

if __name__ == '__main__':
    test_get_some_colors()
