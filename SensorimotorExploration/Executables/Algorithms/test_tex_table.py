

from SensorimotorExploration.Algorithm.utils.functions import get_table_from_dict



if __name__ == '__main__':
    result ={'proprio_autonomous': {'social': {'max': [2.7833423038070362],
        'mean': [1.7590966871178011],
        'min': [0.093600812761614929]},
        'whole': {'max': [3.8589819466695827],
        'mean': [1.9914148477846227],
        'min': [0.0]}},
        'proprio_social': {'social': {'max': [2.7333776068651026],
        'mean': [1.5959070416769057],
        'min': [0.13557667138312895]},
        'whole': {'max': [3.8589819466695827],
        'mean': [1.98835384274002],
        'min': [0.0]}},
        'simple_autonomous': {'social': {'max': [2.7833423038070362],
        'mean': [1.929001258548708],
        'min': [0.1549633533337193]},
        'whole': {'max': [3.8472667454006242],
        'mean': [2.117981234925201],
        'min': [0.0]}},
        'simple_social': {'social': {'max': [2.7833423038070362],
        'mean': [1.9720842031298642],
        'min': [0.1549633533337193]},
        'whole': {'max': [3.9241875511051973],
        'mean': [2.105806828535528],
        'min': [0.0]}}}

    # result = {'a':0}
    print(get_table_from_dict(result))