import numpy
from io import StringIO

from . import _aeqdsk


def test_writeread():
    """
    Test that data can be written then read back
    """
    data = {
        "shot": 66832,
        "time": 2384.0,
        "jflag": 1,
        "lflag": 0,
        "limloc": "SNB",
        "mco2v": 3,
        "mco2r": 1,
        "qmflag": "CLC",
        "tsaisq": 11.7513361,
        "rcencm": 169.550003,
        "bcentr": -2.06767821,
        "pasmat": 1213135.38,
        "cpasma": 1207042.13,
        "rout": 168.491165,
        "zout": -6.82398081,
        "aout": 63.0725098,
        "eout": 1.73637426,
        "doutu": 0.160389453,
        "doutl": 0.329588085,
        "vout": 19912044.0,
        "rcurrt": 170.800049,
        "zcurrt": 7.52815676,
        "qsta": 6.26168156,
        "betat": 0.60095495,
        "betap": 0.326897353,
        "ali": 1.47733176,
        "oleft": 3.73984718,
        "oright": 4.84749842,
        "otop": 32.4465942,
        "obott": 20.2485809,
        "qpsib": 4.39304399,
        "vertn": -0.675418258,
        "rco2v": [216.307495, 155.99646, 121.109322],
        "dco2v": [27324300900000.0, 29309569900000.0, 29793563200000.0],
        "rco2r": [125.545105],
        "dco2r": [32812950400000.0],
        "shearb": 4.02759838,
        "bpolav": 0.282110274,
        "s1": 2.155056,
        "s2": 1.09512568,
        "s3": 0.640428185,
        "qout": 7.93196821,
        "olefs": -50.0,
        "orighs": -50.0,
        "otops": -50.0,
        "sibdry": -0.016445132,
        "areao": 19292.2441,
        "wplasm": 309183.625,
        "terror": 0.000789557525,
        "elongm": 1.18666041,
        "qqmagx": 0.620565712,
        "cdflux": -0.0285836495,
        "alpha": 1.32450712,
        "rttt": 155.478485,
        "psiref": 0.0,
        "xndnt": 0.0,
        "rseps1": 147.703217,
        "zseps1": -116.341461,
        "rseps2": -999.0,
        "zseps2": -999.0,
        "sepexp": 5.94302845,
        "obots": -50.0,
        "btaxp": -2.18051338,
        "btaxv": -2.03286076,
        "aaq1": 30.0145092,
        "aaq2": 46.8485107,
        "aaq3": 54.2332726,
        "seplim": 3.73984718,
        "rmagx": 172.453949,
        "zmagx": 9.105937,
        "simagx": 0.380751818,
        "taumhd": 64.3431244,
        "betapd": 0.473303556,
        "betatd": 0.870102286,
        "wplasmd": 447656.406,
        "diamag": -2.33697938e-05,
        "vloopt": 0.110378414,
        "taudia": 93.1602173,
        "qmerci": 0.0,
        "tavem": 10.0,
    }

    output = StringIO()

    # Write to string
    _aeqdsk.write(data, output)

    # Move to the beginning of the buffer
    output.seek(0)

    # Read from string
    data2 = _aeqdsk.read(output)

    # Check that data and data2 are the same
    for key in data:
        if isinstance(data[key], str):
            assert data2[key] == data[key]
        else:
            # Number, or list of numbers
            numpy.testing.assert_allclose(data2[key], data[key])
