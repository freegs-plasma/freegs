"""

fields - Lists the variables stored in the file, a default value, and a description

"""

from . import _fileutils as fu

import warnings

# List of file data variables, default values, and documentation
# This is used in both reader and writer
fields = [("tsaisq", 0.0, "total chi2 from magnetic probes, flux loops, Rogowski and external coils"),
          ("rcencm", 100., "major radius in cm for vacuum field BCENTR"),
          ("bcentr", 1.0, "vacuum toroidal magnetic field in Tesla at RCENCM"),
          ("pasmat", 1e6, "measured plasma toroidal current in Ampere"),
          
          ("cpasma", 1e6, "fitted plasma toroidal current in Ampere-turn"),
          ("rout", 100., "major radius of geometric center in cm"),
          ("zout", 0.0, "Z of geometric center in cm"),
          ("aout", 50., "plasma minor radius in cm"),
          
          ("eout", 1.0, "Plasma boundary elongation"),
          ("doutu", 1.0, "upper triangularity"),
          ("doutl", 1.0, "lower triangularity"),
          ("vout", 1000., "plasma volume in cm3"),
          
          ("rcurrt", 100., "major radius in cm of current centroid"),
          ("zcurrt", 0.0, "Z in cm at current centroid"),
          ("qsta", 5.0, "equivalent safety factor q*"),
          ("betat", 1.0, "toroidal beta in %"),
          
          ("betap", 1.0, "poloidal beta with normalization average poloidal magnetic BPOLAV defined through Ampere's law"),
          ("ali", 0.0, "li with normalization average poloidal magnetic defined through Ampere's law"),
          ("oleft", 10., "plasma inner gap in cm"),
          ("oright", 10., "plasma outer gap in cm"),
          
          ("otop", 10., "plasma top gap in cm"),
          ("obott", 10., "plasma bottom gap in cm"),
          ("qpsib", 5., "q at 95% of poloidal flux"),
          ("vertn", 1.0, "vacuum field (index? -- seems to be float) at current centroid"),
    
          #fmt_1040 = r '^\s*' + 4 * r '([\s\-]\d+\.\d+[Ee][\+\-]\d\d)'

          #read(neqdsk, 1040)(rco2v(k, jj), k = 1, mco2v)
          (None, None, None),  # New line
          ("rco2v", lambda data: [0.0] * data["mco2v"], "1D array : path length in cm of vertical CO2 density chord"),
          
          #read(neqdsk, 1040)(dco2v(jj, k), k = 1, mco2v)
          (None, None, None),  # New line
          ("dco2v", lambda data: [0.0] * data["mco2v"], "line average electron density in cm3 from vertical CO2 chord"),
          
          #read(neqdsk, 1040)(rco2r(k, jj), k = 1, mco2r)
          (None, None, None),  # New line
          ("rco2r", lambda data: [0.0] * data["mco2r"], "path length in cm of radial CO2 density chord"),
          
          #read(neqdsk, 1040)(dco2r(jj, k), k = 1, mco2r)
          (None, None, None),  # New line
          ("dco2r", lambda data: [0.0] * data["mco2r"], "line average electron density in cm3 from radial CO2 chord"),
          
          (None, None, None),  # New line
          
          ("shearb", 0.0, ""), 
          ("bpolav", 1.0, "average poloidal magnetic field in Tesla defined through Ampere's law"),
          ("s1", 0.0, "Shafranov boundary line integrals"),
          ("s2", 0.0, "Shafranov boundary line integrals"),
                              
          ("s3", 0.0, "Shafranov boundary line integrals"),
          ("qout", 0.0, "q at plasma boundary"),
          ("olefs", 0.0, ""),
          ("orighs", 0.0, "outer gap of external second separatrix in cm"),
          
          ("otops", 0.0, "top gap of external second separatrix in cm"),
          ("sibdry", 1.0, ""),
          ("areao", 100., "cross sectional area in cm2"),
          ("wplasm", 0.0, ""), 
          
          ("terror", 0.0, "equilibrium convergence error"),
          ("elongm", 0.0, "elongation at magnetic axis"),
          ("qqmagx", 0.0, "axial safety factor q(0)"),
          ("cdflux", 0.0, "computed diamagnetic flux in Volt-sec"),
                              
          ("alpha", 0.0, "Shafranov boundary line integral parameter"),
          ("rttt", 0.0, "Shafranov boundary line integral parameter"),
          ("psiref", 1.0, "reference poloidal flux in VS/rad"),
          ("xndnt", 0.0, "vertical stability parameter, vacuum field index normalized to critical index value"),
                              
          ("rseps1", 1.0, "major radius of x point in cm"),
          ("zseps1", -1.0, ""),
          ("rseps2", 1.0, "major radius of x point in cm"),
          ("zseps2", 1.0, ""), 
                              
          ("sepexp", 0.0, "separatrix radial expansion in cm"),
          ("obots", 0.0, "bottom gap of external second separatrix in cm"),
          ("btaxp", 1.0, "toroidal magnetic field at magnetic axis in Tesla"),
          ("btaxv", 1.0, "vacuum toroidal magnetic field at magnetic axis in Tesla"),
                              
          ("aaq1", 100.0, "minor radius of q=1 surface in cm, 100 if not found"),
          ("aaq2", 100.0, "minor radius of q=2 surface in cm, 100 if not found"),
          ("aaq3", 100.0, "minor radius of q=3 surface in cm, 100 if not found"),
          ("seplim", 0.0, "> 0 for minimum gap in cm in divertor configurations, < 0 absolute value for minimum distance to external separatrix in limiter configurations"),
          
          ("rmagx", 100., "major radius in cm at magnetic axis"),
          ("zmagx", 0.0, ""),
          ("simagx", 0.0, "Poloidal flux at the magnetic axis"),
          ("taumhd", 0.0, "energy confinement time in ms"),
                          
          ("betapd", 0.0, "diamagnetic poloidal b"),
          ("betatd", 0.0, "diamagnetic toroidal b in %"),
          ("wplasmd", 0.0, "diamagnetic plasma stored energy in Joule"),
          ("diamag", 0.0, "measured diamagnetic flux in Volt-sec"),
          
          ("vloopt", 0.0, "measured loop voltage in volt"),
          ("taudia", 0.0, "diamagnetic energy confinement time in ms"),
          ("qmerci", 0.0, "Mercier stability criterion on axial q(0), q(0) > QMERCI for stability"),
          ("tavem", 0.0, "average time in ms for magnetic and MSE data"),

          # ishot > 91000
          # The next section is dependent on the EFIT version
          # New version of EFIT on 05/24/97 writes aeqdsk that includes
          # data values for parameters nsilop,magpri,nfcoil and nesum.
          (None, True, None),  # New line
          
          ("nsilop", lambda data: len(data.get("csilop", [])), "Number of flux loop signals, len(csilop)"),
          ("magpri", lambda data: len(data.get("cmpr2", [])), "Number of flux loop signals, len(cmpr2) (added to nsilop)"),
          ("nfcoil", lambda data: len(data.get("ccbrsp", [])), "Number of calculated external coil currents, len(ccbrsp)"),
          ("nesum", lambda data: len(data.get("eccurt", [])), "Number of measured E-coil currents"),

          (None, None, None),  # New line

          ("csilop", lambda data: [0.0] * data.get("nsilop", 0), "computed flux loop signals in Weber"),
          ("cmpr2", lambda data: [0.0] * data.get("magpri", 0), ""),
          ("ccbrsp", lambda data: [0.0] * data.get("nfcoil", 0), "computed external coil currents in Ampere"),
          ("eccurt", lambda data: [0.0] * data.get("nesum", 0), "measured E-coil current in Ampere"),
          
          ("pbinj", 0.0, "neutral beam injection power in Watts"),
          ("rvsin", 0.0, "major radius of vessel inner hit spot in cm"),
          ("zvsin", 0.0, "Z of vessel inner hit spot in cm"),
          ("rvsout", 0.0, "major radius of vessel outer hit spot in cm"),
          
          ("zvsout", 0.0, "Z of vessel outer hit spot in cm"),
          ("vsurfa", 0.0, "plasma surface loop voltage in volt, E EQDSK only"),
          ("wpdot", 0.0, "time derivative of plasma stored energy in Watt, E EQDSK only"),
          ("wbdot", 0.0, "time derivative of poloidal magnetic energy in Watt, E EQDSK only"),
          
          ("slantu", 0.0, ""),
          ("slantl", 0.0, ""),
          ("zuperts", 0.0, ""),
          ("chipre", 0.0, "total chi2 pressure"),
          
          ("cjor95", 0.0, ""),
          ("pp95", 0.0, "normalized P'(y) at 95% normalized poloidal flux"),
          ("ssep", 0.0, ""), 
          ("yyy2", 0.0, "Shafranov Y2 current moment"),
          
          ("xnnc", 0.0, ""),
          ("cprof", 0.0, "current profile parametrization parameter"),
          ("oring", 0.0, "not used"),
          ("cjor0", 0.0, "normalized flux surface average current density at 99% of normalized poloidal flux"),
          
          ("fexpan", 0.0, "flux expansion at x point"),
          ("qqmin", 0.0, "minimum safety factor qmin"),
          ("chigamt", 0.0, "total chi2 MSE"),
          ("ssi01", 0.0, "magnetic shear at 1% of normalized poloidal flux"),
          
          ("fexpvs", 0.0, "flux expansion at outer lower vessel hit spot"),
          ("sepnose", 0.0, "radial distance in cm between x point and external field line at ZNOSE"),
          ("ssi95", 0.0, "magnetic shear at 95% of normalized poloidal flux"),
          ("rqqmin", 0.0, "normalized radius of qmin , square root of normalized volume"),
          
          ("cjor99", 0.0, ""),
          ("cj1ave", 0.0, "normalized average current density in plasma outer 5% normalized poloidal flux region"),
          ("rmidin", 0.0, "inner major radius in m at Z=0.0"),
          ("rmidout", 0.0, "outer major radius in m at Z=0.0")]

def write(data, fh):
    """
    data  [dict] - keys are given with documentation in the `fields` list.
        Also includes
         shot [int] - The shot number 
    time  - in ms
    
    """
    # First line identification string
    # Default to date > 1997 since that format includes nsilop etc.
    fh.write("{0:11s}\n".format(data.get("header", " 26-OCT-98 09/07/98  ")))

    # Second line shot number
    fh.write(" {:d}               1\n".format(data.get("shot", 0)))

    # Third line time
    fh.write(" " + fu.f2s(data.get("time", 0.0))+"\n")

    # Fourth line 
    # time(jj),jflag(jj),lflag,limloc(jj), mco2v,mco2r,qmflag
    #   jflag = 0 if error  (? Seems to contradict example)
    #   lflag > 0 if error  (? Seems to contradict example)
    #   limloc  IN/OUT/TOP/BOT: limiter inside/outside/top/bot SNT/SNB: single null top/bottom DN: double null
    #   mco2v   number of vertical CO2 density chords
    #   mco2r   number of radial CO2 density chords
    #   qmflag  axial q(0) flag, FIX if constrained and CLC for float
    fh.write("*{:s}             {:d}                {:d} {:s}  {:d}   {:d} {:s}\n".format(fu.f2s(data.get("time", 0.0)).strip(),
                                                                                          data.get("jflag", 1),
                                                                                          data.get("lflag", 0),
                                                                                          data.get("limloc", "DN"),
                                                                                          data.get("mco2v", 0),
                                                                                          data.get("mco2r", 0),
                                                                                          data.get("qmflag", "CLC")))
    # Output data in lines of 4 values each
    with fu.ChunkOutput(fh, chunksize=4) as output:
          for key, default, description in fields:
            if callable(default):
                # Replace the default function with the value, which may depend on previously read data
                default = default(data)
                
            if key is None:
                output.newline() # Ensure on a new line
            else:
                output.write(data.get(key, default))
        

def read(fh):
    """
    Read an AEQDSK file, returning a dictionary of data
    """
    # First line label. Date.
    header = fh.readline()
    
    # Second line shot number
    shot = int(fh.readline().split()[0])
    
    # Third line time [ms]
    time = float(fh.readline())

    # Fourth line has (up to?) 9 entries
    # time(jj),jflag(jj),lflag,limloc(jj), mco2v,mco2r,qmflag
    words = fh.readline().split()
    
    # Dictionary to hold result
    data = {"header": header,
            "shot": shot,
            "time": time,
            "jflag": int(words[1]),
            "lflag": int(words[2]),
            "limloc": words[3], # e.g. "SNB"
            "mco2v": int(words[4]),
            "mco2r": int(words[5]),
            "qmflag": words[6]} # e.g. "CLC"

    # Read each value from the file, and put into variables
    values = fu.next_value(fh)
    for key, default, doc in fields:
        if key is None:
            continue  # skip
        
        if callable(default):
            default = default(data)
            
        if isinstance(default, list):
            # Read a list the same length as the default
            data[key] = [next(values) for elt in default]
        else:
            value = next(values)
            if isinstance(default, int) and not isinstance(value, int):
                # Expecting an integer, but didn't get one
                warnings.warn("Expecting an integer for '" + key + "' in aeqdsk file")
                break
            data[key] = value
    
    return data
