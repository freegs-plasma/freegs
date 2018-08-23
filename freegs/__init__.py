"""
FreeGS
======

Free boundary Grad-Shafranov solver


.. moduleauthor:: Ben Dudson <benjamin.dudson@york.ac.uk>


License
-------

Copyright 2016-2018 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.

"""

__version__ = "0.1.3"

from .equilibrium import Equilibrium

from . import jtor

from . import machine

from . import control

from .picard import solve

from .dump import OutputFile
