"""
This submodule provides the TerraModel class.  This class holds
the data contained within a single time slice of a TERRA mantle
convection simulation.
"""

import healpy as hp
import netCDF4
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors
import pickle
from . import geographic
from . import plot
from . import flow_conversion
from . import plume_detection

from .lookup_tables import TABLE_FIELDS, SeismicLookupTable, MultiTables
from .properties.profiles import prem_pressure

# Precision of coordinates in TerraModel
COORDINATE_TYPE = np.float32
# Precision of values in TerraModel
VALUE_TYPE = np.float32

# The following are the names of scalar fields which are allowed
# to be present in the _fields attribute of TerraModel, and a description
# of those fields.
_SCALAR_FIELDS = {
    "t": "Temperature field [K]",
    "c": "Scalar composition field [unitless]",
    "p": "Pressure field [GPa]",
    "vp": "P-wave velocity (elastic) [km/s]",
    "vs": "S-wave velocity (elastic) [km/s]",
    "vphi": "Bulk sound velocity (elastic) [km/s]",
    "vp_an": "P-wave velocity (anelastic) [km/s]",
    "vs_an": "S-wave velocity (anelastic) [km/s]",
    "vphi_an": "Bulk sound velocity (anelastic) [km/s]",
    "density": "Density [g/cm^3]",
    "qp": "P-wave quality factor [unitless]",
    "qs": "S-wave quality factor [unitless]",
    "visc": "Viscosity [Pas]",
    "mage": "Time of last melting [yr]",
    "sigma_z": "Radial Stress [Pa]",
    "h_cmb": "CMB heat flux [mW/m^2]",
    "he3": "He3 [mols]",
    "he4": "He4 [mols]",
    "ar36": "Ar36 [mols]",
    "ar40": "Ar40 [mols]",
    "k40": "K40 [mols]",
    "pb204": "Pb204 [mols]",
    "pb206": "Pb206 [mols]",
    "pb207": "Pb207 [mols]",
    "pb208": "Pb208 [mols]",
    "th232": "Th232 [mols]",
    "u235": "U235 [mols]",
    "u238": "U238 [mols]",
    "h2o": "H2O [mols]",
    "ppv": "pPv adundance [%]",
}

# These are 'vector' fields which contain more than one component
# at each node of the grid.
_VECTOR_FIELDS = {
    "u_xyz": "Flow field in Cartesian coordinates (three components) [m/s]",
    "u_enu": "Flow field in geographic coordinates (three components) [m/s]",
    "c_hist": "Composition histogram (_nc components) [unitles]",
}

_VECTOR_FIELD_NCOMPS = {
    "u_xyz": 3,
    "u_enu": 3,
    "c_hist": None,
}

# All fields of any kind
_ALL_FIELDS = {**_SCALAR_FIELDS, **_VECTOR_FIELDS}

# Mapping of fields to NetCDF file variable names.
# Each field name maps to one or more variables.
# Fields which don't have a defined name should not have a key in this dict.
_FIELD_NAME_TO_VARIABLE_NAME = {
    "t": ("temperature",),
    "c_hist": ("composition_fractions",),
    "u_xyz": ("velocity_x", "velocity_y", "velocity_z"),
    "vp": ("vp",),
    "vs": ("vs",),
    "vphi": ("v_bulk",),
    "vp_an": ("vp_anelastic",),
    "vs_an": ("vs_anelastic",),
    "density": ("density",),
    "visc": ("viscosity",),
    "mage": ("meltage",),
    "sigma_z": ("radial_stress",),
    "h_cmb": ("cmb_heat_flux",),
    "he3": ("He3",),
    "he4": ("He4",),
    "ar36": ("Ar36",),
    "ar40": ("Ar40",),
    "k40": ("K40",),
    "pb204": ("Pb204",),
    "pb206": ("Pb206",),
    "pb207": ("Pb207",),
    "pb208": ("Pb208",),
    "th232": ("Th232",),
    "u235": ("U235",),
    "u238": ("U238",),
    "h2o": ("H2O",),
    "ppv": ("pPv",),
}


# Mapping of variable names in NetCDF files to field names
# in this module.  This is a many to one mapping.
_VARIABLE_NAME_TO_FIELD_NAME = {}
for key, vals in _FIELD_NAME_TO_VARIABLE_NAME.items():
    for val in vals:
        _VARIABLE_NAME_TO_FIELD_NAME[val] = key

# Mapping of field name to default colour scale
_FIELD_COLOUR_SCALE = {
    field: ("turbo_r" if field.startswith("v") or field == "density" else "turbo")
    for field in _SCALAR_FIELDS
}


class NewFieldNameError(Exception):
    """
    Exception type raised when trying to use an incorrect field name
    """

    def __init__(self, field):
        self.message = f"for new field '{field}' you must also pass `label='label'`."
        super().__init__(self.message)


class FieldNameError(Exception):
    """
    Exception type raised when trying to use an incorrect field name
    """

    def __init__(self, field):
        self.message = f"'{field}' is not a valid TerraModel field name"
        super().__init__(self.message)


class PlumeFieldError(Exception):
    """
    Exception type raised when correct fields not available for plume detection
    """

    def __init__(self, field):
        self.message = (
            f"'{field}' is required as a field in the TerraModel for plume detection."
        )
        super().__init__(self.message)


class NoFieldError(Exception):
    """
    Exception type raised when trying to access a field which is not present
    """

    def __init__(self, field):
        self.message = f"Model does not contain field {field}"
        super().__init__(self.message)


class NoSphError(Exception):
    """
    Exception type raised when trying to access spherical harmonics which have
    not yet been calculated.
    """

    def __init__(self, field):
        self.message = f"Spherical hamronic coefficients for {field} have not yet been calculated, use `calc_spherical_harmonics`"
        super().__init__(self.message)


class FieldDimensionError(Exception):
    """
    Exception type raised when trying to set a field when the dimensions
    do not match the coordinates in the model
    """

    def __init__(self, model, array, name=""):
        self.message = (
            f"Field array {name} has incorrect first two dimensions. "
            + f"Expected {(model._nlayers, model._npts)}; got {array.shape[0:2]}"
        )
        super().__init__(self.message)


class VersionError(Exception):
    """
    Exception type raised when old version of unversioned netCDF files
    are passed in.
    """

    def __init__(self, version):
        self.message = f"NetCDF file version '{version}' is not supported. Please convert with convert_files.convert"
        super().__init__(self.message)


class SizeError(Exception):
    """
    Exception type raised when input param is wrong shape
    """

    def __init__(self):
        self.message = f"Input params lons, lats, field mut be of same length"
        super().__init__(self.message)


class LayerMethodError(Exception):
    """
    Exception type raised when trying to call incompatible TerraModel method for
    a TerraModelLayer object
    """

    def __init__(self, name):
        self.message = f"Method {name} is incompatible with TerraModelLayer objects"
        super().__init__(self.message)


class FileFormatError(Exception):
    """
    Exception type raised when a netCDF file is not correctly formatted.
    """

    def __init__(self, file, name, expected_value, actual_value):
        self.message = (
            f"Unexpected value for {name} in file '{file}'. "
            + f"Expected {expected_value} but got {actual_value}"
        )
        super().__init__(self.message)


class TerraModel:
    """
    Class holding a TERRA model at a single point in time.

    A TerraModel contains the coordinates of each lateral point, the
    radii of each layer, and zero or more fields at each of these points.

    Fields are either 2D (for scalar fields like temperature) or 3D
    (for multicomponent arrays like flow velocity) NumPy arrays.

    There are two kinds of methods which give you information about
    the contents of a TerraModel:

    1. Methods starting with ``get_`` return a reference to something held
       within the model.  Hence things returned
       by ``get``ters can be modified and the TerraModel instance is also
       modified.  For example, ``get_field("vp")`` returns the array containing
       the P-wave velocity for the whole model, and you can update any values
       within it.  You should **not** change the shape or length of any arrays
       returned from ``get_`` methods, as this will make the internal state
       of a TerraModel inconsistent.

    2. Methods which are simply a noun (like ``number_of_compositions``)
       return a value and this cannot be used to change the model.

    Model coordinates
    -----------------
    TerraModels have a number of layers, and within each layer there are
    a number of lateral points, of which there are always the same number.
    Points on each layer at the same point index always have the same
    coordinates.

    Use ``TerraModel.get_lateral_points()`` to obtain the longitude and
    latitude of the lateral points.

    ``TerraModel.get_radii()`` returns instead the radii (in km) of
    each of the layers in the model.

    Model fields
    ------------
    Models fields (as returned by ``TerraModel.get_field``) are simply
    NumPy arrays.  The first index is the layer index, and the second
    is the point index.

    As an example, if for a model ``m`` you obtain the temperature field
    with a call ``temp = m.get_field("t")``, the lateral coordinates
    ``lon, lat = m.get_lateral_points()`` and the radii ``r = m.get_radii()``,
    the temperature at ``lon[ip], lat[ip]`` and radius ``r[ir]`` is given
    by ``temp[ir,ip]``.

    Nearest neighbours
    ------------------
    The nearest lateral point index to an arbitrary geographic location
    can be obtained by ``m.nearest_index(lon, lat)``, whilst the nearest
    ``n`` neighbours can be obtained with ``m.nearest_indices(lon, lat, n)``.
    ``m.nearest_neighbours(lon, lat, n)`` also returns the distances to
    each near-neighbour.
    """

    def __init__(
        self,
        lon,
        lat,
        r,
        surface_radius=None,
        fields={},
        c_histogram_names=None,
        c_histogram_values=None,
        lookup_tables=None,
        pressure_func=None,
    ):
        """
        Construct a TerraModel.

        The basic TerraModel constructor requires the user to supply a set
        of radii ``r`` in km, and a set of lateral point coordinates as
        two separate arrays of longitude and latitude (in degrees).
        These then define the 2D grid on which the field values are defined.

        Fields are passed in as a dict mapping field name onto either a
        2D array (for scalar fields like temperature), or a 3D array
        for multicomponent fields, like flow velocity or composition histograms.

        Composition histograms enable one to translate from temperature,
        pressure and composition to seismic velocity by assuming the
        model contains a mechanical mixture of some number of different
        chemical components, and performing Voigt averaging over the
        elastic parameters predicted for these compositions, weighted
        by their proportion.  The field ``"c_hist"`` holds a 3D array
        whose last dimension gives the proportion of each component.
        At each depth and position, the proportions must sum to unity,
        and this is checked in the constructor.

        When using a composition histogram, you may pass the
        ``c_histogram_names`` argument, giving the name of each component,
        and ``c_histogram_values``, giving the composition value of each.
        The ith component name corresponds to the proportion of the ith
        slice of the last dimension of the ``"c_hist"`` array.

        Seismic lookup tables should be passed to this constructor
        when using multicomponent composition histograms as a ``dict``
        whose keys are the names in ``c_histogram_name`` and whose values
        are instances of ``terratools.lookup_tables.SeismicLookupTable``.
        Alternatively, ``lookup_tables`` may be an instance of
        ``terratools.lookup_tables.MultiTables``.

        :param lon: Position in longitude of lateral points (degrees)
        :param lat: Position in latitude of lateral points (degrees).  lon and
            lat must be the same length
        :param r: Radius of nodes in radial direction.  Must increase monotonically.
        :param surface_radius: Radius of surface of the model in km, if not the
            same as the largest value of ``r``.  This may be useful
            when using parts of models.
        :param fields: dict whose keys are the names of field, and whose
            values are numpy arrays of dimension (nlayers, npts) for
            scalar fields, and (nlayers, npts, ncomps) for a field
            with ``ncomps`` components at each point
        :param c_histogram_names: The names of each composition of the
            composition histogram, passed as a ``c_hist`` field
        :param c_histogram_values: The values of each composition of the
            composition histogram, passed as a ``c_hist`` field
        :param lookup_tables: A dict mapping composition name to the file
            name of the associated seismic lookup table; or a
            ``lookup_tables.MultiTables``
        :param pressure_func: Function which takes a single argument
            (the radius in km) and returns pressure in Pa.  By default
            pressure is taken from PREM.  The user is responsible for
            ensuring that ``pressure_func`` accepts all values in the radius
            range of the model.
        """

        nlayers = len(r)
        self._nlayers = nlayers

        npts = len(lon)
        if len(lat) != npts:
            raise ValueError("length of lon and lat must be the same")

        # Number of lateral points
        self._npts = npts
        # Longitude and latitude in degrees of lateral points
        self._lon = np.array(lon, dtype=COORDINATE_TYPE)
        self._lat = np.array(lat, dtype=COORDINATE_TYPE)
        # Radius of each layer
        self._radius = np.array(r, dtype=COORDINATE_TYPE)

        # Check for monotonicity of radius
        if not np.all(self._radius[1:] - self._radius[:-1] > 0):
            raise ValueError("radii must increase or decrease monotonically")

        # Surface radius
        self._surface_radius = (
            self._radius[-1] if surface_radius is None else surface_radius
        )
        if self._surface_radius < self._radius[-1]:
            raise ValueError(
                f"surface radius given ({surface_radius} km) is "
                + f"less than largest model radius ({self._radius[-1]} km)"
            )

        # Fit a nearest-neighbour search tree
        self._knn_tree = _fit_nn_tree(self._lon, self._lat)

        # The names of the compositions if using a composition histogram approach
        self._c_hist_names = c_histogram_names

        # The values of the compositions if using a composition histogram approach
        # This is not currently used, but will be shown if present
        self._c_hist_values = c_histogram_values

        # A set of lookup tables
        self._lookup_tables = lookup_tables

        # All the fields are held within _fields, either as scalar or
        # 'vector' fields.
        self._fields = {}

        # Use PREM for pressure if a function is not supplied
        if pressure_func is None:
            self._pressure_func = self.pressure_func_prem
        else:
            self._pressure_func = pressure_func

        # Check fields have the right shape and convert
        for key, val in fields.items():
            array = np.array(val, dtype=VALUE_TYPE)

            if _is_scalar_field(key):
                self._check_field_shape(val, key, scalar=True)

            elif _is_vector_field(key):
                expected_ncomps = _expected_vector_field_ncomps(key)
                if expected_ncomps is None:
                    self._check_field_shape(val, key, scalar=False)
                else:
                    if array.shape != (nlayers, npts, expected_ncomps):
                        raise FieldDimensionError(self, val, key)

                # Special check for composition histograms
                if key == "c_hist":
                    if not _compositions_sum_to_one(array):
                        sums = np.sum(array)
                        min, max = np.min(sums), np.max(sums)
                        raise ValueError(
                            "composition proportions must sum to one for each point"
                            f" (range is [{min}, {max}])"
                        )

            else:
                raise FieldNameError(key)

            self.set_field(key, array)

        # Check lookup table arguments
        if self._lookup_tables is not None and self._c_hist_names is None:
            raise ValueError(
                "must pass a list of composition histogram names "
                + "as c_histogram_names if passing lookup_tables"
            )

        if self._c_hist_names is not None and self._lookup_tables is not None:
            if isinstance(self._lookup_tables, MultiTables):
                lookup_tables_keys = self._lookup_tables._lookup_tables.keys()
            else:
                lookup_tables_keys = self._lookup_tables.keys()

            if sorted(self._c_hist_names) != sorted(lookup_tables_keys):
                raise ValueError(
                    "composition names in c_histogram_names "
                    + f"({self._c_hist_names}) are not "
                    + "the same as the keys in lookup_tables "
                    + f"({self._lookup_tables.keys()})"
                )

            # Check composition values if given
            if self._c_hist_values is not None:
                if len(self._c_hist_values) != len(self._c_hist_names):
                    raise ValueError(
                        "length of c_histogram_values must be "
                        + f"{len(self._c_hist_names)}, the same as for "
                        + "c_histogram_names.  Is actually "
                        + f"{len(self._c_hist_value)}."
                    )

            # Convert to MultiTables if not already
            if not isinstance(self._lookup_tables, MultiTables):
                self._lookup_tables = MultiTables(self._lookup_tables)

    def pressure_func_prem(self, r):
        _pressure = prem_pressure()
        return _pressure(1000 * self.to_depth(r))

    def __repr__(self):
        return f"""TerraModel:
           number of radii: {self._nlayers}
             radius limits: {(float(np.min(self._radius)), float(np.max(self._radius)))}
  number of lateral points: {self._npts}
                    fields: {[name for name in self.field_names()]}
         composition names: {self.get_composition_names()}
        composition values: {self.get_composition_values()}
         has lookup tables: {self.has_lookup_tables()}"""

    def add_lookup_tables(self, lookup_tables):
        """
        Add set of lookup tables to the model.  The tables must have the
        same keys as the model has composition names.

        :param lookup_tables: A ``lookup_tables.MultiTables`` containing
            a lookup table for each composition in the model.
        """
        if not isinstance(lookup_tables, MultiTables):
            raise ValueError(
                "Tables must be provided as a lookup_tables.MultiTables object"
            )

        table_keys = lookup_tables._lookup_tables.keys()
        if sorted(self.get_composition_names()) != sorted(table_keys):
            raise ValueError(
                "Tables must have the same keys as the model compositions. "
                + f"Got {table_keys}; need {self.get_composition_names()}"
            )

        self._lookup_tables = lookup_tables

    def field_names(self):
        """
        Return the names of the fields present in a TerraModel.

        :returns: list of the names of the fields present.
        """
        return self._fields.keys()

    def evaluate(self, lon, lat, r, field, method="triangle", depth=False):
        """
        Evaluate the value of field at radius r km, latitude lat degrees
        and longitude lon degrees.

        Note that if r is below the bottom of the model the value at the
        lowest layer is returned; and likewise if r is above the top
        of the model, the value at the highest layer is returned.

        There are two evaluation methods:

        #. ``"triangle"``: Finds the triangle surrounding the point of
           interest and performs interpolation between the values at
           each vertex of the triangle
        #. ``"nearest"``: Just returns the value of the closest point of
           the TerraModel

        In either case, linear interpolation is performed between the two
        surrounding layers of the model.

        :param lon: Longitude in degrees of point of interest
        :param lat: Latitude in degrees of points of interest
        :param r: Radius in km of point of interest
        :param field: String giving the name of the field of interest
        :param method: String giving the name of the evaluation method; a
            choice of 'triangle' (default) or 'nearest'.
        :param depth: If True, treat r as a depth rather than a radius
        :returns: value of the field at that point
        """
        _check_field_name(field)
        self._check_has_field(field)

        if method not in ("triangle", "nearest"):
            raise ValueError("method must be one of 'triangle' or 'nearest'")

        isscalar = False
        if np.isscalar(lon):
            isscalar = True
            lon = np.array([lon])
        if np.isscalar(lat):
            lat = np.array([lat])
        if np.isscalar(r):
            r = np.array([r])

        # Convert lists to numpy arrays
        lon = np.array(lon)
        lat = np.array(lat)
        r = np.array(r)

        radii = self.get_radii()

        if depth:
            r = self.to_radius(r)

        lons, lats = self.get_lateral_points()
        array = self.get_field(field)

        # Find bounding layers
        ilayer1, ilayer2 = _bounding_indices(r, radii)
        r1, r2 = radii[ilayer1], radii[ilayer2]

        if method == "triangle":
            # Get three nearest points, which should be the surrounding
            # triangle
            idx1, idx2, idx3 = self.nearest_indices(lon, lat, 3).T

            # For the two layers, laterally interpolate the field
            # Note that this relies on NumPy's convention on indexing, where
            # indexing an array with fewer indices than dimensions acts
            # as if the missing, trailing dimensions were indexed with `:`.
            # (E.g., `np.ones((1,2,3))[0,0]` is `[1., 1., 1.]`.)
            val_layer1 = geographic.triangle_interpolation(
                lon,
                lat,
                lons[idx1],
                lats[idx1],
                array[ilayer1, idx1],
                lons[idx2],
                lats[idx2],
                array[ilayer1, idx2],
                lons[idx3],
                lats[idx3],
                array[ilayer1, idx3],
            )

            val_layer2 = geographic.triangle_interpolation(
                lon,
                lat,
                lons[idx1],
                lats[idx1],
                array[ilayer2, idx1],
                lons[idx2],
                lats[idx2],
                array[ilayer2, idx2],
                lons[idx3],
                lats[idx3],
                array[ilayer2, idx3],
            )

        elif method == "nearest":
            index = self.nearest_index(lon, lat)
            val_layer1 = array[ilayer1, index]
            val_layer2 = array[ilayer2, index]

        # Linear interpolation between the adjacent layers
        mask = ilayer1 != ilayer2
        value = val_layer1

        if sum(mask) > 0:
            value[mask] = (
                (r2[mask] - r[mask]) * val_layer1[mask]
                + (r[mask] - r1[mask]) * val_layer2[mask]
            ) / (r2[mask] - r1[mask])

        if isscalar:
            return value[0]
        else:
            return value

    def evaluate_from_lookup_tables(
        self, lon, lat, r, fields=TABLE_FIELDS, method="triangle", depth=False
    ):
        """
        Evaluate the value of a field at radius ``r`` km, longitude
        ``lon`` degrees and latitude ``lat`` degrees by using
        the composition or set of composition proportions at that point
        and a set of seismic lookup tables to convert to seismic
        properties.

        :param lon: Longitude in degrees of point of interest
        :param lat: Latitude in degrees of points of interest
        :param r: Radius in km of point of interest
        :param fields: Iterable of strings giving the names of the
            field of interest, or a single string.  If a single string
            is passed in, then a single value is returned.  By default,
            all fields are returned.
        :param method: String giving the name of the evaluation method; a
            choice of ``'triangle'`` (default) or ``'nearest'``.
        :param depth: If ``True``, treat ``r`` as a depth rather than a radius
        :returns: If a set of fields are passed in, or all are requested
            (the default), a ``dict`` mapping the names of the fields to
            their values.  If a single field is requested, the value
            of that field.
        """
        # Check that the fields we have requested are all valid
        if isinstance(fields, str):
            if fields not in TABLE_FIELDS:
                raise ValueError(
                    f"Field {fields} is not a valid "
                    + f"seismic property. Must be one of {TABLE_FIELDS}."
                )
        else:
            for field in fields:
                if field not in TABLE_FIELDS:
                    raise ValueError(
                        f"Field {field} is not a valid "
                        + f"seismic property. Must be one of {TABLE_FIELDS}."
                    )

        # Convert to radius now
        if depth:
            r = self.to_radius(r)

        # Composition names
        c_names = self.get_composition_names()

        # Get composition proportions and temperature in K
        c_hist = self.evaluate(lon, lat, r, "c_hist", method=method)
        t = self.evaluate(lon, lat, r, "t", method=method)

        # Pressure for this model in Pa
        p = self._pressure_func(r)

        # Evaluate chosen things from lookup tables
        fraction_dict = {c_name: fraction for c_name, fraction in zip(c_names, c_hist)}
        if isinstance(fields, str):
            value = self._lookup_tables.evaluate(p, t, fraction_dict, fields)
            return value
        else:
            values = {
                field: self._lookup_tables.evaluate(p, t, fraction_dict, field)
                for field in fields
            }
            return values

    def write_pickle(self, filename):
        """
        Save the terra model as a python pickle format with the
        given filename.

        :param filename: filename to save terramodel to.
        :type filename: str

        :return: nothing
        """
        f = open(filename, "wb")
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        return

    def set_field(self, field, values):
        """
        Create a new field within a TerraModel from a predefined array,
        replacing any existing field data.

        :param field: Name of field
        :type field: str
        :param values: numpy.array containing the field.  For scalars it
            should have dimensions corresponding to (nlayers, npts),
            where nlayers is the number of layers and npts is the number
            of lateral points.  For multi-component fields, it should
            have dimensions (nlayers, npts, ncomps), where ncomps is the
            number of components
        :type values: numpy.array
        """
        _check_field_name(field)
        array = np.array(values, dtype=VALUE_TYPE)
        self._check_field_shape(array, field, scalar=_is_scalar_field(field))
        self._fields[field] = np.array(array, dtype=VALUE_TYPE)

    def new_field(self, name, ncomps=None, label=None):
        """
        Create a new, empty field with key ``name``.

        :param name: Name of new field.
        :param ncomps: Number of components for a multicomponent field.
        :returns: the new field
        """
        global _SCALAR_FIELDS, _VECTOR_FIELDS, _ALL_FIELDS, _VECTOR_FIELD_NCOMPS
        if ncomps is not None and ncomps < 1:
            raise ValueError(f"ncomps cannot be less than 1 (is {ncomps})")

        is_vector = _is_vector_field(name)
        ncomps_expected = _expected_vector_field_ncomps(name) if is_vector else None
        if not is_vector and ncomps is not None and ncomps > 1:
            is_vector = True
            ncomps_expected = ncomps

        if label == None:
            _check_new_field_name(name)
        elif is_vector:
            _VECTOR_FIELDS[name] = label
            _VECTOR_FIELD_NCOMPS[name] = ncomps
        else:
            _SCALAR_FIELDS[name] = label

        _ALL_FIELDS = {**_SCALAR_FIELDS, **_VECTOR_FIELDS}

        nlayers = self._nlayers
        npts = self._npts

        if is_vector:
            # For a vector field, either we have not set ncomps and we use the
            # expected number, or there is no expected number and we use what is
            # passed in.  Fields without an expected number of components
            # cannot be made unless ncomps is set.
            if ncomps_expected is None:
                if ncomps is None:
                    raise ValueError(
                        f"Field {name} has no expected number "
                        + "of components, so ncomps must be passed"
                    )
                self.set_field(name, np.zeros((nlayers, npts), dtype=VALUE_TYPE))
            else:
                if ncomps is not None:
                    if ncomps != ncomps_expected:
                        raise ValueError(
                            f"Field {name} should have "
                            + f"{ncomps_expected} fields, but {ncomps} requested"
                        )
                else:
                    ncomps = ncomps_expected

            self.set_field(name, np.zeros((nlayers, npts, ncomps), dtype=VALUE_TYPE))

        else:
            # Scalar field; should not ask for ncomps at all
            if ncomps is not None:
                raise ValueError(f"Scalar field {name} cannot have {ncomps} components")
            self.set_field(name, np.zeros((nlayers, npts), dtype=VALUE_TYPE))

        return self.get_field(name)

    def has_field(self, field):
        """
        Return True if this TerraModel contains a field called ``field``.

        :param field: Name of field
        :returns: True if field is present, and False otherwise
        """
        return field in self._fields.keys()

    def has_lookup_tables(self):
        """
        Return `True` if this TerraModel contains thermodynamic lookup
        tables used to convert temperature, pressure and composition
        into seismic properties.

        :returns: `True` is the model has tables, and `False` otherwise
        """
        return self._lookup_tables is not None

    def get_field(self, field):
        """
        Return the array containing the values of field in a TerraModel.

        :param field: Name of the field
        :returns: the field of interest as a numpy.array
        """
        self._check_has_field(field)
        return self._fields[field]

    def get_spherical_harmonics(self, field):
        """
        Return the spherical harmonic coefficients and power per l (and maps if calculated) or raise NoSphError

        :param field: Name of field
        :type field: str

        :returns: dictionary containing spherical harmonic coefficients and power per l
                  at each layer
        """
        if self._check_has_sph(field):
            return self._sph[field]

    def _check_has_sph(self, field):
        """
        Return True or False if spherical harmonics for the given field
        have been calcualted or not.
        """
        if field in self._sph.keys():
            return True
        else:
            raise NoSphError(field)

    def _check_has_field(self, field):
        """
        If field is not present in this model, raise a NoFieldError
        """
        if not self.has_field(field):
            raise NoFieldError(field)

    def _check_field_shape(self, array, name, scalar=True):
        """
        If the first two dimensions of array are not (nlayers, npts),
        raise an error, and likewise raise an error if the array is
        not rank-2 or rank-3.

        :raises: FieldDimensionError
        """
        if len(array.shape) not in (2, 3):
            raise FieldDimensionError(self, array, name)
        if (scalar and array.shape != (self._nlayers, self._npts)) or (
            not scalar and array.shape[0:2] != (self._nlayers, self._npts)
        ):
            raise FieldDimensionError(self, array, name)

    def number_of_compositions(self):
        """
        If a model contains a composition histogram field ('c_hist'),
        return the number of compositions; otherwise return None.

        :returns: number of compositions or None
        """
        if self.has_field("c_hist"):
            return self.get_field("c_hist").shape[2]
        else:
            return None

    def get_composition_names(self):
        """
        If a model contains a composition histogram field ('c_hist'),
        return the names of the compositions; otherwise return None.

        :returns: list of composition names
        """
        if self.has_field("c_hist"):
            return self._c_hist_names
        else:
            return None

    def get_composition_values(self):
        """
        If a model contains a composition histogram field ('c_hist'),
        return the values of the compositions; otherwise return None.

        :returns: list of composition values
        """
        if self.has_field("c_hist"):
            return self._c_hist_values
        else:
            return None

    def get_lateral_points(self):
        """
        Return two numpy.arrays, one each for the longitude and latitude
        (in degrees) of the lateral points of each depth slice of the fields
        in a model.

        :returns: (lon, lat) in degrees
        """
        return self._lon, self._lat

    def get_lookup_tables(self):
        """
        Return the `terratools.lookup_tables.MultiTables` object which
        holds the model's lookup tables if present, and `None` otherwise.

        :returns: the lookup tables, or `None`.
        """
        if self.has_lookup_tables():
            return self._lookup_tables
        else:
            return None

    def get_radii(self):
        """
        Return the radii of each layer in the model, in km.

        :returns: radius of each layer in km
        """
        return self._radius

    def mean_radial_profile(self, field):
        """
        Return the mean of the given field at each radius.

        :param field: name of field
        :type field: str

        :returns profile: mean values of field at each radius.
        :rtype profile: 1d numpy array of floats.
        """

        # shape is [nradii, npoints]
        field_values = self.get_field(field)

        # take mean across the radii layers
        profile = np.mean(field_values, axis=1)

        return profile

    def radial_profile(self, lon, lat, field, method="nearest"):
        """
        Return the radial profile of the given field
        at a given longitude and latitude point.

        :param lon: Longitude at which to get radial profile.
        :type lon: float

        :param lat: Latitude at which to get radial profile.
        :type lat: float

        :param field: Name of field.
        :type field: str

        :param method: Method by which the lateral points are evaluated.
            if ``method`` is ``"nearest"`` (the default), the nearest
            point to (lon, lat) is found.  If ``method`` is ``"triangle"``,
            then triangular interpolation is used to calculate the value
            of the field at the exact (lon, lat) point.
        :type method: str

        :returns profile: values of field for each radius
                          at a given longitude and latitude.  The radii
                          of each point can be obtained using
                          ``TerraModel.get_radii()``.
        :rtype profile: 1d numpy array of floats.
        """

        if method == "nearest":
            i = self.nearest_index(lon, lat)
            # shape is [nradii, npoints]
            field_values = self.get_field(field)
            # Ensure we return a copy, since this isn't a 'get_'ter
            profile = field_values[:, i].copy()

        else:
            radii = self.get_radii()
            lons = lon * np.ones_like(radii)
            lats = lat * np.ones_like(radii)
            profile = self.evaluate(lons, lats, radii, field, method=method)

        return profile

    def nearest_index(self, lon, lat):
        """
        Return the index or indices of the lateral point(s) nearest to the
        one or more points supplied.  lon and lat may either both be a scalar or
        both an array of points; behaviour is undefined if a mix is
        provided.

        :param lon: Longitude of point(s) of interest (degrees)
        :param lat: Latitude of point(s) of interest (degrees)
        :returns: the index or indices of the nearest lateral point.
            This is a scalar for scalar input, and an array for array input.
        """
        scalar_input = False
        if np.isscalar(lon) and np.isscalar(lat):
            scalar_input = True

        indices = self.nearest_indices(lon, lat, 1)
        if scalar_input:
            return indices[0]
        else:
            return np.array([idx[0] for idx in indices])

    def nearest_indices(self, lon, lat, n):
        """
        Return the indices of the lateral point(s) nearest to the
        one or more points supplied.  lon and lat may either both be a scalar or
        both an array of points; behaviour is undefined if a mix is
        provided.

        :param lon: Longitude of point(s) of interest (degrees)
        :param lat: Latitude of point(s) of interest (degrees)
        :param n: Number of nearest neighbours to find
        :returns: the indices of the nearest n lateral points.
            This is vector for scalar input, and a vector of vectors for array
            input.
        """
        if n < 1:
            raise ValueError("n must be 1 or more")

        scalar_input = False
        if np.isscalar(lon) and np.isscalar(lat):
            scalar_input = True

        indices, _ = self.nearest_neighbors(lon, lat, n)

        return indices

    def nearest_neighbors(self, lon, lat, n):
        """
        Return the indices of the lateral point(s) nearest to the
        one or more points supplied, and the distances from the test point
        to each point.  lon and lat may either both be a scalar or
        both an array of points; behaviour is undefined if a mix is
        provided.

        Distances are in radians about the centre of the sphere; to
        convert to great circle distances, multiply by the radius of
        interest.

        :param lon: Longitude of point(s) of interest (degrees)
        :param lat: Latitude of point(s) of interest (degrees)
        :param n: Number of nearest neighbours to find
        :returns: (indices, distances), where the first item contains the
            indices of the nearest n lateral points and the second item gives
            the distances in radians about the centre on the sphere on
            which the points all lie.
            These are vectors for scalar input, and a vector of vectors for array
            input.
        """
        if n < 1:
            raise ValueError("n must be 1 or more")

        scalar_input = False

        if np.isscalar(lon) and np.isscalar(lat):
            scalar_input = True
            lon = np.array([lon])
            lat = np.array([lat])
        elif len(lon) != len(lat):
            raise ValueError("lon and lat must be the same length")

        latlon = np.array([lat, lon]).T
        latlon_radians = np.radians(latlon)
        distances, indices = self._knn_tree.kneighbors(latlon_radians, n_neighbors=n)

        if scalar_input:
            return indices[0], distances[0]
        else:
            return indices, distances

    def nearest_layer(self, radius, depth=False):
        """
        Find the layer nearest to the given radius.

        :param radius: Radius of interest in km.
        :param depth: If True, treat input radius as a depth instead,
            and return index and depth rather than index and radius.
        :returns: layer index and radius of layer in km if depth is False
            (the default); otherwise return layer index and depth in km
        """
        radii = self.get_radii()
        if depth:
            radius = self.to_radius(radius)

        index = _nearest_index(radius, radii)

        if depth:
            return index, self.to_depth(radii[index])
        else:
            return index, radii[index]

    def pressure_at_radius(self, r):
        """
        Evaluate the pressure in the model at a radius of ``r`` km.

        :param r: Radius in km
        :returns: Pressure in GPa
        """
        return self._pressure_func(r)

    def to_depth(self, radius):
        """
        Convert a radius in km to a depth in km.

        :param radius: Radius in km
        :returns: Depth in km
        """
        return self._surface_radius - radius

    def to_radius(self, depth):
        """
        Convert a radius in km to a depth in km.

        :param depth: Depth in km
        :returns: Radius in km
        """
        return self._surface_radius - depth

    def calc_spherical_harmonics(
        self,
        field,
        nside=2**6,
        lmax=16,
        savemap=False,
        use_pixel_weights=False,
        v_field_ind=None,
    ):
        """
        Function to calculate spherical harmonic coefficients for given global field.
        Model is re-gridded to an equal area healpix grid of size nside (see
        https://healpix.sourceforge.io/ for details) and then expanded to spherical
        harmonic coefficients up to degree lmax, with pixels being uniformally weighted
        by 4pi/n_pix (see https://healpy.readthedocs.io/en/latest/index.html for details).

        :param field: input field
        :type field: str

        :param nside: healpy param, number of sides for healpix grid, power
                      of 2 less than 2**30 (default 2**6)
        :type nside: int (power of 2)

        :param lmax: maximum spherical harmonic degree (default 16)
        :type lmax: int

        :param savemap: Default (``False``) do not save the healpix map
        :type savemap: bool
        """

        if _is_vector_field(field) and v_field_ind == None:
            print("v_field_ind not supplied, defaulting to 0")
            v_field_ind = 0

        field_values = (
            self.get_field(field)
            if _is_scalar_field(field)
            else self.get_field(field)[:, :, v_field_ind]
        )

        lons, lats = self.get_lateral_points()

        # Check that lon, lat and field are same length
        if (
            len(lons) != len(lats)
            or len(lats) != field_values.shape[1]
            or len(lons) != field_values.shape[1]
        ):
            raise (SizeError)

        nr = len(self.get_radii())
        hp_ir = {}
        for r in range(nr):
            hpmap = _pixelise(field_values[r, :], nside, lons, lats)
            power_per_l = hp.sphtfunc.anafast(hpmap, lmax=lmax)
            hp_coeffs = hp.sphtfunc.map2alm(
                hpmap, lmax=lmax, use_pixel_weights=use_pixel_weights
            )
            if savemap:
                hp_ir[r] = {
                    "map": hpmap,
                    "power_per_l": power_per_l,
                    "coeffs": hp_coeffs,
                }
            else:
                hp_ir[r] = {"power_per_l": power_per_l, "coeffs": hp_coeffs}
        try:
            if _is_scalar_field(field):
                self._sph[field] = hp_ir
            elif _is_vector_field(field):
                self._sph[f"{field}{v_field_ind}"] = hp_ir
        except:
            self._sph = {}
            if _is_scalar_field(field):
                self._sph[field] = hp_ir
            elif _is_vector_field(field):
                self._sph[f"{field}{v_field_ind}"] = hp_ir

    def plot_hp_map(
        self,
        field,
        index=None,
        radius=None,
        depth=False,
        fig=None,
        ax=None,
        return_cbar=False,
        nside=2**6,
        title=None,
        delta=None,
        extent=(-180, 180, -90, 90),
        method="nearest",
        v_field_ind=None,
        show=True,
        **subplots_kwargs,
    ):
        """
        Create heatmap of a field recreated from the spherical harmonic coefficients
        :param field: name of field as created using ``data.calc_spherical_harmonics()``
        :type field: str

        :param index: index of layer to plot
        :type index: int

        :param radius: radius to plot (nearest model radius is shown)
        :type radius: float

        :param depth: interpret radius as depth
        :type depth: bool

        :param fig: figure handle
        :type fig: matplotlib.figure.Figure

        :param ax: axis handle
        :type ax: matplotlib.axes._axes.Axes

        :param return_cbar: flag to return colorbar
        :type ax: bool

        :param nside: healpy param, number of sides for healpix grid, power
            of 2 less than 2**30 (default 2**6)
        :type nside: int (power of 2)

        :param title: name of field to be included in title
        :type title: str

        :param delta: Grid spacing of plot in degrees
        :type delta: float

        :param extent: Tuple giving the longitude and latitude extent of
            plot, in the form (min_lon, max_lon, min_lat, max_lat), all
            in degrees
        :type extent: tuple of length 4

        :param method: May be one of: "nearest" (plot nearest value to each
            plot grid point); or "mean" (mean value in each pixel)
        :type method: str

        :param show: If True (the default), show the plot
        :type show: bool

        :param **subplots_kwargs: Extra keyword arguments passed to
            `matplotlib.pyplot.subplots`

        :returns: figure, axis and colourbar handles
        """
        if _is_vector_field(field) and v_field_ind == None:
            print("v_field_ind not supplied, defaulting to 0")
            v_field_ind = 0

        if radius is None and index is None:
            raise ValueError("Either radius or index must be given")
        if index is None:
            layer_index, layer_radius = self.nearest_layer(radius, depth)
        else:
            radii = self.get_radii()
            nlayers = len(radii)
            if index < 0 or index >= nlayers:
                raise ValueError(f"index must be between 0 and {nlayers}")

            layer_index = index
            layer_radius = radii[index]
        if _is_scalar_field(field):
            dat = self.get_spherical_harmonics(field)[layer_index]["coeffs"]
            lmax = (
                len(self.get_spherical_harmonics(field)[layer_index]["power_per_l"]) - 1
            )
        elif _is_vector_field(field):
            dat = self.get_spherical_harmonics(f"{field}{v_field_ind}")[layer_index][
                "coeffs"
            ]
            lmax = (
                len(
                    self.get_spherical_harmonics(f"{field}{v_field_ind}")[layer_index][
                        "power_per_l"
                    ]
                )
                - 1
            )
        npix = hp.nside2npix(nside)
        radii = self.get_radii()
        rad = radii[layer_index]
        #        lmax = len(self.get_spherical_harmonics(field)[layer_index]["power_per_l"]) - 1
        hp_remake = hp.sphtfunc.alm2map(dat, nside=nside, lmax=lmax)

        lon, lat = hp.pix2ang(nside, np.arange(0, npix), lonlat=True)
        mask = lon > 180.0
        lon2 = (lon - 360) * mask
        lon = lon2 + lon * ~mask
        if title == None:
            label = field if _is_scalar_field(field) else f"{field}{v_field_ind}"
        else:
            label = title

        returned = plot.layer_grid(
            lon,
            lat,
            rad,
            hp_remake,
            delta=delta,
            extent=extent,
            label=label,
            fig=fig,
            ax=ax,
            return_cbar=return_cbar,
        )

        fig = returned[0]
        ax = returned[1]
        cbar = returned[2] if return_cbar else None

        if depth:
            ax.set_title(f"Depth = {int(layer_radius)} km")
        else:
            ax.set_title(f"Radius = {int(layer_radius)} km")

        if show:
            fig.show()

        if return_cbar:
            return fig, ax, cbar
        else:
            return fig, ax

    def plot_spectral_heterogeneity(
        self,
        field,
        title=None,
        saveplot=False,
        savepath=None,
        fig=None,
        ax=None,
        return_cbar=False,
        lmin=1,
        lmax=None,
        lyrmin=1,
        lyrmax=-1,
        v_field_ind=None,
        show=True,
        **kwargs,
    ):
        """
        Plot spectral heterogenity maps of the given field, that is the power
        spectrum over depth.
        :param field: name of field to plot as created using model.calc_spherical_harmonics()
        :type field: str

        :param title: title of plot
        :type title: str

        :param saveplot: flag to save an image of the plot to file
        :type saveplot: bool

        :param savepath: path under which to save plot to
        :type savepath: str

        :param fig: figure handle
        :type fig: matplotlib.figure.Figure

        :param ax: axis handle
        :type ax: matplotlib.axes._axes.Axes

        :param return_cbar: flag to return colorbar
        :type return_cbar: bool

        :param lmin: minimum spherical harmonic degree to plot (default=1)
        :type lmin: int

        :param lmax: maximum spherical harmonic degree to plot (default to plot all)
        :type lmax: int

        :param lyrmin: min layer to plot (default omits boundary)
        :type lyrmin: int

        :param lyrmax: max layer to plot (default omits boundary)
        :type lyrmax: int

        :param show: if True (default) show the plot
        :type show: bool

        :param **subplots_kwargs: Extra keyword arguments passed to
            `matplotlib.pyplot.subplots`

        :returns: figure, axis, colourbar handles
        """
        if _is_vector_field(field):
            if v_field_ind == None:
                print("v_field_ind not supplied, defaulting to 0")
                v_field_ind = 0
            dat = self.get_spherical_harmonics(f"{field}{v_field_ind}")
        else:
            dat = self.get_spherical_harmonics(f"{field}")
        nr = len(dat)
        lmax_dat = len(dat[0]["power_per_l"]) - 1
        powers = np.zeros((nr, lmax_dat + 1))
        for r in range(nr):
            powers[r, :] = dat[r]["power_per_l"][:]

        if lmax == None or lmax > lmax_dat:
            lmax = lmax_dat

        radii = self.get_radii()
        depths = self.get_radii()[-1] - radii

        returned = plot.spectral_heterogeneity(
            powers,
            title,
            depths,
            lmin,
            lmax,
            saveplot,
            savepath,
            lyrmin,
            lyrmax,
            fig=fig,
            ax=ax,
            return_cbar=return_cbar,
            **kwargs,
        )

        fig = returned[0]
        ax = returned[1]
        cbar = returned[2] if return_cbar else None

        if show:
            fig.show()

        if return_cbar:
            return fig, ax, cbar
        else:
            return fig, ax

    def calc_bulk_composition(self):
        """
        Calculate the bulk composition field from composition histograms.
        Stored as new scalar field 'c'
        """
        c_hist = self.get_field("c_hist")
        bulk_composition = np.zeros((c_hist.shape[0], c_hist.shape[1]))
        cnames = self.get_composition_names()
        cvals = self.get_composition_values()

        for i, value in enumerate(cvals):
            bulk_composition += c_hist[:, :, i] * value

        self.set_field("c", bulk_composition)

    def plot_layer(
        self,
        field,
        radius=None,
        index=None,
        depth=False,
        delta=None,
        fig=None,
        ax=None,
        return_cbar=False,
        extent=(-180, 180, -90, 90),
        method="nearest",
        coastlines=True,
        cmap=None,
        vmin=None,
        vmax=None,
        v_field_ind=None,
        show=True,
    ):
        """
        Create a heatmap of the values of a particular field at the model
        layer nearest to ``radius`` km.

        :param field: Name of field of interest
        :param radius: Radius in km at which to show map.  The nearest
            model layer to this radius is shown.
        :param index: Rather than using a certain radius, plot the
            field exactly at a layer index
        :param depth: If True, interpret the radius as a depth instead
        :param delta: Grid spacing of plot in degrees
        :param fig: figure handle
        :param ax: axis handle
        :param return_cbar: flag to return colorbar
        :param extent: Tuple giving the longitude and latitude extent of
            plot, in the form (min_lon, max_lon, min_lat, max_lat), all
            in degrees
        :param method: May be one of: "nearest" (plot nearest value to each
            plot grid point); or "mean" (mean value in each pixel)
        :param coastlines: If ``True`` (default), plot coastlines.
            This may lead to a segfault on machines where cartopy is not
            installed in the recommended way.  In this case, pass ``False``
            to avoid this.
        :param v_field_ind: index of vector field component to plot
        :param show: If ``True`` (the default), show the plot
        :returns: figure and axis handles
        """
        if _is_vector_field(field) and v_field_ind == None:
            print("v_field_ind not supplied, defaulting to 0")
            v_field_ind = 0

        if radius is None and index is None:
            raise ValueError("Either radius or index must be given")
        if index is None:
            layer_index, layer_radius = self.nearest_layer(radius, depth)
        else:
            radii = self.get_radii()
            nlayers = len(radii)
            if index < 0 or index >= nlayers:
                raise ValueError(f"index must be between 0 and {nlayers}")

            layer_index = index
            layer_radius = radii[index]

        lon, lat = self.get_lateral_points()
        values = (
            self.get_field(field)[layer_index]
            if _is_scalar_field(field)
            else self.get_field(field)[layer_index, :, v_field_ind]
        )
        label = (
            _SCALAR_FIELDS[field] if _is_scalar_field(field) else _VECTOR_FIELDS[field]
        )

        if cmap is None and _is_scalar_field(field):
            cmap = _FIELD_COLOUR_SCALE[field]
        else:
            cmap = "viridis"

        returned = plot.layer_grid(
            lon,
            lat,
            layer_radius,
            values,
            delta=delta,
            extent=extent,
            label=label,
            method=method,
            coastlines=coastlines,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            fig=fig,
            ax=ax,
            return_cbar=return_cbar,
        )

        fig = returned[0]
        ax = returned[1]
        cbar = returned[2] if return_cbar else None

        if depth:
            ax.set_title(f"Depth {int(layer_radius)} km")

        if show:
            fig.show()

        if return_cbar:
            return fig, ax, cbar
        else:
            return fig, ax

    def plot_section(
        self,
        field,
        lon,
        lat,
        azimuth,
        distance,
        fig=None,
        ax=None,
        return_cbar=False,
        minradius=None,
        maxradius=None,
        delta_distance=1,
        delta_radius=50,
        method="nearest",
        levels=25,
        cmap=None,
        v_field_ind=None,
        show=True,
    ):
        """
        Create a plot of a cross-section through a model for one
        of the fields in the model.

        :param field: Name of field to plot
        :type field: str

        :param lon: Longitude of starting point of section in degrees
        :type lon: float

        :param lat: Latitude of starting point of section in degrees
        :type lat: float

        :param azimuth: Azimuth of cross section at starting point in degrees
        :type azimuth: float

        :param distance: Distance of cross section, given as the angle
            subtended at the Earth's centre between the starting and
            end points of the section, in degrees.
        :type distance: float

        :param fig: figure handle
        :type fig: matplotlib.figure.Figure

        :param ax: axis handle
        :type ax: matplotlib.axes._axes.Axes

        :param return_cbar: flag to return colorbar
        :type return_cbar: bool

        :param minradius: Minimum radius to plot in km.  If this is smaller
            than the minimum radius in the model, the model's value is used.
        :type minradius: float

        :param maxradius: Maximum radius to plot in km.  If this is larger
            than the maximum radius in the model, the model's value is used.
        :type maxradius: float

        :param method: May be one of "nearest" (default) or "triangle",
            controlling how points are calculated at each plotting grid
            point.  "nearest" simply finds the nearest model point to the
            required grid points; "triangle" perform triangular interpolation
            around the grid point.
        :type method: str

        :param delta_distance: Grid spacing in lateral distance, expressed
            in units of degrees of angle subtended about the centre of the
            Earth.  Default 1°.
        :type delta_distance: float

        :param delta_radius: Grid spacing in radial directions in km.  Default
            50 km.
        :type delta_radius: float

        :param levels: Number of levels or set of levels to plot
        :type levels: int or set of floats

        :param cmap: Colour map to be used (default "turbo")
        :type cmap: str

        :param v_field_ind: Index for component of vector field to plot
        :type v_field_ind: int

        :param show: If `True` (default), show the plot
        :type show: bool

        :returns: figure,axis and colourbar handles
        """
        if _is_vector_field(field) and v_field_ind == None:
            print("v_field_ind not supplied, defaulting to 0")
            v_field_ind = 0
        if not self.has_field(field) and not self.has_lookup_tables():
            raise ValueError(
                f"Model does not contain field '{field}', nor does it "
                + "contain lookup tables with which to compute it"
            )

        model_radii = self.get_radii()
        min_model_radius = np.min(model_radii)
        max_model_radius = np.max(model_radii)

        if minradius is None:
            minradius = min_model_radius
        if maxradius is None:
            maxradius = max_model_radius

        # For user-supplied numbers, clip them to lie in the range
        # min_model_radius to max_model_radius
        minradius = np.clip(minradius, min_model_radius, max_model_radius)
        maxradius = np.clip(maxradius, min_model_radius, max_model_radius)

        # Catch cases where both values are above or below the model and
        # which have been clipped
        if minradius >= maxradius:
            raise ValueError("minradius must be less than maxradius")

        # Allow us to plot at least two points
        if (maxradius - minradius) / 2 < delta_radius:
            delta_radius = (maxradius - minradius) / 2
        radii = np.arange(minradius, maxradius, delta_radius)
        distances = np.arange(0, distance, delta_distance)

        nradii = len(radii)
        ndistances = len(distances)

        grid = np.empty((ndistances, nradii))
        for i, distance in enumerate(distances):
            this_lon, this_lat = geographic.angular_step(lon, lat, azimuth, distance)
            for j, radius in enumerate(radii):
                if self.has_field(field):
                    result = self.evaluate(
                        this_lon,
                        this_lat,
                        radius,
                        field,
                        method=method,
                    )
                    grid[i, j] = (
                        result if _is_scalar_field(field) else result[v_field_ind]
                    )
                elif self.has_lookup_tables():
                    grid[i, j] = self.evaluate_from_lookup_tables(
                        this_lon, this_lat, radius, field, method=method
                    )

        label = (
            _SCALAR_FIELDS[field] if _is_scalar_field(field) else _VECTOR_FIELDS[field]
        )

        if cmap is None and _is_scalar_field(field):
            cmap = _FIELD_COLOUR_SCALE[field]
        else:
            cmap = "viridis"

        returned = plot.plot_section(
            distances,
            radii,
            grid,
            cmap=cmap,
            levels=levels,
            show=show,
            label=label,
            fig=fig,
            ax=ax,
            return_cbar=return_cbar,
        )

        fig = returned[0]
        ax = returned[1]
        cbar = returned[2] if return_cbar else None

        if return_cbar:
            return fig, ax, cbar
        else:
            return fig, ax

    def add_adiabat(self):
        """
        Add a theoretical adiabat to the temperatures in the
        TerraModel object. The adiabat is linear in the upper
        mantle and then is fit to a quadratic in the lower
        mantle.

        :param: none
        :return: none
        """

        radii = self.get_radii()
        surface_radius = radii[-1]
        depths = surface_radius - radii

        for d in depths:
            dt = _calculate_adiabat(d)
            layer_index, layer_radius = self.nearest_layer(radius=d, depth=True)
            self._fields["t"][layer_index] = self._fields["t"][layer_index] + dt

        return

    def add_geog_flow(self):
        """
        Add the field u_enu which holds the flow vector which
        has been rotated from cartesian to geographical.

        :param: none
        :return: none
        """

        flow_xyz = self.get_field("u_xyz")

        flow_enu = np.zeros(flow_xyz.shape)

        for point in range(self._npts):
            lon = self._lon[point]
            lat = self._lat[point]

            # get flow vector for one lon, lat
            # at all radii
            flows_point_all_radii = flow_xyz[:, point]

            # rotate vectors
            flow_enu_point = flow_conversion.rotate_vector(
                lon=lon, lat=lat, vec=flows_point_all_radii
            )

            # populate array with rotated vectors
            flow_enu[:, point] = flow_enu_point

        self.set_field(field="u_enu", values=flow_enu)

    def detect_plumes(
        self,
        depth_range=(400, 2600),
        algorithm="HDBSCAN",
        n_init="auto",
        epsilon=150,
        minsamples=150,
    ):
        """
        Uses the temperature and velocity fields to detect mantle plumes.
        Our scheme is a two stage process, first plume-like regions identified
        using a K-means clustering algorithm, then the resultant points are
        spatially clustered using a density based clustering algorithm to identify
        individual plumes. An inner 'plumes' class is created within the TerraModel to
        store information pertaining to detected plumes.

        :param depth_range: (min_depth, max_depth) over which to look for plumes
        :param algorithm: Spatial clustering algorithm - 'DBSCAN' and 'HDBSCAN' supported
        :param n_init: Number of times to run k-means with different starting centroids
        :param epsilon: Threshold distance parameter for DBSCAN, min_cluster_size for HDBSCAN
        :param minsamples: Minimum number of samples in a neighbourhood for DBSCAN and HDBSCAN
        :return: none
        """

        # First we need to check that we have the correct fields
        fields = self.field_names()
        if "t" not in fields:
            raise PlumeFieldError("t")
        if "u_enu" not in fields:
            if "u_xyz" in fields:
                print("adding geographic flow velocities")
                self.add_geog_flow()
            else:
                raise PlumeFieldError("u_xyz")

        # Perform K-means analysis save the binary locations of plumes
        print("k-means analysis")
        #        self._kmeans_plms=plume_detection.plume_kmeans(self,depth_range=depth_range)
        kmeans, plm_layers, plm_depths = plume_detection.plume_kmeans(
            self, depth_range=depth_range, n_init=n_init
        )

        # Now the density based clustering to identify individual plumes
        print("density based clustering")
        clust_result = plume_detection.plume_dbscan(
            self,
            kmeans,
            algorithm=algorithm,
            epsilon=epsilon,
            minsamples=minsamples,
            depth_range=depth_range,
        )

        # Initialize Plumes inner class
        self.plumes = self.Plumes(kmeans, plm_layers, plm_depths, clust_result, self)

    class Plumes:
        """
        An inner class of TerraModel, this class hold information pertaining to plumes
        which have been detected using the `model.detect_plumes` method.
        """

        def __init__(self, kmeans, plm_layers, plm_depths, clust_result, model):
            """
            Initialise new plumes inner class

            :param kmeans: Array of shape (nps,maxlyr-minlyr+1) where nps is the number
                of points in radial layer of a TerraModel and minlyr and maxlyr are the
                min and max layers over which we searched for plumes. Array contains binary information on whether a plume was detected.
            :param plm_layers: Layers of the TerraModel over which we searched for plumes
            :param plm_depths: Depths corresponding to the plm_layers
            :param clust_result: Cluster labels assigned by the spatial clustering
            :param model: TerraModel, needed to access fields in the inner class
            :return: none
            """
            self._kmeans_plms = kmeans
            self.plm_lyrs_range = plm_layers
            self.plm_depth_range = plm_depths
            self._plm_clusts = clust_result[0]
            self.n_plms = clust_result[1]
            self.n_noise = clust_result[2]
            self._model = model

            # Get lon and lat locations for points in plumes
            pnts_in_plm = np.argwhere(self._kmeans_plms)
            pnts = np.zeros((np.shape(pnts_in_plm)[0], 3))
            n = 0
            for i, d in enumerate(plm_depths):
                boolarr = self._kmeans_plms[:, i].astype(dtype=bool)
                pnts[n : (n + np.sum(boolarr)), 0] = model.get_lateral_points()[0][
                    boolarr
                ]  # fill lons
                pnts[n : (n + np.sum(boolarr)), 1] = s = model.get_lateral_points()[1][
                    boolarr
                ]  # fill lats
                pnts[n : (n + np.sum(boolarr)), 2] = self.plm_depth_range[
                    i
                ]  # fill depths
                n = n + np.sum(boolarr)

            self._pnts_plms = pnts

            self.plm_depths = {}
            for plumeID in range(self.n_plms):
                plume_nth = self._pnts_plms[self._plm_clusts == plumeID]
                self.plm_depths[plumeID] = np.unique(plume_nth[:, 2])

            # Add the lon,lat,depth of points assoicated with each plume
            self.plm_coords = {}
            for plumeID in range(self.n_plms):
                pnts_plmid = self._pnts_plms[self._plm_clusts == plumeID]
                deps = np.unique(self.plm_depths[plumeID])
                self.plm_coords[plumeID] = {}
                for d, dep in enumerate(deps):
                    self.plm_coords[plumeID][d] = pnts_plmid[pnts_plmid[:, 2] == dep]

        def calc_centroids(self):
            """
            Method calculates the centroids of each plume at each layer
            that the plume has been detected.

            :param: none
            :return: none
            """

            self.centroids = {}
            for plumeID in range(self.n_plms):
                self.centroids[plumeID] = plume_detection.plume_centroids(plumeID, self)

        def radial_field(self, field):
            """
            Method to find the values of a given field at points which have been
            detected as plumes.

            :param field: A field which exists in the TerraModel.
            :return: none
            """

            if field not in self._model.field_names():
                raise FieldNameError(field)

            # initialise dictionary which will store the plume fields
            if not hasattr(self, "plm_flds"):
                self.plm_flds = {}

            # create mask from kmeans outputs
            minlyr = np.min(self.plm_lyrs_range)
            maxlyr = np.max(self.plm_lyrs_range)
            mask = np.transpose(self._kmeans_plms.astype(dtype=bool))

            self.plm_flds[field] = {}
            fld = np.flip(self._model.get_field(field)[minlyr : maxlyr + 1, :], axis=0)[
                mask
            ]

            if _is_vector_field(field):
                for i in range(np.shape(self._model.get_field(field))[-1]):
                    fld = np.flip(
                        self._model.get_field(field)[minlyr : maxlyr + 1, :, i], axis=0
                    )[mask]
                    self.plm_flds[field][i] = {}

                    for plumeID in range(self.n_plms):
                        fld_plm = fld[
                            self._plm_clusts == plumeID
                        ]  # get data for this plume
                        pnts_plmid = self._pnts_plms[self._plm_clusts == plumeID]
                        deps = np.unique(self.plm_depths[plumeID])
                        self.plm_flds[field][i][plumeID] = {}

                        for d, dep in enumerate(deps):
                            self.plm_flds[field][i][plumeID][d] = fld_plm[
                                pnts_plmid[:, 2] == dep
                            ]  # get points at this depth

            else:
                fld = np.flip(
                    self._model.get_field(field)[minlyr : maxlyr + 1, :], axis=0
                )[mask]

                for plumeID in range(self.n_plms):
                    fld_plm = fld[
                        self._plm_clusts == plumeID
                    ]  # get data for this plume
                    pnts_plmid = self._pnts_plms[self._plm_clusts == plumeID]
                    deps = np.unique(self.plm_depths[plumeID])
                    self.plm_flds[field][plumeID] = {}

                    for d, dep in enumerate(deps):
                        self.plm_flds[field][plumeID][d] = fld_plm[
                            pnts_plmid[:, 2] == dep
                        ]  # get points at this depth

        def plot_kmeans_stack(
            self,
            centroids=0,
            delta=None,
            extent=(-180, 180, -90, 90),
            method="nearest",
            coastlines=True,
            fig=None,
            ax=None,
            return_cbar=False,
            show=True,
        ):
            """
            Create a heatmap of vertically stacked results of k-means analysis

            :param centroids: layer for which to plot centroids, eg 0 will plot
                plot the centroid of the uppermost layer for each plume, None
                will cause to not plot centorids.
            :param delta: Grid spacing of plot in degrees
            :param extent: Tuple giving the longitude and latitude extent of
                plot, in the form (min_lon, max_lon, min_lat, max_lat), all
                in degrees
            :param method: May be one of: "nearest" (plot nearest value to each
                plot grid point); or "mean" (mean value in each pixel)
            :param coastlines: If ``True`` (default), plot coastlines.
                This may lead to a segfault on machines where cartopy is not
                installed in the recommended way.  In this case, pass ``False``
                to avoid this.
            :param fig: figure handle
            :param ax: axis handle
            :param return_cbar: flag to return colorbar
            :param show: If ``True`` (the default), show the plot
            :returns: figure and axis handles
            """

            if not hasattr(self, "centroids"):
                print("calculating centroid of plume layers")
                self.calc_centroids()

            sumkmeans = np.sum(self._kmeans_plms, axis=1)
            lon, lat = self._model.get_lateral_points()
            label = "n-layers plume detected"
            radius = 0.0

            returned = plot.layer_grid(
                lon,
                lat,
                radius,
                sumkmeans,
                delta=delta,
                extent=extent,
                label=label,
                method=method,
                coastlines=coastlines,
                fig=fig,
                ax=ax,
                return_cbar=return_cbar,
            )

            fig = returned[0]
            ax = returned[1]
            cbar = returned[2] if return_cbar else None

            mindep = np.min(self.plm_depth_range)
            maxdep = np.max(self.plm_depth_range)

            ax.set_title(f"Depth range {int(mindep)} - {int(maxdep)} km")

            if centroids != None:
                for p in range(self.n_plms):
                    lon, lat, rad = self.centroids[p][centroids, :]
                    plot.point(ax, lon, lat, text=p)

            if show:
                fig.show()

            if return_cbar:
                return fig, ax, cbar
            else:
                return fig, ax

        def plot_plumes_3d(
            self,
            elev=10,
            azim=70,
            roll=0,
            dist=20,
            cmap="terrain",
            show=True,
        ):
            """
            Call to generate 3D scatter plot of points which constitute plumes
            coloured by plumeID

            :param elev: camera elevation (degrees)
            :param azim: camera azimuth (degrees)
            :param roll: camera roll (degrees)
            :param dist: camera distance (unitless)
            :param cmap: string corresponding to matplotlib colourmap
            :param show: If ``True`` (the default), show the plot
            """

            fig, ax = plot.plumes_3d(
                self, elev=elev, azim=azim, roll=roll, dist=dist, cmap=cmap
            )

            if show:
                fig.show()


class TerraModelLayer(TerraModel):
    """
    A subclass of the TerraModel superclass, TerraModelLayer is for storing 2D layer
    information which is written out of a TERRA simulation. Typically this might be some
    boundary information, eg CMB heat flux or radial surface radial stresses, but could be
    from any radial layer of the simulation in principle.

    Methods of the TerraModel class which are not compatible with TerraModelLayer are
    overwritten and will raise a LayerMethodError exception
    """

    def add_adiabat(self):
        raise LayerMethodError(self.add_adiabat.__name__)

    def get_1d_profile(self, *args):
        raise LayerMethodError(self.get_1d_profile.__name__)

    def plot_section(self, *args, **kwargs):
        raise LayerMethodError(self.plot_section.__name__)


def read_netcdf(
    files, fields=None, surface_radius=6370.0, test_lateral_points=False, cat=False
):
    """
    Read a TerraModel from a set of NetCDF files.

    :param files: List or iterable of file names of TERRA NetCDF model
        files
    :param fields: Iterable of field names to be read in.  By default all
        fields are read in.
    :param surface_radius: Radius of the surface of the model in km
        (default 6370 km)
    :returns: a new `TerraModel` or `TerraModelLayer`, depending on
        the contents of the file
    """
    if len(files) == 0:
        raise ValueError("files argument cannot be empty")

    # Check fields are readable
    if fields is not None:
        for field in fields:
            _check_field_name(field)

    # Initial assumption that file does not contain composition histogram
    ncomps = 0

    # Total number of lateral points and number of layers,
    # allowing us to preallocate arrays.  Consistency is checked on the next pass.
    npts_total = 0
    if not cat:
        for file_number, file in enumerate(files):
            nc = netCDF4.Dataset(file)
            if "nps" not in nc.dimensions:
                raise ValueError(f"File {file} does not contain the dimension 'nps'")
            npts_total += nc.dimensions["nps"].size

            _check_version(nc)

            if "depths" not in nc.dimensions:
                raise ValueError(f"File {file} does not contain the dimension 'Depths'")
            if file_number == 0:
                nlayers = nc.dimensions["depths"].size
                # Take the radii from the first file
                _r = np.array(surface_radius - nc["depths"][:], dtype=COORDINATE_TYPE)

                # If composition is present, get number of compositions from here
                # to check for consistency later
                if "composition_fractions" in nc.variables:
                    ncomps = nc.dimensions["compositions"].size + 1

        nfiles = len(files)
    else:
        file = files
        nc = netCDF4.Dataset(file)
        if "nps" not in nc.dimensions:
            raise ValueError(f"File {file} does not contain the dimension 'nps'")
        if "record" not in nc.dimensions:
            raise ValueError(f"Expecting concatenated file with dimension 'record'")
        npts_total = nc.dimensions["nps"].size * nc.dimensions["record"].size

        _check_version(nc)

        if "depths" not in nc.dimensions:
            raise ValueError(f"File {file} does not contain the dimension 'Depths'")
        nlayers = nc.dimensions["depths"].size
        _r = np.array(surface_radius - nc["depths"][:], dtype=COORDINATE_TYPE)
        nfiles = nc.dimensions["record"].size
        if "composition_fractions" in nc.variables:
            ncomps = nc.dimensions["compositions"].size + 1

    # Passed to constructor
    _fields = {}
    _lat = np.empty((npts_total,), dtype=COORDINATE_TYPE)
    _lon = np.empty((npts_total,), dtype=COORDINATE_TYPE)
    _c_hist_names = []
    _c_hist_values = []

    npts_pointer = 0

    if cat:
        nc = netCDF4.Dataset(file)

    for file_number in range(nfiles):
        # read in next file if not loading from concatenated file
        if not cat:
            file = files[file_number]
            nc = netCDF4.Dataset(file)

        # Check the file has the right things
        if len(nc["depths"][:]) != 1:
            for dimension in ("nps", "depths"):
                assert (
                    dimension in nc.dimensions
                ), f"Can't find {dimension} in dimensions of file {file}"
            if ncomps > 0:
                assert (
                    "compositions" in nc.dimensions
                ), f"Can't find compositions in dimensions of file {file}"

        # Number of lateral points in this file
        npts = nc.dimensions["nps"].size
        # Range of points in whole array to fill in
        npts_range = range(npts_pointer, npts_pointer + npts)

        if file_number > 0:
            # Check the radii are the same for this file as the first

            assert np.all(
                _r == surface_radius - nc["depths"][:]
            ), f"radii in file {file} do not match those in {files[0]}"

        # Assume that the latitudes and longitudes are the same for each
        # depth slice, and so are repeated
        if not cat:
            this_slice_lat = nc["latitude"][:]
            this_slice_lon = nc["longitude"][:]
        else:
            this_slice_lat = nc["latitude"][file_number, :]
            this_slice_lon = nc["longitude"][file_number, :]

        _lat[npts_range] = this_slice_lat
        _lon[npts_range] = this_slice_lon

        # Test this assumption
        if test_lateral_points:
            # Indexing with a single `:` gets an N-dimensional array, not
            # a vector
            all_lats = nc["latitude"][:]
            all_lons = nc["longitude"][:]
            for idep in range(1, nlayers):
                assert np.all(
                    this_slice_lat == all_lats[idep, :]
                ), f"Latitudes of depth slice {idep} do not match those of slice 0"
                assert np.all(
                    this_slice_lon == all_lons[idep, :]
                ), f"Longitudes of depth slice {idep} do not match those of slice 0"

        # Now read in fields, with some special casing
        fields_to_read = _ALL_FIELDS.keys() if fields == None else fields
        fields_read = set()

        for var in nc.variables:
            # Skip 'variables' like Latitude, Longitude and Depths which
            # give the values of the dimensions
            if var in ("latitude", "longitude", "depths"):
                continue

            field_name = _field_name_from_variable(var)
            if field_name not in fields_to_read:
                continue

            # Handle scalar fields
            if _is_scalar_field(field_name):
                if not cat:
                    field_data = nc[var][:]
                else:
                    field_data = nc[var][file_number, :]

                if field_name not in _fields.keys():
                    _fields[field_name] = np.empty(
                        (nlayers, npts_total), dtype=VALUE_TYPE
                    )
                _fields[field_name][:, npts_range] = field_data
                fields_read.add(field_name)

            # Special case for flow field
            if "u_xyz" in fields_to_read and field_name == "u_xyz":
                if field_name in fields_read:
                    continue
                else:
                    fields_read.add(field_name)

                u_ncomps = _VECTOR_FIELD_NCOMPS[field_name]
                uxyz = np.empty((nlayers, npts, u_ncomps), dtype=VALUE_TYPE)

                if not cat:
                    uxyz[:, :, 0] = nc["velocity_x"][:]
                    uxyz[:, :, 1] = nc["velocity_y"][:]
                    uxyz[:, :, 2] = nc["velocity_z"][:]
                else:
                    uxyz[:, :, 0] = nc["velocity_x"][file_number, :]
                    uxyz[:, :, 1] = nc["velocity_y"][file_number, :]
                    uxyz[:, :, 2] = nc["velocity_z"][file_number, :]

                if field_name not in _fields.keys():
                    _fields[field_name] = np.empty(
                        (nlayers, npts_total, u_ncomps), dtype=VALUE_TYPE
                    )
                _fields[field_name][:, npts_range, :] = uxyz

            # Special case for c_hist
            if "c_hist" in fields_to_read and field_name == "c_hist":
                if field_name in fields_read:
                    continue
                else:
                    fields_read.add(field_name)

                # Check for consistency in number of compositions
                this_ncomps = nc.dimensions["compositions"].size + 1
                if ncomps != this_ncomps:
                    raise FileFormatError(
                        file, "number of compositions", ncomps, this_ncomps
                    )

                # Get the composition attributes.
                # The first (ncomps - 1) compositions are those stored in
                # the composition_fractions variable, in that order.
                # Use the fact that we know there should be ncomps
                # attributes to check for consistency and get the names
                _check_has_composition_attributes(file, nc, ncomps)

                # Get the names and values from the first file, and check
                # for consistency in the other files
                for composition_index in range(ncomps):
                    composition_number = composition_index + 1
                    composition_name = getattr(
                        nc["composition_fractions"],
                        f"composition_{composition_number}_name",
                    )
                    composition_val = getattr(
                        nc["composition_fractions"],
                        f"composition_{composition_number}_c",
                    )

                    # This will only be filled the first time around
                    if len(_c_hist_names) >= composition_number:
                        # Check the names and values are the same for all files
                        if _c_hist_names[composition_index] != composition_name:
                            raise FileFormatError(
                                file,
                                f"composition_fractions:composition_{composition_number}_name",
                                _c_hist_names[composition_index],
                                composition_name,
                            )
                        if _c_hist_values[composition_index] != composition_val:
                            raise FileFormatError(
                                file,
                                f"composition_fractions:composition_{composition_number}_val",
                                _c_hist_values[composition_index],
                                composition_val,
                            )
                    else:
                        _c_hist_names.append(composition_name)
                        _c_hist_values.append(composition_val)

                if field_name not in _fields.keys():
                    _fields[field_name] = np.empty(
                        (nlayers, npts_total, ncomps), dtype=VALUE_TYPE
                    )

                # Handle case that indices are in different order in file
                # compared to TerraModel
                for icomp in range(ncomps - 1):
                    if not cat:
                        _fields[field_name][:, npts_range, icomp] = nc[var][icomp, :, :]
                    else:
                        _fields[field_name][:, npts_range, icomp] = nc[var][
                            file_number, icomp, :, :
                        ]

                # Calculate final composition fraction slice using the property
                # that all composition fractions must sum to 1
                _fields[field_name][:, npts_range, ncomps - 1] = 1 - np.sum(
                    [_fields[field_name][:, npts_range, i] for i in range(ncomps - 1)],
                    axis=0,
                )

        if not cat:
            nc.close()

        npts_pointer += npts

    # Check for need to sort points in increasing radius
    must_flip_radii = len(_r) > 1 and _r[1] < _r[0]
    if must_flip_radii:
        _r = np.flip(_r)

    # Remove duplicate points
    _, unique_indices = np.unique(np.stack((_lon, _lat)), axis=1, return_index=True)
    # np.unique returns indices which sort the values, so sort the
    # indices to preserve the original point order, except the duplicates
    unique_indices = np.sort(unique_indices)
    _lon = _lon[unique_indices]
    _lat = _lat[unique_indices]

    # Remove duplicate points and flip radius axis if needed
    for field_name, array in _fields.items():
        ndims = array.ndim
        if ndims == 2:
            if must_flip_radii:
                _fields[field_name] = array[::-1, unique_indices]
            else:
                _fields[field_name] = array[:, unique_indices]
        elif ndims == 3:
            if must_flip_radii:
                _fields[field_name] = array[::-1, unique_indices, :]
            else:
                _fields[field_name] = array[:, unique_indices, :]
        else:
            # Shouldn't be able to happen
            raise ValueError(
                f"field {field_name} has an unexpected number of dimensions ({ndims})"
            )
    if len(_r) == 1:
        return TerraModelLayer(
            r=_r,
            lon=_lon,
            lat=_lat,
            fields=_fields,
            c_histogram_names=_c_hist_names,
            c_histogram_values=_c_hist_values,
        )
    else:
        return TerraModel(
            r=_r,
            lon=_lon,
            lat=_lat,
            fields=_fields,
            c_histogram_names=_c_hist_names,
            c_histogram_values=_c_hist_values,
        )


def load_model_from_pickle(filename):
    """
    Load a terra model saved using the save() function above.

    :param filename: filename to load terramodel from.
    :type filename: str

    :return: loaded terra model
    :rtype: TerraModel object
    """

    f = open(filename, "rb")
    m = pickle.load(f)
    f.close()
    return m


def _calculate_adiabat(depth):
    """
    Calculate a theoretical adiabat at a given depth.
    The adiabat has a linear slope of 0.5 K/km in the
    upper mantle and fit by a quadratic in the lower mantle.
    The upper and lower mantle adiabats are smoothed around
    the 660 km transition depth. The value given is relative
    to a 1600 K mantle potential temperature.

    :param depth: depth to get adiabat temperature at
    :type depth: float

    :return: adiabat temperature value relative to a 1600 K
             potential temperature.
    :rtype: float
    """

    # calculate temp if it were a linear profile (for upper mantle)
    lin = (0.5 * depth) + 1600

    # calculate if it were a quadratic profile (for lower mantle)
    quad = (-0.00002 * depth**2) + (0.4 * depth) + 1700

    # smooth transition between linear and quadratic
    sig = 1 / (1 + (np.exp((-1 * depth - 660) / 60)))

    # 1600, the potential temperature, is removed
    # so only the adiabat relative to the potential temp
    # of 1600 is returned
    adiabat = lin * (1 - sig) + (quad * sig) - 1600

    return adiabat


def _compositions_sum_to_one(compfracs, atol=np.finfo(VALUE_TYPE).eps):
    """
    Return ``True`` if the sum of composition fractions is equal to 1 (within
    ``atol``) for all points; otherwise return ``False``.
    """

    return np.allclose(np.sum(compfracs, axis=2), 1, atol=atol)


def _check_version(nc):
    """
    Check the version of the netCDF file to allow for read
    """
    # Check if version global attribute exists
    try:
        version = nc.getncattr("version")
    except:
        version = 0.0

    # Old file types raise exception
    if version < 1.0:
        raise VersionError(version)


def _check_has_composition_attributes(file, nc, ncomps):
    """
    If the netCDF4.Dataset ``nc`` does not contain the correct
    attributes for the ``"composition_fractions"`` variable,
    raise a ``FileFormatError``.
    """
    for composition_number in range(1, ncomps + 1):
        for attribute in ("name", "c"):
            att_name = f"composition_{composition_number}_{attribute}"
            if att_name not in nc["composition_fractions"].ncattrs():
                raise FileFormatError(
                    file,
                    "composition_fractions:" + att_name,
                    "it to be present",
                    "no such attribute",
                )


def _is_valid_field_name(field):
    """
    Return True if field is a valid name of a field in a TerraModel.
    """
    return field in _ALL_FIELDS.keys()


def _pixelise(signal, nside, lons, lats):
    """
    Grid input data to healpix grid
    :param signal: input data length n
    :param nside: healpy param, number of sides for healpix grid
    :param lons: input longitudes length n
    :param lats: input latitudes length n
    :returns: healpix grid
    """
    npix = hp.nside2npix(nside)
    pixnum = hp.ang2pix(nside, lons, lats, lonlat=True)
    amap = np.zeros(npix)
    count = np.zeros(npix)
    nsample = len(signal)
    for i in range(nsample):
        pix = pixnum[i]
        amap[pix] += signal[i]
        count[pix] += 1.0
    for i in range(npix):
        if count[i] > 0:
            amap[i] = amap[i] / count[i]
        else:
            amap[i] = hp.UNSEEN
    return amap


def _variable_names_from_field(field):
    """
    Return the netCDF variable name(s) of a field from the TerraModel field name.
    The values returned are tuples.
    """
    return _FIELD_NAME_TO_VARIABLE_NAME[field]


def _field_name_from_variable(field):
    """
    Return the TerraModel field name of a NetCDF file variable name.

    If there is no field name associated with this variable name,
    return ``None``.
    """
    try:
        field_name = _VARIABLE_NAME_TO_FIELD_NAME[field]
    except KeyError:
        field_name = None
    return field_name


def _check_new_field_name(field):
    """
    If field is not currently a valid field name, raise a NewFieldNameError
    """
    if not _is_valid_field_name(field):
        raise NewFieldNameError(field)


def _check_field_name(field):
    """
    If field is not a valid field name, raise a FieldNameError
    """
    if not _is_valid_field_name(field):
        raise FieldNameError(field)


def _is_scalar_field(field):
    """
    Return True if field is a scalar field
    """
    return field in _SCALAR_FIELDS.keys()


def _is_vector_field(field):
    """
    Return True if field is a 'vector' field
    """
    return field in _VECTOR_FIELDS.keys()


def _expected_vector_field_ncomps(field):
    """
    Return the expected number of components in a 'vector' field,
    or None if it may take any value
    """
    return _VECTOR_FIELD_NCOMPS[field]


def _fit_nn_tree(lon, lat):
    """
    Fit a nearest neighbour lookup tree to a set of longitude and
    latitude points on the surface of a sphere, where the input
    is in degrees.

    Rough timing suggests that using the geographic coordinates and
    Haversine distance takes ~440 µs per lookup, whereas using
    Cartesian coordinates and a Euclidian distance leads to
    lookups taking ~15 ms.

    # FIXME: Properly test which distance metrics and tree types
    #        are fastest.
    """
    lon_radians = np.radians(lon)
    lat_radians = np.radians(lat)
    coords = np.array([[lat, lon] for lat, lon in zip(lat_radians, lon_radians)])
    tree = NearestNeighbors(n_neighbors=1, metric="haversine").fit(coords)
    return tree


def _nearest_index(value, values):
    """
    Return the index of the nearest value in ``values``.  If
    ``value`` is smaller or greater than all ``values``, respectively
    return the minimum or maximum.

    Requires that ``values`` is sorted in increasing order.
    """
    nvals = len(values)
    index = np.searchsorted(values, value)

    # Outside the range
    if index == nvals:
        return nvals - 1
    elif index == 0:
        return index

    # Need to decide whether the one above or below is closer
    if value - values[index - 1] > values[index] - value:
        return index
    else:
        return index - 1


def _bounding_indices(value, values):
    """
    Return the indices of ``values`` ``i1`` and ``i2`` where
    ``values[i1] <= value < values[i2]``.

    If ``value`` is less or greater than the smallest and largest
    values, respectively return the same index for both.
    In other words, the returned index is 0 when value is below all
    values, and ``len(values)-1`` when it is above.

    If value is exactly the same as one of the values, return the
    corresponding index twice also.

    Requires that ``values`` is sorted in increasing order.
    """
    nvals = len(values)
    index = np.searchsorted(values, value)

    inc = np.isin(value, values)

    return np.clip(np.array([index - 1 + inc, index]), 0, nvals - 1)
