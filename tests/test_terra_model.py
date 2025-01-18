import unittest

import numpy as np
import os

_CARTOPY_INSTALLED = True
try:
    from cartopy.mpl.geoaxes import GeoAxesSubplot
except ImportError:
    _CARTOPY_INSTALLED = False

from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes

from terratools import terra_model
from terratools.terra_model import TerraModel

# Tolerance for match of floating point numbers, given that TerraModel
# coordinates and values have a defined precision
coord_tol = np.finfo(terra_model.COORDINATE_TYPE).eps
value_tol = np.finfo(terra_model.VALUE_TYPE).eps

# Random number generator to use here
_RNG = np.random.default_rng()


# Helper functions for the tests
def dummy_model(nlayers=3, npts=4, with_fields=False, **kwargs):
    lon, lat, r = random_coordinates(nlayers, npts)
    if with_fields:
        fields = {
            "t": random_field(nlayers, npts),
            "u_xyz": random_field(nlayers, npts, 3),
        }
    else:
        fields = {}
    return TerraModel(lon, lat, r, fields=fields, **kwargs)


def random_coordinates(nlayers, npts):
    """Return a set of random coordinates (satisfying the constraints)
    which can be passed to construct a TerraModel.
    FIXME: In theory there may be duplicate points as they are random."""
    rmin, rmax = np.sort(7000 * np.random.rand(2))
    r = np.sort(np.random.uniform(rmin, rmax, nlayers))
    lon = 360 * (np.random.rand(npts) - 0.5)
    lat = 180 * (np.random.rand(npts) - 0.5)
    return lon, lat, r


def random_field(nlayers, npts, ncomps=None, dtype=terra_model.VALUE_TYPE):
    if ncomps is None:
        return _RNG.random((nlayers, npts), dtype=dtype)
    else:
        return _RNG.random((nlayers, npts, ncomps), dtype=dtype)


def read_test_lateral_points():
    """Read a subset of points on the TERRA grid in a box from
    -2° to 2 ° in longitude and latitude"""
    dir = os.path.dirname(__file__)
    file = os.path.join(dir, "data", "TERRA_grid_lon-2_2_lat-2_2.txt")
    data = np.loadtxt(file, dtype=np.float32)
    lon = data[:, 0]
    lat = data[:, 1]
    return lon, lat


def fields_are_equal(field1, field2):
    return np.allclose(field1, field2, atol=value_tol)


def coords_are_equal(coords1, coords2):
    return np.allclose(coords1, coords2, atol=coord_tol)


class TestTerraModelHelpers(unittest.TestCase):
    """Tests for non-class methods"""

    def test_is_valid_field_name(self):
        """Test for validity of a field name"""
        self.assertFalse(terra_model._is_valid_field_name("incorrect field name"))
        self.assertTrue(terra_model._is_valid_field_name("t"))

    def test_variable_name_from_field(self):
        """Translation of field name to NetCDF variable name(s)"""

        self.assertEqual(terra_model._variable_names_from_field("t"), ("temperature",))
        self.assertEqual(
            terra_model._variable_names_from_field("u_xyz"),
            ("velocity_x", "velocity_y", "velocity_z"),
        )
        self.assertEqual(
            terra_model._variable_names_from_field("c_hist"), ("composition_fractions",)
        )

        with self.assertRaises(KeyError):
            terra_model._variable_names_from_field("incorrect field name")

    def test_field_name_from_variable(self):
        """Translation of NetCDF variable name to field name"""
        self.assertEqual(terra_model._field_name_from_variable("temperature"), "t")
        self.assertEqual(terra_model._field_name_from_variable("absent field"), None)

    def test_check_field_name(self):
        self.assertEqual(terra_model._check_field_name("vp"), None)
        with self.assertRaises(terra_model.FieldNameError):
            terra_model._check_field_name("incorrect field name")

    def test_is_scalar_field(self):
        self.assertTrue(terra_model._is_scalar_field("c"))
        self.assertFalse(terra_model._is_scalar_field("c_hist"))

    def test_is_vector_field(self):
        self.assertTrue(terra_model._is_vector_field("u_xyz"))
        self.assertFalse(terra_model._is_vector_field("t"))

    def test_expected_vector_field_ncomps(self):
        self.assertEqual(terra_model._expected_vector_field_ncomps("u_xyz"), 3)
        self.assertEqual(terra_model._expected_vector_field_ncomps("u_enu"), 3)
        self.assertEqual(terra_model._expected_vector_field_ncomps("c_hist"), None)

    def test_compositions_sum_to_one(self):
        comps = np.zeros((4, 3, 2))
        self.assertFalse(terra_model._compositions_sum_to_one(comps))
        comps[:, :, -1] = 1
        self.assertTrue(terra_model._compositions_sum_to_one(comps))
        comps[:, :, -1] = 0.9
        self.assertTrue(terra_model._compositions_sum_to_one(comps, atol=0.2))
        self.assertFalse(terra_model._compositions_sum_to_one(comps, atol=1.0e-7))
        self.assertFalse(terra_model._compositions_sum_to_one(comps))


class TestTerraModelConstruction(unittest.TestCase):
    """Tests for construction and validation of fields"""

    def test_invalid_field_dimensions(self):
        npts = 3
        nlayers = 2
        scalar_field = random_field(nlayers, npts)
        vector_field = random_field(nlayers, npts, 3)
        fields = {"t": scalar_field, "u_xyz": vector_field}

        with self.assertRaises(terra_model.FieldDimensionError):
            lon, lat, r = random_coordinates(nlayers, npts + 1)
            TerraModel(lon, lat, r, fields=fields)

        with self.assertRaises(terra_model.FieldDimensionError):
            lon, lat, r = random_coordinates(nlayers, npts - 1)
            TerraModel(lon, lat, r, fields=fields)

        with self.assertRaises(terra_model.FieldDimensionError):
            lon, lat, r = random_coordinates(nlayers + 1, npts)
            TerraModel(lon, lat, r, fields=fields)

        with self.assertRaises(terra_model.FieldDimensionError):
            lon, lat, r = random_coordinates(nlayers - 1, npts)
            TerraModel(lon, lat, r, fields=fields)

    def test_invalid_field_name(self):
        npts = 10
        nlayers = 3
        field = random_field(nlayers, npts)
        lon, lat, r = random_coordinates(nlayers, npts)
        with self.assertRaises(terra_model.FieldNameError):
            TerraModel(lon, lat, r, fields={"incorrect field name": field})

    def test_invalid_ncomps(self):
        nlayers = 3
        npts = 2
        u_xyz = random_field(nlayers, npts, 2)
        lon, lat, r = random_coordinates(nlayers, npts)
        with self.assertRaises(terra_model.FieldDimensionError):
            TerraModel(lon, lat, r, fields={"u_xyz": u_xyz})

    def test_radii_not_motonic(self):
        with self.assertRaises(ValueError):
            TerraModel([1], [1], [1, 3, 2])

    def test_radii_not_increasing(self):
        with self.assertRaises(ValueError):
            TerraModel([1], [1], [3, 2, 1])

    def test_lon_lat_not_same_length(self):
        with self.assertRaises(ValueError):
            TerraModel([1], [2, 3], [1, 2, 3])

    def test_composition_proportions_do_not_sum_to_one(self):
        nlayers = 3
        npts = 2
        ncomps = 4
        lon, lat, r = random_coordinates(nlayers, npts)
        c_hist_field = random_field(nlayers, npts, ncomps)
        with self.assertRaises(ValueError):
            TerraModel(lon, lat, r, fields={"c_hist": c_hist_field})

    def test_surface_radius_too_small(self):
        lon, lat, r = random_coordinates(3, 2)
        with self.assertRaises(ValueError):
            TerraModel(lon, lat, r, surface_radius=r[-1] - 1)

    def test_construction(self):
        """Ensure the things we pass in are put in the right place"""
        nlayers = 3
        npts = 10
        lon, lat, r = random_coordinates(nlayers, npts)
        scalar_field_names = ("t", "c", "vp", "vs", "density", "p")
        scalar_fields = [random_field(nlayers, npts) for _ in scalar_field_names]
        u_field = random_field(nlayers, npts, 3)
        c_hist_field = random_field(nlayers, npts, 2)
        # Ensure each composition histogram sums to unity
        c_hist_field[:, :, 1] = 1 - c_hist_field[:, :, 0]
        fields = {name: field for name, field in zip(scalar_field_names, scalar_fields)}
        fields["u_xyz"] = u_field
        fields["u_enu"] = u_field
        fields["c_hist"] = c_hist_field
        c_hist_names = ["A", "B"]
        c_hist_values = [1, 2]

        model = TerraModel(
            lon,
            lat,
            r,
            fields=fields,
            c_histogram_names=c_hist_names,
            c_histogram_values=c_hist_values,
        )

        _lon, _lat = model.get_lateral_points()
        self.assertTrue(coords_are_equal(lon, _lon))
        self.assertTrue(coords_are_equal(lat, _lat))

        self.assertTrue(coords_are_equal(model.get_radii(), r))

        for field_name, field in zip(scalar_field_names, scalar_fields):
            self.assertTrue(fields_are_equal(model.get_field(field_name), field))

        self.assertTrue(fields_are_equal(model.get_field("c_hist"), c_hist_field))

        for field in ("u_xyz", "u_enu"):
            self.assertTrue(fields_are_equal(model.get_field(field), u_field))

        self.assertTrue(fields_are_equal(model.get_field("c_hist"), c_hist_field))
        self.assertEqual(model.number_of_compositions(), 2)
        self.assertEqual(model.get_composition_names(), ["A", "B"])
        self.assertEqual(model.get_composition_values(), [1, 2])

        # Use set because we don't need to enforce that the fields
        # are in the same order
        self.assertEqual(set(model.field_names()), set(fields.keys()))

        for field in (*scalar_field_names, "u_enu", "u_xyz", "c_hist"):
            self.assertTrue(model.has_field(field))
        self.assertFalse(model.has_field("vs_an"))


class TestTerraModelGetters(unittest.TestCase):
    """Tests for getters"""

    def test_invalid_field_name(self):
        """Check that an error is thrown when asking for an invalid field"""
        model = dummy_model()
        model.new_field("t")
        with self.assertRaises(terra_model.FieldNameError):
            model.evaluate(0, 0, 4000, "incorrect field name")

    def test_missing_field(self):
        model = dummy_model()
        model.new_field("t")
        with self.assertRaises(terra_model.NoFieldError):
            model.evaluate(0, 0, 4000, "u_xyz")

    def test_get_field(self):
        model = dummy_model(with_fields=True)
        self.assertIs(model.get_field("t"), model._fields["t"])

        temp = model.get_field("t")
        temp[0, 0] = 1
        self.assertTrue(np.all(model.get_field("t") == temp))

    def test_get_radii(self):
        r = [1, 2, 3]
        model = TerraModel([1, 2], [3, 4], r)
        self.assertIs(model.get_radii(), model._radius)

    def test_get_lateral_points(self):
        lon, lat, r = random_coordinates(3, 4)
        model = TerraModel(lon, lat, r)
        _lon, _lat = model.get_lateral_points()
        self.assertIs(_lon, model._lon)
        self.assertIs(_lat, model._lat)

    def test_get_composition_names(self):
        model = dummy_model(c_histogram_names=["A", "B"])
        model.new_field("c_hist", 2)
        self.assertEqual(model.get_composition_names(), ["A", "B"])

    def test_nearest_layer(self):
        model = TerraModel([1], [1], [5, 7, 9])
        self.assertCountEqual(model.nearest_layer(4), (0, 5.0), 2)
        self.assertCountEqual(model.nearest_layer(5.1), (0, 5.0), 2)
        self.assertCountEqual(model.nearest_layer(6.5), (1, 7.0), 2)
        self.assertCountEqual(model.nearest_layer(7.0), (1, 7.0), 2)

    def test_get_composition_values(self):
        model = dummy_model(c_histogram_names=["A", "B"], c_histogram_values=[1, 2])
        model.new_field("c_hist", 2)
        self.assertEqual(model.get_composition_values(), [1, 2])

    def test_pressure_at_radius(self):
        # Default PREM pressure
        model = TerraModel(lon=[1], lat=[1], r=[6371], surface_radius=6371)
        self.assertAlmostEqual(model.pressure_at_radius(3480), 135.751e9, 2)
        # Some arbitrary function
        model = TerraModel(lon=[1], lat=[1], r=[6371], pressure_func=lambda r: 2 * r)
        self.assertEqual(model.pressure_at_radius(100), 200)


class TestTerraModelNewField(unittest.TestCase):
    def test_wrong_ncomps(self):
        model = dummy_model()
        with self.assertRaises(ValueError):
            model.new_field("t", 1)
        with self.assertRaises(ValueError):
            model.new_field("u_xyz", 4)

    def test_no_ncomps(self):
        model = dummy_model()
        with self.assertRaises(ValueError):
            model.new_field("c_hist")

    def test_new_field(self):
        model = dummy_model()
        nlayers = len(model.get_radii())
        npts = len(model.get_lateral_points()[0])

        self.assertFalse(model.has_field("t"))
        self.assertFalse(model.has_field("u_xyz"))
        self.assertFalse(model.has_field("c_hist"))

        model.new_field("t")
        self.assertTrue(model.has_field("t"))
        self.assertTrue(
            fields_are_equal(model.get_field("t"), np.zeros((nlayers, npts)))
        )

        model.new_field("u_xyz")
        self.assertTrue(model.has_field("u_xyz"))
        self.assertTrue(
            fields_are_equal(model.get_field("u_xyz"), np.zeros((nlayers, npts, 3)))
        )

        model.new_field("c_hist", 2)
        self.assertTrue(model.has_field("c_hist"))
        self.assertTrue(
            fields_are_equal(model.get_field("c_hist"), np.zeros((nlayers, npts, 2)))
        )


class TerraModelDepthConversion(unittest.TestCase):
    def test_to_depth(self):
        model = dummy_model(surface_radius=10000)
        self.assertEqual(model.to_depth(2500), 7500)
        model = dummy_model()
        surface_radius = model.get_radii()[-1]
        self.assertAlmostEqual(model.to_depth(1), surface_radius - 1, 3)

    def test_to_radius(self):
        model = dummy_model(surface_radius=10000)
        self.assertEqual(model.to_radius(2500), 7500)
        model = dummy_model()
        surface_radius = model.get_radii()[-1]
        self.assertAlmostEqual(model.to_radius(1), surface_radius - 1, 3)


class TestTerraModelRepr(unittest.TestCase):
    def test_repr(self):
        npts = 3
        nlayers = 3
        lon = [1, 2, 3]
        lat = [10, 20, 30]
        r = [1000, 1999, 2000]
        t_field = random_field(nlayers, npts)
        c_hist_field = random_field(nlayers, npts, 2)
        # Require compositions sum to 1
        c_hist_field[:, :, 1] = 1 - c_hist_field[:, :, 0]
        cnames = ["a", "b"]
        model = TerraModel(
            lon,
            lat,
            r,
            fields={"t": t_field, "c_hist": c_hist_field},
            c_histogram_names=cnames,
        )

        # Ensure we print these numbers out in the default format for
        # the NumPy version in use, as this changes in minor releases;
        # see 'legacy' at:
        # https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
        rmin = terra_model.COORDINATE_TYPE(r[0])
        rmax = terra_model.COORDINATE_TYPE(r[-1])

        self.assertEqual(
            model.__repr__(),
            f"""TerraModel:
           number of radii: 3
             radius limits: ({rmin}, {rmax})
  number of lateral points: 3
                    fields: ['t', 'c_hist']
         composition names: ['a', 'b']
        composition values: None
         has lookup tables: False""",
        )


class TestTerraBulkComposition(unittest.TestCase):
    def test_calc_bulk_composition(self):
        npts = 16
        nlayers = 3
        lon = np.linspace(0, 180, npts)
        lat = np.linspace(0, 90, npts)
        r = [1000, 1999, 2000]
        t_field = random_field(nlayers, npts)
        c_hist_field = random_field(nlayers, npts, 3)
        # Ensure values sum to 1
        c_hist_field[:, :, 2] = 1 - c_hist_field[:, :, 0] - c_hist_field[:, :, 1]
        cnames = [
            "composition_0",
            "composition_1",
            "composition_2",
        ]
        cvals = [0.0, 0.2, 1.0]
        model = TerraModel(
            lon,
            lat,
            r,
            fields={"t": t_field, "c_hist": c_hist_field},
            c_histogram_names=cnames,
            c_histogram_values=cvals,
        )

        model.calc_bulk_composition()
        self.assertEqual(model.get_field("c").shape, model.get_field("t").shape)

        test_value = np.sum(c_hist_field[0, 0, :] * np.array(cvals))
        self.assertAlmostEqual(model.get_field("c")[0, 0], test_value)


class TestTerraModelNearestIndex(unittest.TestCase):
    def test_nearest_index(self):
        lon = [20, 22, 0.1, 25]
        lat = [20, 22, 0.1, 24]
        r = [10, 20]
        model = TerraModel(lon, lat, r)
        self.assertEqual(model.nearest_index(0, 0), 2)
        self.assertCountEqual(model.nearest_index([0, 0.1], [0, 0.1]), [2, 2], 2)


class TestTerraModelNearestIndices(unittest.TestCase):
    def test_nearest_indices_zero_n(self):
        with self.assertRaises(ValueError):
            dummy_model().nearest_indices(0, 0, 0)

    def test_nearest_indices(self):
        lon = [20, 22, 0.1, 25]
        lat = [20, 22, 0.1, 24]
        r = [10, 20]
        model = TerraModel(lon, lat, r)
        self.assertCountEqual(model.nearest_indices(0, 0, n=3), [2, 0, 1])

        indices = model.nearest_indices([0, 0.1], [0, 0.1], n=4)
        for inds in indices:
            self.assertCountEqual(inds, [2, 0, 1, 3], 4)


class TestTerraModelNearestNeighbors(unittest.TestCase):
    def test_nearest_neighbors_zero_n(self):
        with self.assertRaises(ValueError):
            dummy_model().nearest_neighbors(0, 0, 0)

    def test_nearest_neighbors(self):
        lon = [0, 0, 0, 0]
        lat_radians = [0, 0.1, -0.2, 0.3]
        lat = np.degrees(lat_radians)
        r = [1]
        model = TerraModel(lon, lat, r)
        indices, distances = model.nearest_neighbors(0, 0, 4)
        self.assertTrue(np.allclose(distances, [0, 0.1, 0.2, 0.3], atol=1e-7))
        self.assertTrue(np.allclose(indices, [0, 1, 2, 3], atol=1e-7))

        indices, distances = model.nearest_neighbors([0, 0], [0, np.degrees(0.1)], n=4)
        self.assertTrue(np.allclose(distances[0], [0, 0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(indices[0], [0, 1, 2, 3]))
        self.assertTrue(np.allclose(distances[1], [0.0, 0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(indices[1], [1, 0, 3, 2]))


class TestTerraModelEvaluate(unittest.TestCase):
    def test_wrong_method(self):
        with self.assertRaises(ValueError):
            dummy_model(with_fields=True).evaluate(
                0, 0, 1, "t", method="unsupported method"
            )

    def test_evaluate_depth(self):
        model = dummy_model(with_fields=True)
        radii = model.get_radii()
        mid_radius = np.min(radii) + (np.max(radii) - np.min(radii)) / 2
        surface_radius = np.max(radii)
        mid_depth = surface_radius - mid_radius
        self.assertEqual(
            model.evaluate(0, 0, mid_radius, "t"),
            model.evaluate(0, 0, mid_depth, "t", depth=True),
        )
        self.assertCountEqual(
            model.evaluate(0, 0, mid_radius, "u_xyz"),
            model.evaluate(0, 0, mid_depth, "u_xyz", depth=True),
            3,
        )

    def test_evaluate_vectorize_triangle(self):
        model = dummy_model(with_fields=True)
        radii = model.get_radii()
        mid_radius = np.min(radii) + (np.max(radii) - np.min(radii)) / 2

        v1 = model.evaluate(0.0, 0.0, mid_radius, "t")
        v2 = model.evaluate(1.0, 1.0, mid_radius * 1.5, "t")
        vs = model.evaluate(
            np.array([0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([mid_radius, mid_radius * 1.5]),
            "t",
        )
        self.assertTrue(np.allclose(vs, [v1, v2]))

    def test_evaluate_vectorize_nearest(self):
        model = dummy_model(with_fields=True)
        radii = model.get_radii()
        mid_radius = np.min(radii) + (np.max(radii) - np.min(radii)) / 2

        v1 = model.evaluate(0.0, 0.0, mid_radius, "t", method="nearest")
        v2 = model.evaluate(1.0, 1.0, mid_radius * 1.5, "t", method="nearest")
        vs = model.evaluate(
            np.array([0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([mid_radius, mid_radius * 1.5]),
            "t",
            method="nearest",
        )
        self.assertTrue(np.allclose(vs, [v1, v2]))

    def test_evaluate_radii_interpolation(self):
        model = TerraModel(lon=[-1, 1, 0], lat=[1, 1, -1], r=[1, 2])
        model.new_field("t")
        temp = model.get_field("t")
        temp[0, :] = 10
        temp[1, :] = 20
        self.assertAlmostEqual(model.evaluate(0, 0, 1.1, "t"), 11)
        self.assertAlmostEqual(model.evaluate(0, 0, 0.9, "t", depth=True), 11)

    def test_evaluate_lateral_interpolation(self):
        def test_func(lon, lat):
            return lat

        lon, lat = read_test_lateral_points()
        radii = [3480, 6370]
        model = TerraModel(lon, lat, radii)
        model.new_field("t")
        temp = model.get_field("t")
        # Make temp follow an analytical function
        test_r = radii[0]
        for i in range(len(radii)):
            temp[i, :] = test_func(lon, lat)

        test_lon, test_lat = 4 * (np.random.rand(2) - 0.5)
        self.assertAlmostEqual(
            model.evaluate(test_lon, test_lat, test_r, "t"),
            test_func(test_lon, test_lat),
            delta=1,
        )

    def test_evaluate_nearest(self):
        lon = [0, 1, 2]
        lat = [0, 1, 0]
        r = [1]
        model = TerraModel(lon, lat, r)
        u_xyz = model.new_field("u_xyz")
        u_xyz[0, 0, :] = [1, 2, 3]
        u_xyz[0, 1, :] = [10, 20, 30]
        u_xyz[0, 2, :] = [-2, -4, 10]
        self.assertTrue(
            np.all(
                model.evaluate(1.9, 0.1, 1, "u_xyz", method="nearest") == [-2, -4, 10]
            )
        )


class TestRadialProfile(unittest.TestCase):
    def test_radial_profile_triangular_interpolation(self):
        # Create model
        lons, lats = read_test_lateral_points()
        radii = np.arange(0.5, 1, 0.1)
        m = TerraModel(lons, lats, radii)
        t = m.new_field("t")

        def layer_func(lons, lats, index):
            return index * lons**2 * np.sin(np.radians(lats))

        for ilayer in range(len(radii)):
            t[ilayer, :] = layer_func(lons, lats, ilayer)

        # True values
        test_lon, test_lat = -1, -1
        test_profile = layer_func(test_lon, test_lat, np.arange(len(radii)))

        profile = m.radial_profile(test_lon, test_lat, "t", method="triangle")

        self.assertTrue(np.allclose(profile, test_profile, atol=0.001))

    def test_radial_profile_is_not_a_view(self):
        """
        Ensure that when getting a radial profile with `method="nearest"`
        the returned array can't be used to change the model.
        """
        m = dummy_model(with_fields=True)
        field_copy = m.get_field("t").copy()
        profile = m.radial_profile(0, 0, "t")
        profile[:] = 999
        self.assertTrue(np.all(m.get_field("t") == field_copy))


class TestModelHealpy(unittest.TestCase):
    def test_hp_sph(self):
        model = dummy_model(with_fields=True)
        model.calc_spherical_harmonics("t")
        a = model.get_spherical_harmonics("t")
        self.assertEqual(len(a), 3)
        self.assertEqual(len(a[0]), 2)

    def test_plot_spectral_heterogeneity(self):
        model = dummy_model(with_fields=True)
        model.calc_spherical_harmonics("t")
        fig, ax = model.plot_spectral_heterogeneity("t", lyrmin=0, lyrmax=-1)
        self.assertIsInstance(fig, Figure)
        self.assertEqual(ax.get_xlabel(), "L")
        self.assertEqual(ax.get_ylabel(), "Depth (km)")


class TestNearestIndex(unittest.TestCase):
    def test_below(self):
        self.assertEqual(terra_model._nearest_index(0.9, [1, 2, 3]), 0)

    def test_above(self):
        self.assertEqual(terra_model._nearest_index(3.1, [1, 2, 3]), 2)

    def test_between(self):
        self.assertEqual(terra_model._nearest_index(2.1, [1, 2, 3]), 1)
        self.assertEqual(terra_model._nearest_index(2.6, [1, 2, 3]), 2)


class TestBoundingIndices(unittest.TestCase):
    def test_below(self):
        self.assertCountEqual(
            terra_model._bounding_indices(-5, [-4, -3, -2]), (0, 0), 2
        )

    def test_above(self):
        self.assertCountEqual(terra_model._bounding_indices(0, [-4, -3, -2]), (2, 2), 2)

    def test_between(self):
        self.assertCountEqual(
            terra_model._bounding_indices(-2.1, [-4, -3, -2]), (1, 2), 2
        )

    def test_equal(self):
        self.assertCountEqual(
            terra_model._bounding_indices(-3, [-4, -3, -2]), (1, 1), 2
        )


class TestPlotSection(unittest.TestCase):
    def test_plot_section(self):
        model = dummy_model(with_fields=True)
        fig, ax = model.plot_section("t", 10, 20, 30, 120, show=False)
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, PolarAxes)

    def test_plot_section_one_layer(self):
        model = dummy_model(with_fields=True)
        radii = model.get_radii()
        radius_diff = radii[-1] - radii[0]
        fig, ax = model.plot_section(
            "t", 10, 20, 30, 120, delta_radius=radius_diff + 1, show=False
        )
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, PolarAxes)


if _CARTOPY_INSTALLED:

    class TestPlotLayer(unittest.TestCase):
        def test_errors(self):
            model = dummy_model(with_fields=True)

            with self.assertRaises(ValueError):
                model.plot_layer("t")

            with self.assertRaises(ValueError):
                model.plot_layer("t", index=-1)
            with self.assertRaises(ValueError):
                model.plot_layer("t", index=len(model.get_radii()) + 1)

        def test_plot_layer_radius(self):
            model = dummy_model(with_fields=True)
            fig, ax = model.plot_layer("t", 4000, show=False)
            self.assertIsInstance(fig, Figure)
            self.assertIsInstance(ax, GeoAxesSubplot)

        def test_plot_layer_depth(self):
            model = dummy_model(with_fields=True)
            fig, ax = model.plot_layer("t", 100, depth=True, show=False)
            self.assertIsInstance(fig, Figure)
            self.assertIsInstance(ax, GeoAxesSubplot)

        def test_plot_layer_index(self):
            model = dummy_model(with_fields=True)
            fig, ax = model.plot_layer("t", index=2, depth=True, show=False)
            self.assertIsInstance(fig, Figure)
            self.assertIsInstance(ax, GeoAxesSubplot)

    class TestPlotHealpy(unittest.TestCase):
        def test_plot_hp_map(self):
            model = dummy_model(with_fields=True)
            model.calc_spherical_harmonics("t")
            fig, ax = model.plot_hp_map("t", index=1)
            self.assertIsInstance(fig, Figure)
            self.assertIsInstance(ax, GeoAxesSubplot)


if __name__ == "__main__":
    unittest.main()
