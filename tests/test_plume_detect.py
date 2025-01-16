"""
Test the reading of files in the NetCDF format
"""

import unittest

import numpy as np
import os
import re
import tempfile

from netCDF4 import Dataset
from terratools import terra_model
from tests import test_terra_model_read_netcdf as tst


class TestPlumeDetect(unittest.TestCase):
    def test_terra_model_read_netcdf(self):
        npts = 10000
        nlayers = 8
        lon, lat, r, fields, c_hist_names = tst.random_model(
            npts, nlayers, density=True
        )

        with tempfile.TemporaryDirectory() as dir:
            filebase = os.path.join(dir, "test_netcdf_file_")
            filenames = tst.write_seismic_netcdf_files(
                filebase, lon, lat, r, fields, c_hist_names=c_hist_names
            )
            model = terra_model.read_netcdf(filenames)

        # run plume detection and calculate centroids at each layer
        model.detect_plumes()
        model.plumes.calc_centroids()
        model.plumes.radial_field("t")
        model.plumes.buoyancy_flux(300, depth=True)

        self.assertTrue(hasattr(model.plumes, "centroids"))
        self.assertTrue(hasattr(model.plumes, "n_plms"))
        self.assertTrue(hasattr(model.plumes, "n_noise"))
        self.assertTrue(hasattr(model.plumes, "plm_coords"))
        self.assertTrue(hasattr(model.plumes, "plm_lyrs_range"))
        self.assertTrue(hasattr(model.plumes, "plm_depth_range"))
        self.assertTrue(hasattr(model.plumes, "plm_depths"))
        self.assertTrue(hasattr(model.plumes, "plm_flds"))
        self.assertTrue(hasattr(model.plumes, "flux"))
        self.assertTrue(hasattr(model.plumes, "excess_t"))


if __name__ == "__main__":
    unittest.main()
