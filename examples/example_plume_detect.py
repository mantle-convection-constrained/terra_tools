# %% [markdown]
"""
# example_plume_detect

This example demonstrates how to read in terra output in NetCDF format
and detect plumes based on temperature and radial velocity.
"""

# %% [markdown]
"""
Let's import all the necessary python objects.
"""

# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from terratools.terra_model import *
from terratools.example_data import example_terra_model

# %% [markdown]
"""
Now let's download and read in the example mantle convection model
netcdf file.
"""

# %%
# Download and cache model
path = example_terra_model()

# read in the model
model = read_netcdf([path])

# %% [markdown]
"""
In order to detect plumes our TerraModel needs to contain velocity (`u_xyz`) and
temperature (`t`) fields. Let's check
"""
# %%

print(model.field_names())

# %% [markdown]
"""
The velocity field `u_xyz` is in cartesian coordinates. To get the radial velocity,
which is needed to identify plumes, we need to convert to geographic coordinates.
Terratools has a built in command to do this, after running you will see that the
field `u_enu` has been added to the list of fields in the TerraModel.
"""
# %%

model.add_geog_flow()
print(model.field_names())


# %% [markdown]
"""
Now we attempt to detect plumes. Note that if you forget the previous step the
`TerraModel.detect_plumes` method will try to calculate the geographic flow
velocities.

Running `model.detect_plumes` without any arguments will use some default parameters
in the plume identification scheme.
"""
# %%

model.detect_plumes()


# %% [markdown]
"""
We can pass in arguments to change the depth range over which we look for plumes
(`depth_range`), the number of times that k-means is run with different initial
centroids (`n-init`), the spatial clustering algorithm which is used (`algorithm`),
the threshold distance if using `algorithm="DBSCAN"` or minimum cluster size if using
`algorithm="HDBSCAN"`, and the minium number of samples in a neighbourhood (`minsamples`).

For example, we could choose to look for plumes over a depth range of 500 km - 2000 km
by setting `depth_range=(500,2000)`
"""
# %%

model.detect_plumes(depth_range=(500, 2000))


# %% [markdown]
"""
Detecting plumes adds an inner `plumes` class to the TerraModel.
The number of plumes detected can be returned with
"""
# %%

print(model.plumes.n_plms)

# %% [markdown]
"""
We can calculate the centroids of each plume at each radial layer in which they were
detected.
"""
# %%

model.plumes.calc_centroids()

# %% [markdown]
"""
We can get the centroids for the plume with plumeID = 1. The columns are lon, lat, depth.
"""
# %%

plm1_centroids = model.plumes.centroids[1]


# %% [markdown]
"""
We can also find the values of any field in within a plume.
For example if we wanted the temperature field we would use
"""
# %%

model.plumes.radial_field("t")

# %% [markdown]
"""
For the plume with plumeID = 1, the temperature field in the upper most layer in
which the plume was detected is
"""
# %%

plm1_top_temp = model.plumes.plm_flds["t"][1][0]

# %% [markdown]
"""
We can get the corresponding coordinates for these points. Columns are
lon, lat, depth.
"""
# %%

plm1_top_coords = model.plumes.plm_coords[1][0]

# %% [markdown]
"""
We include a method for calculating plume buoyancy fluxes.
By default the routine assumes a constant thermal expansivity with depth, but this you
can toggle a depth varying expansivity for example by passing `eos=Murnghan`.
Buoyancy flux calculations require a density field to be present in the TerraModel so
we can not calculate it with this example data, but your command might look something
like this `model.plumes.buoyancy_flux(400,depth=True)`.


Finally lets produce a map of the stacked array of points which were considered to be
plume-like by the k-means clustering. We will also plot centroid in the top layer of
each detected plume.
"""
# %%

model.plumes.plot_kmeans_stack(centroids=0, delta=1)
plt.show()
