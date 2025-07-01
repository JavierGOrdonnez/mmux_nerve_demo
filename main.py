import json
import os
import pathlib as pl

#!/usr/bin/env python
import time
from pathlib import Path

import numpy
import numpy as np
import s4l_v1 as s4l
import s4l_v1.analysis as analysis
import s4l_v1.document as document
import s4l_v1.model as model
import s4l_v1.units as units
import XCoreModeling as xcm
from s4l_v1 import Vec3
from s4l_v1._api.application import get_app_safe, run_application
from s4l_v1.model import Unit
from s4l_v1.simulation import emlf, neuron

start_time = time.time()


x11_available = "DISPLAY" in os.environ
if get_app_safe() is None:
    run_application(disable_ui_plugins=not x11_available)
app = get_app_safe()

osparc = True
if osparc:
    if "INPUTS_FOLDER" in os.environ:
        input_folder = Path(os.environ["INPUTS_FOLDER"])
        output_folder = Path(os.environ["OUTPUTS_FOLDER"])
        comp_service = False
    elif "INPUT_FOLDER" in os.environ:
        input_folder = Path(os.environ["INPUT_FOLDER"])
        output_folder = Path(os.environ["OUTPUT_FOLDER"])
        comp_service = True
    else:
        raise Exception("No INPUTS_FOLDER or INPUTS found in environment variables")
    current_path = Path(os.getcwd())
else:
    input_folder = Path("/home/smu/work/inputs/input_1")
    output_folder = Path("/home/smu/work/outputs/output_1")


# ## Reading the Parameter File
input_values_path = pl.Path(input_folder) / "values.json"
# Gets the electrode parameters
input_values = json.loads(input_values_path.read_text())
filename_model = os.path.join(input_folder, "Nerve_Model.sab")


def GetsSurfaceContact():

	# -*- coding: utf-8 -*-
	import s4l_v1 as s4l
	import XCoreModeling as xcm

	ents=s4l.model.AllEntities()

	contact=ents['Contact 1']
	contact=contact.Clone()
	nerve=ents['Nerve']
	nerve=nerve.Clone()

	slice=xcm.Slice([contact,nerve])

	surf=xcm.CoverWireBody(slice) # type: ignore
	surf[0].Name='Interface'

	surface=xcm.MeasureArea([surf[0]])

	print (surface)

	return surface

def Creates_EM_Simulation(sigmas):
	sigma_blood=sigmas[0] #0.6624597361833767
	sigma_connective=sigmas[1] #0.07919585744745661
	sigma_interst=sigmas[2] #0.07919585744745661
	sigmas_nerve=sigmas[3] #0.3479543931346832
	sigmas_saline=sigmas[4] #2.0
	sigmas_fasc_tra=sigmas[5] # 0.16
	sigmas_fasc_lon=sigmas[6] #0.57

	# Creating the simulation
	simulation = emlf.ElectroQsOhmicSimulation()

	# Mapping the components and entities
	component__plane_x = simulation.AllComponents["Plane X+"]
	component__plane_x = simulation.AllComponents["Plane X-"]
	component__plane_y = simulation.AllComponents["Plane Y+"]
	component__plane_y = simulation.AllComponents["Plane Y-"]
	component__plane_z = simulation.AllComponents["Plane Z+"]
	component__plane_z = simulation.AllComponents["Plane Z-"]
	entity__contact1 = model.AllEntities()["Contact 1"]
	entity__blood2 = model.AllEntities()["Blood 2"]
	entity__fascicle1 = model.AllEntities()["Fascicle 1"]
	entity__fascicle3 = model.AllEntities()["Fascicle 3"]
	entity__fascicle4 = model.AllEntities()["Fascicle 4"]
	entity__fascicle2 = model.AllEntities()["Fascicle 2"]
	entity__fascicle7 = model.AllEntities()["Fascicle 7"]
	entity__connective_2b = model.AllEntities()["Connective_2b"]
	entity__blood3 = model.AllEntities()["Blood 3"]
	entity__fascicle5 = model.AllEntities()["Fascicle 5"]
	entity__connective_2a = model.AllEntities()["Connective_2a"]
	entity__saline = model.AllEntities()["Saline"]
	entity__connective = model.AllEntities()["Connective"]
	entity__nerve = model.AllEntities()["Nerve"]
	entity__silicone = model.AllEntities()["Silicone"]
	entity__interstitial = model.AllEntities()["Interstitial"]
	entity__fascicle6 = model.AllEntities()["Fascicle 6"]
	entity__contact2 = model.AllEntities()["Contact 2"]
	entity__blood1 = model.AllEntities()["Blood 1"]

	# Ensure all components and entities are not None
	components = [
		component__plane_x, component__plane_y, component__plane_z,
	]
	for component in components:
		assert component is not None, f"Component {component} could not be retrieved."

	# Ensure none of the entities are None
	entities = [
		entity__contact1, entity__blood2, entity__fascicle1, entity__fascicle3,
		entity__fascicle4, entity__fascicle2, entity__fascicle7, entity__connective_2b,
		entity__blood3, entity__fascicle5, entity__connective_2a, entity__saline,
		entity__connective, entity__nerve, entity__silicone, entity__interstitial,
		entity__fascicle6, entity__contact2, entity__blood1
	]
	for entity in entities:
		assert entity is not None, f"Entity {entity} could not be retrieved."


	# Adding a new MaterialSettings
	material_settings = emlf.MaterialSettings()
	components = [entity__silicone]
	material_settings.Name = "Silicone"
	simulation.Add(material_settings, components)

	# Adding a new MaterialSettings
	material_settings = emlf.MaterialSettings()
	components = [entity__blood1, entity__blood2, entity__blood3]
	material_settings.Name = "Blood"
	material_settings.MassDensity = 1049.75, Unit("kg/m^3") # type: ignore
	material_settings.ElectricProps.Conductivity = sigma_blood, Unit("S/m")
	material_settings.ElectricProps.RelativePermittivity = 5258.608390020375
	simulation.Add(material_settings, components)

	# Adding a new MaterialSettings
	material_settings = emlf.MaterialSettings()
	components = [entity__connective, entity__connective_2a, entity__connective_2b]
	material_settings.Name = "Connective Tissue"
	material_settings.MassDensity = 1026.5, Unit("kg/m^3") # type: ignore
	material_settings.ElectricProps.Conductivity = sigma_connective, Unit("S/m")
	material_settings.ElectricProps.RelativePermittivity = 302705.19018286216
	simulation.Add(material_settings, components)

	# Adding a new MaterialSettings
	material_settings = emlf.MaterialSettings()
	components = [
		entity__fascicle1,
		entity__fascicle2,
		entity__fascicle3,
		entity__fascicle4,
		entity__fascicle5,
		entity__fascicle6,
		entity__fascicle7,
	]
	material_settings.Name = "Fascicles"
	material_settings.ElectricProps.ConductivityAnisotropic = True
	material_settings.ElectricProps.ConductivityDiagonalElements = numpy.array(
		[sigmas_fasc_tra, sigmas_fasc_tra, sigmas_fasc_lon]
	), Unit("S/m")
	simulation.Add(material_settings, components)

	# Adding a new MaterialSettings
	material_settings = emlf.MaterialSettings()
	components = [entity__interstitial]
	material_settings.Name = "Connective Tissue"
	material_settings.MassDensity = 1026.5, Unit("kg/m^3") # # type: ignore
	material_settings.ElectricProps.Conductivity = sigma_interst, Unit("S/m")
	material_settings.ElectricProps.RelativePermittivity = 302705.19018286216
	simulation.Add(material_settings, components)

	# Adding a new MaterialSettings
	material_settings = emlf.MaterialSettings()
	components = [entity__nerve]
	material_settings.Name = "Nerve"
	material_settings.MassDensity = 1075.0, Unit("kg/m^3") # # type: ignore
	material_settings.ElectricProps.Conductivity = sigmas_nerve, Unit("S/m")
	material_settings.ElectricProps.RelativePermittivity = 69911.4914652573
	simulation.Add(material_settings, components)

	# Adding a new MaterialSettings
	material_settings = emlf.MaterialSettings()
	components = [entity__saline]
	material_settings.Name = "Saline"
	material_settings.ElectricProps.Conductivity =sigmas_saline, Unit("S/m")
	simulation.Add(material_settings, components)

	# Editing BoundarySettings "Boundary Settings
	boundary_settings = [
		x
		for x in simulation.AllSettings
		if isinstance(x, emlf.BoundarySettings) and x.Name == "Boundary Settings"
	][0]
	components = [
		component__plane_x,
		component__plane_x,
		component__plane_y,
		component__plane_y,
		component__plane_z,
		component__plane_z,
	]
	simulation.Add(boundary_settings, components)
	boundary_settings.BoundaryType = boundary_settings.BoundaryType.enum.Flux

	# Adding a new BoundarySettings
	boundary_settings = emlf.BoundarySettings()
	components = [entity__contact1]
	boundary_settings.Name = "Anode"
	boundary_settings.DirichletValue = 0.5, units.Volts
	simulation.Add(boundary_settings, components)

	# Adding a new BoundarySettings
	boundary_settings = emlf.BoundarySettings()
	components = [entity__contact2]
	boundary_settings.Name = "Catode"
	boundary_settings.DirichletValue = -0.5, units.Volts
	simulation.Add(boundary_settings, components)

	# Editing GlobalGridSettings "Grid (Empty)
	global_grid_settings = simulation.GlobalGridSettings
	global_grid_settings.DiscretizationMode = (
		global_grid_settings.DiscretizationMode.enum.Manual
	)
	global_grid_settings.MaxStep = numpy.array([0.05, 0.05, 0.05]), units.MilliMeters
	global_grid_settings.Resolution = (
		numpy.array([0.625, 0.625, 0.625]),
		units.MilliMeters,
	)
	global_grid_settings.PaddingMode = global_grid_settings.PaddingMode.enum.Manual
	global_grid_settings.BottomPadding = numpy.array([0.1, 0.1, 0.1]), units.MilliMeters
	global_grid_settings.TopPadding = numpy.array([0.1, 0.1, 0.1]), units.MilliMeters

	# Adding a new ManualGridSettings
	manual_grid_settings = simulation.AddManualGridSettings(
		[
			entity__blood1,
			entity__blood2,
			entity__blood3,
			entity__connective,
			entity__connective_2a,
			entity__connective_2b,
			entity__fascicle1,
			entity__fascicle2,
			entity__fascicle3,
			entity__fascicle4,
			entity__fascicle5,
			entity__fascicle6,
			entity__fascicle7,
			entity__interstitial,
			entity__nerve,
		]
	)
	manual_grid_settings.Name = "Nerve"
	manual_grid_settings.MaxStep = numpy.array([0.01, 0.01, 0.05]), units.MilliMeters
	manual_grid_settings.Priority = 0.0

	# Adding a new ManualGridSettings
	manual_grid_settings = simulation.AddManualGridSettings(
		[entity__contact1, entity__contact2, entity__saline, entity__silicone]
	)
	manual_grid_settings.Name = "Else"
	manual_grid_settings.MaxStep = numpy.array([0.05, 0.05, 0.05]), units.MilliMeters
	manual_grid_settings.Priority = 0.0

	# Adding a new ManualVoxelerSettings
	manual_voxeler_settings = emlf.ManualVoxelerSettings()
	components = [entity__saline]
	manual_voxeler_settings.Name = "Saline"
	simulation.Add(manual_voxeler_settings, components)

	# Adding a new ManualVoxelerSettings
	manual_voxeler_settings = emlf.ManualVoxelerSettings()
	components = [entity__silicone]
	manual_voxeler_settings.Name = "Silicone"
	manual_voxeler_settings.Priority = 1
	simulation.Add(manual_voxeler_settings, components)

	# Adding a new ManualVoxelerSettings
	manual_voxeler_settings = emlf.ManualVoxelerSettings()
	components = [entity__contact1, entity__contact2]
	manual_voxeler_settings.Name = "Contacts"
	manual_voxeler_settings.Priority = 2
	simulation.Add(manual_voxeler_settings, components)

	# Adding a new ManualVoxelerSettings
	manual_voxeler_settings = emlf.ManualVoxelerSettings()
	components = [entity__nerve]
	manual_voxeler_settings.Name = "Nerve"
	manual_voxeler_settings.Priority = 3
	simulation.Add(manual_voxeler_settings, components)

	# Adding a new ManualVoxelerSettings
	manual_voxeler_settings = emlf.ManualVoxelerSettings()
	components = [entity__interstitial]
	manual_voxeler_settings.Name = "Intrafascicular"
	manual_voxeler_settings.Priority = 4
	simulation.Add(manual_voxeler_settings, components)

	# Adding a new ManualVoxelerSettings
	manual_voxeler_settings = emlf.ManualVoxelerSettings()
	components = [
		entity__fascicle1,
		entity__fascicle2,
		entity__fascicle3,
		entity__fascicle4,
		entity__fascicle5,
		entity__fascicle6,
		entity__fascicle7,
	]
	manual_voxeler_settings.Name = "Fascicles"
	manual_voxeler_settings.Priority = 7
	simulation.Add(manual_voxeler_settings, components)

	# Adding a new ManualVoxelerSettings
	manual_voxeler_settings = emlf.ManualVoxelerSettings()
	components = [entity__blood1, entity__blood2, entity__blood3]
	manual_voxeler_settings.Name = "Blood"
	manual_voxeler_settings.Priority = 6
	simulation.Add(manual_voxeler_settings, components)

	# Adding a new ManualVoxelerSettings
	manual_voxeler_settings = emlf.ManualVoxelerSettings()
	components = [entity__connective, entity__connective_2a, entity__connective_2b]
	manual_voxeler_settings.Name = "Connective"
	manual_voxeler_settings.Priority = 5
	simulation.Add(manual_voxeler_settings, components)

	# Editing SolverSettings "Solver
	solver_settings = simulation.SolverSettings
	solver_settings.PredefinedTolerances = (
		solver_settings.PredefinedTolerances.enum.High
	)

	# Update the materials with the new frequency parameters
	simulation.UpdateAllMaterials()

	# Update the grid with the new parameters
	simulation.UpdateGrid()

	# Add the simulation to the UI
	document.AllSimulations.Add(simulation)

	return simulation

def Creates_Electrode(length, gap, angle, radius, silicone_extra):

	center = Vec3(0, 0, 1)

	angle = (360 - angle) * 2 * np.pi / 360
	start = 0.5 * angle
	end = -0.5 * angle

	# Creates the Arc
	arc = xcm.CreateArc(center, radius, start, end)
	vertices = [v.Position for v in xcm.GetVertices(arc)]

	verts = [arc]
	for v in vertices:
		verts.append(s4l.model.CreatePolyLine([center, v]))

	s1 = s4l.model.Unite(verts)
	xcm.CoverWireBody(s1)

	contact = ents["Arc 1"]
	contact.Name = "Contact 1"

	arc = xcm.CreateArc(center, radius + 0.05, start - 0.1, end + 0.1)
	vertices = [v.Position for v in xcm.GetVertices(arc)]

	verts = [arc]
	for v in vertices:
		verts.append(s4l.model.CreatePolyLine([center, v]))

	s1 = s4l.model.Unite(verts)
	xcm.CoverWireBody(s1)

	silicone = ents["Arc 1"]
	silicone.Name = "Silicone"

	e1 = s4l.model.CreatePolyLine([Vec3(0, 0, 0), Vec3(0, 0, length)])
	xcm.SweepAlongPath(contact, e1, True, False)
	T = s4l.Translation(Vec3(0, 0, -0.5 * length))
	contact.ApplyTransform(T)
	e1.Delete()

	contact1 = contact.Clone()
	contact1.Name = "Contact 2"

	silicone_length=2*(silicone_extra+length)+gap

	e2 = s4l.model.CreatePolyLine([Vec3(0, 0, 0), Vec3(0, 0, silicone_length)])
	xcm.SweepAlongPath(silicone, e2, True, False)
	T = s4l.Translation(Vec3(0, 0, -0.5 * silicone_length))
	silicone.ApplyTransform(T)
	e2.Delete()

	dt = 0.5 * length + 0.5 * gap
	T = s4l.Translation(Vec3(0, 0, dt))
	contact.ApplyTransform(T)
	T = s4l.Translation(Vec3(0, 0, -dt))
	contact1.ApplyTransform(T)


def Gets_Flux(em_sensor_extractor):
    # Adding a new CurrentExtractor
    inputs = [
        em_sensor_extractor.Outputs["EM Potential(x,y,z,f0)"],
        em_sensor_extractor.Outputs["J(x,y,z,f0)"],
    ]
    current_extractor = analysis.extractors.CurrentExtractor(inputs=inputs)
    current_extractor.UpdateAttributes()
    current_extractor.Update()
    flux = np.real(current_extractor.GetOutput(0).GetComponent(0))[0]

    return flux


def ExtractThresholdsInfo(sim):
    """The function extracts the titration results from an electrophysiological simulation"""

    assert sim.HasResults() == 1, "Neuronal Simulation Should Have Results"

    results = sim.Results()
    titration_evaluator = s4l.analysis.neuron_evaluators.TitrationEvaluator()
    titration_evaluator.Inputs[0].Connect(results["Titration Sensor"]["Titration"])
    titration_evaluator.Update(0)
    tf = list(titration_evaluator.TitrationFactor)  # titration factor
    tl = list(titration_evaluator.LocationOfFirstSpike)  # section of the first spike
    ts = list(titration_evaluator.TimeOfFirstSpike)  # latency, in [ms]
    nnames = list(titration_evaluator.NeuronName)  # name of the neuron spiking

    return tf


def ExtractsResults(simulation):

	## Normalize the field
	assert simulation.HasResults(), "EM Simulation Should Have Results"
	print("Scaling E_Potential")
	simulation_extractor = simulation.Results()
	em_sensor_extractor = simulation_extractor["Overall Field"]

	###### flux
	flux = Gets_Flux(em_sensor_extractor)

	# Power Density (Without NaN for Thermal Simulation)
	scale = 0.001 / flux

	inputs = [em_sensor_extractor.Outputs["EM Potential(x,y,z,f0)"]]
	user_defined_field_normalizer = s4l.analysis.field.UserDefinedFieldNormalizer(
		inputs=inputs
	)
	user_defined_field_normalizer.Target.Value = scale
	user_defined_field_normalizer.UpdateAttributes()
	user_defined_field_normalizer.Update()
	s4l.document.AllAlgorithms.Add(user_defined_field_normalizer)

	# Adding a new UserDefinedFieldNormalizer
	inputs = [em_sensor_extractor.Outputs["EM E(x,y,z,f0)"]]
	user_defined_field_normalizer = analysis.field.UserDefinedFieldNormalizer(
		inputs=inputs
	)
	user_defined_field_normalizer.Target.Value = scale
	user_defined_field_normalizer.Description = "Field Scaling E-Field"
	user_defined_field_normalizer.UpdateAttributes()
	document.AllAlgorithms.Add(user_defined_field_normalizer)

	inputse = [user_defined_field_normalizer.Outputs["EM E(x,y,z,f0)"]]
	field_masking_filtere = s4l.analysis.core.FieldMaskingFilter(inputs=inputse)
	field_masking_filtere.UseNaN = True
	field_masking_filtere.SetAllMaterials(True)
	field_masking_filtere.Name = "Mask"

	ids = field_masking_filtere.MaterialIds()
	materials = field_masking_filtere.MaterialNames()
	field_masking_filtere.UpdateAttributes()
	field_masking_filtere.Update()

	listd = ["Contact 1", "Contact 2", "Silicone","Saline"]
	for i in range(len(ids)):
		if materials[i] in listd:
			field_masking_filtere.SetMaterial(ids[i], False)

	field_masking_filtere.UpdateAttributes()
	field_masking_filtere.Update()

	# Adding a new FieldInterpolationFilter
	inputs = [field_masking_filtere.Outputs["EM E(x,y,z,f0)"]]	

	# Adding a new VolumeAverageElectricFieldEvaluator
	#inputs = [user_defined_field_normalizer.Outputs["EM E(x,y,z,f0)"]]
	volume_average_electric_field_evaluator = (
		analysis.em_evaluators.VolumeAverageElectricFieldEvaluator(inputs=inputs)
	)
	volume_average_electric_field_evaluator.CubeLength = 0.0001, units.Meters
	volume_average_electric_field_evaluator.UpdateAttributes()
	document.AllAlgorithms.Add(volume_average_electric_field_evaluator)

	# Extracting Peak Averaged
	evaluator = volume_average_electric_field_evaluator.Outputs[
		"Volume-Average Report [ICNIRP 2010]"
	]
	assert evaluator.Update()
	keys = evaluator.Data.RowKeys()
	values = evaluator.Data.ToList()

	for i in range(len(keys)):
		print("Key-value pairs ExtractResults: ", keys[i].split(" (")[0], values[i][0])

	peak_averaged_field = values[0][0]

	return [flux, peak_averaged_field]  # Flux Current, Peak Averaged E-Field


def CreatesNeuroCache(axonlist,pulse_duration):
    sim = neuron.Simulation()
    s4l.document.AllSimulations.Add(sim)

    # Setup Settings
    setup_settings = sim.SetupSettings
    setup_settings.TitrationStrategy = 1
    setup_settings.PerformTitration = True
    setup_settings.DepolarizationDetection.enum.Threshold
    setup_settings.DepolarizationThreshold = 80.0

    automatic_axon_neuron_settings = neuron.AutomaticAxonNeuronSettings()
    automatic_axon_neuron_settings.Temperature = 37

    # Neuron Settings
    diams = []
    for axon in axonlist:
        sim.Add(automatic_axon_neuron_settings, [axon])

    # Adding a new SourceSettings
    source_settings = sim.AddGenericSource([])
    source_settings.SourceType = source_settings.SourceType.enum.DataObject
    source_settings.SourceDataObject.DataOriginType = (
        source_settings.SourceDataObject.DataOriginType.enum.Pipeline
    )
    source_settings.SourceDataObject.DataOrigin = (
        document.AllAlgorithms["Field Scaling"].Outputs[0].raw
    )
    source_settings.SourceDataObject.DataOrigin.Update()
    source_settings.PulseType = source_settings.PulseType.enum.Bipolar
    source_settings.InitialTime = 0.0002, units.Seconds
    source_settings.AmplitudeP1 = 1
    source_settings.DurationP1 = pulse_duration*0.001, units.Seconds

    # Editing SolverSettings "Solver
    solver_settings = sim.SolverSettings
    solver_settings.NumberOfThreads = 5
    solver_settings.Duration = 0.012, units.Seconds

    return sim

def Create_Axon_Distribution():

    axon_ent = s4l.model.CreateGroup("Axons_Folder")
    point_ent = s4l.model.CreateGroup("Points_Folder")

    options = xcm.MeshingOptions()
    options.EdgeLength = 0.04  # To change to change the density of points
    options.MinEdgeLength = 0.011
    ns = 1  # downsampling factor

    cnt = 0
    cn = 0
    axonn = 0
    axons = []
    for ent in ents["Fascicles_Folder"].Entities:
        # for ent in ents['Fascicles_Meshes'].Entities:

        # Gets the bounding box of the fascicle
        entity_bounds = xcm.GetBoundingBox([ent])
        L = 0.95 * (entity_bounds[1][2] - entity_bounds[0][2])
        z0 = 0.5 * (entity_bounds[1][2] + entity_bounds[0][2])

        # Creates a slice of the fascicle at given location
        slice = xcm.CreatePlanarSlice(ent, Vec3(0, 0, 0), Vec3(0, 0, 1), 1)
        xcm.CoverWireBody(slice)
        slice.Name = "tempslice_" + ent.Name
        name = ent.Name

        slice = xcm.ConvertToTriangleMesh(slice)
        slice.Name = name

        # slice=xcm.ConvertToTriangleMesh(slice,max_edge_length=0.1) #target=0.1, min=0.001
        xcm.RemeshTriangleMesh(slice, options)

        # Creates a subfolder of axons and point for each fascicle
        entfold = s4l.model.CreateGroup("S_Points_" + name)
        splines = s4l.model.CreateGroup("S_Splines_" + name)

        # Gets the surface of the fascicle and calculates the equivalent diameter
        surf = xcm.MeasureArea([slice])
        diam = 2 * np.sqrt(surf / np.pi)

        # Creates a surface mesh on the fascicle cross section. The centers of
        # the triangles are candidate location of the axons
        faceter = s4l.analysis.core.ModelToGridFilter()
        faceter.Entity = slice
        faceter.MaximumEdgeLength = 0.3 * diam
        faceter.Update(0)
        target_grid = faceter.GetOutput(0)
        # L=86

        cm = 0
        # Creates the axon trajectories
        for i in range(0, target_grid.NumberOfCells, ns):
            p = 1e3 * target_grid.GetCellCenter(i)
            point = s4l.model.CreatePoint(p)
            point.Name = "P_" + name + "_" + str(cm) + "_Random"
            entfold.Add(point)

            # z = np.random.uniform(zz_min,zz_max)
            L1 = L * np.random.normal(1, 0.025)
            x = p[0]
            y = p[1]
            z = p[2]
            a = s4l.model.CreateSpline(
                [Vec3(x, y, z0 - 0.5 * L1), Vec3(x, y, z0 + 0.5 * L1)]
            )
            a.Name = name + "_Spline_" + str(cm) + "_Random"
            splines.Add(a)
            axons.append(a)

            cm += 1
            cn += 1
            axonn += 1

        cnt += 1

        # Adds the axon trajectories and the point in the folders
        axon_ent.Add(splines)
        point_ent.Add(entfold)

        # Delete the slice from the model
        slice.Delete()

    print("Total number of axons:", axonn)

    # Creates Unique Folder for Fibers and Points
    fold = s4l.model.CreateGroup("Functionalization_" + str(cn))
    fold.Add(axon_ent)
    fold.Add(point_ent)

    return axons


ents = s4l.model.AllEntities()
xcm.Import(filename_model)

# ## Creation of the Electrode Geometry
# Read the electrode parameters
radius = 0.7 
length = input_values["number_1"]
gap = input_values["number_2"]
angle = input_values["number_3"]
silicone_extra = input_values["number_4"]

### TODO for testing post-pro -- remove later, start always from scratch
model_path = output_folder / "model.smash"
# Creates Electrode Parameterized
Creates_Electrode(length, gap, angle, radius, silicone_extra)
s4l.document.SaveAs(model_path) # type: ignore

# ### Creation of Axon Distribution
axons = Create_Axon_Distribution()
# Converts into myelinated axons
senn_props = s4l.model.SennNeuronProperties()
senn_props.AxonDiameter = 10
axonlist = model.CreateAxonNeurons(axons, senn_props)

# ## Creation and Execution EM Simulation
# Creates the EM Simulation
length = input_values["number_1"]
gap = input_values["number_2"]
angle = input_values["number_3"]
silicone_extra = input_values["number_4"]

sigmas=np.zeros(7)
sigmas[0]=input_values["number_5"] #0.6624597361833767 sigma_blood=
sigmas[1]=input_values["number_6"] #0.07919585744745661 sigma_connective=
sigmas[2]=input_values["number_7"] #0.07919585744745661 sigma_interst=
sigmas[3]=input_values["number_8"] #0.3479543931346832 sigmas_nerve=
sigmas[4]=input_values["number_9"] #2.0 sigmas_saline=
sigmas[5]=input_values["number_10"] # 0.16 sigmas_fasc_tra=
sigmas[6]=input_values["number_11"] #0.57 sigmas_fasc_lon=

simulation = Creates_EM_Simulation(sigmas)
simulation.UpdateGrid()
simulation.CreateVoxels()
simulation.RunSimulation(wait=True)

# Extracts the Potential
[current, peak_averaged_field] = ExtractsResults(simulation)

# ### Creates and Runs The Neuronal Simulation
pulse_duration=input_values["number_12"]
neuron_simulation = CreatesNeuroCache(axonlist,pulse_duration)
neuron_simulation.RunSimulation(wait=True)
tf = ExtractThresholdsInfo(neuron_simulation)

s4l.document.Save() # type: ignore 
## Save model (to be able to inspect it)

# Calculates Isopercentiles
isop5 = np.percentile(tf, 5)
isop50 = np.percentile(tf, 50)
isop95 = np.percentile(tf, 95)
charge5=isop5*pulse_duration
charge50=isop50*pulse_duration # [ms]*[mA]= [uC]
charge95=isop95*pulse_duration #

# Get Interface Surface
surface=GetsSurfaceContact()/100 # from mm2 to cm2 
charged5=charge5/surface
charged50=charge50/surface

# Shannon Criteria
shannon5=np.log10(charged5)+np.log10(charge5)
shannon50=np.log10(charged50)+np.log10(charge50)

end_time = time.time()
print(f"Total time spent in pipeline: {end_time - start_time} seconds")

# ### Provide Output Data
impedance=1./current # [kOhm]


## convert to the format that the (de)jsonifier understand
output_values = {
    "number_1": float(impedance), 
    "number_2": float(peak_averaged_field), 
    "number_3": float(isop5), 
    "number_4": float(isop50), 
    "number_5": float(isop95),
    "number_6": float(charge5), 
    "number_7": float(charge50), 
    "number_8": float(charge95),
    "number_9":float(charged5),
    "number_10":float(charged50),
    "number_11":float(shannon5),
    "number_12":float(shannon50)
    }
print(output_values)
output_values_path = pl.Path(output_folder / "values.json")
output_values_path.write_text(json.dumps(output_values))
