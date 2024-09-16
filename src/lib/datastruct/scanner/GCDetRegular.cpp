/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/scanner/GCDetRegular.hpp"

#include "geometry/GCConstants.hpp"

#include <cstdlib>
#include <fstream>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

GCDetRegular::GCDetRegular(GCScanner* pp_scanner) : mp_scanner(pp_scanner)
{
	mp_Xpos = std::make_unique<Array1D<float>>();
	mp_Ypos = std::make_unique<Array1D<float>>();
	mp_Zpos = std::make_unique<Array1D<float>>();
	mp_Xorient = std::make_unique<Array1D<float>>();
	mp_Yorient = std::make_unique<Array1D<float>>();
	mp_Zorient = std::make_unique<Array1D<float>>();
}

void GCDetRegular::generateLUT()
{
	allocate();
	size_t num_blocks = mp_scanner->dets_per_ring / mp_scanner->dets_per_block;
	float block_length;
	if (num_blocks == 2)
	{
		block_length =
		    mp_scanner->crystalSize_trans * mp_scanner->dets_per_block;
	}
	else
	{
		float scanner_longerRadius =
		    mp_scanner->scannerRadius / (std::cos(PI / (float)num_blocks));
		block_length = 2 * std::sqrt((std::pow(scanner_longerRadius, 2) -
		                              std::pow(mp_scanner->scannerRadius, 2)));
	}
	for (size_t doi = 0; doi < mp_scanner->num_doi; doi++)
	{
		float block_distance =
		    mp_scanner->scannerRadius + mp_scanner->crystalDepth * doi;
		for (size_t ring = 0; ring < mp_scanner->num_rings; ring++)
		{
			// Gap between each ring (Currently, equal distance)
			float z_gap = mp_scanner->axialFOV / ((float)mp_scanner->num_rings);
			float z_pos = -mp_scanner->axialFOV / 2.0 + ring * z_gap;
			for (size_t block = 0; block < num_blocks; block++)
			{
				float block_angle = block * TWOPI / num_blocks;
				float x_block = block_distance * std::cos(block_angle);
				float y_block = block_distance * std::sin(block_angle);
				for (size_t det = 0; det < mp_scanner->dets_per_block; det++)
				{
					// relative_det_pos is a number between 0 and 1 describing
					// how far we've gone in the current block
					float relative_det_pos = 0.0;
					if (mp_scanner->dets_per_block != 1)
					{
						relative_det_pos =
						    -(det / (float)(mp_scanner->dets_per_block - 1)) +
						    0.5;
					}
					float relative_x_det =
					    block_length *
					    (relative_det_pos)*std::cos(block_angle + PI / 2.0);
					float relative_y_det =
					    block_length *
					    (relative_det_pos)*std::sin(block_angle + PI / 2.0);

					size_t idx = det + block * mp_scanner->dets_per_block +
					             ring * (mp_scanner->dets_per_ring) +
					             doi * (mp_scanner->num_rings *
					                    mp_scanner->dets_per_ring);

					setXpos(idx, x_block + relative_x_det);
					setYpos(idx, y_block + relative_y_det);
					setZpos(idx, z_pos);
					setXorient(idx, std::cos(block_angle + PI));
					setYorient(idx, std::sin(block_angle + PI));
					setZorient(idx, 0.0);
				}
			}
		}
	}
}

void GCDetRegular::allocate()
{
	size_t num_dets = mp_scanner->getTheoreticalNumDets();
	mp_Xpos->allocate(num_dets);
	mp_Ypos->allocate(num_dets);
	mp_Zpos->allocate(num_dets);
	mp_Xorient->allocate(num_dets);
	mp_Yorient->allocate(num_dets);
	mp_Zorient->allocate(num_dets);
}

void GCDetRegular::writeToFile(const std::string& detReg_fname) const
{
	std::ofstream file;
	file.open(detReg_fname.c_str(), std::ios::binary | std::ios::out);
	if (!file.is_open())
	{
		throw std::runtime_error("Error opening file " + detReg_fname +
		                         "for writing.");
	}
	for (size_t j = 0; j < getNumDets(); j++)
	{
		float Xpos10 = (*mp_Xpos)[j];
		float Ypos10 = (*mp_Ypos)[j];
		float Zpos10 = (*mp_Zpos)[j];

		file.write((char*)(&(Xpos10)), sizeof(float));
		file.write((char*)(&(Ypos10)), sizeof(float));
		file.write((char*)(&(Zpos10)), sizeof(float));

		file.write((char*)(&((*mp_Xorient)[j])), sizeof(float));
		file.write((char*)(&((*mp_Yorient)[j])), sizeof(float));
		file.write((char*)(&((*mp_Zorient)[j])), sizeof(float));
	}
}

// GETTERS AND SETTERS
float GCDetRegular::getXpos(det_id_t detID) const
{
	return (*mp_Xpos)[detID];
}
float GCDetRegular::getYpos(det_id_t detID) const
{
	return (*mp_Ypos)[detID];
}
float GCDetRegular::getZpos(det_id_t detID) const
{
	return (*mp_Zpos)[detID];
}
float GCDetRegular::getXorient(det_id_t detID) const
{
	return (*mp_Xorient)[detID];
}
float GCDetRegular::getYorient(det_id_t detID) const
{
	return (*mp_Yorient)[detID];
}
float GCDetRegular::getZorient(det_id_t detID) const
{
	return (*mp_Zorient)[detID];
}

void GCDetRegular::setXpos(det_id_t detID, float f)
{
	(*mp_Xpos)[detID] = f;
}
void GCDetRegular::setYpos(det_id_t detID, float f)
{
	(*mp_Ypos)[detID] = f;
}
void GCDetRegular::setZpos(det_id_t detID, float f)
{
	(*mp_Zpos)[detID] = f;
}
void GCDetRegular::setXorient(det_id_t detID, float f)
{
	(*mp_Xorient)[detID] = f;
}
void GCDetRegular::setYorient(det_id_t detID, float f)
{
	(*mp_Yorient)[detID] = f;
}
void GCDetRegular::setZorient(det_id_t detID, float f)
{
	(*mp_Zorient)[detID] = f;
}

size_t GCDetRegular::getNumDets() const
{
	return this->mp_Xpos->getSize(0);
}


#if BUILD_PYBIND11
void py_setup_gcdetregular(py::module& m)
{
	auto c = pybind11::class_<GCDetRegular, GCDetectorSetup>(m, "GCDetRegular");
	c.def(py::init<GCScanner*>());

	c.def("generateLUT", &GCDetRegular::generateLUT);
	c.def("setXpos", &GCDetRegular::setXpos);
	c.def("setYpos", &GCDetRegular::setYpos);
	c.def("setZpos", &GCDetRegular::setZpos);
	c.def("setXorient", &GCDetRegular::setXorient);
	c.def("setYorient", &GCDetRegular::setYorient);
	c.def("setZorient", &GCDetRegular::setZorient);
	c.def("getScanner", &GCDetRegular::getScanner);

	c.def("getXposArray",
	      [](const GCDetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* posArr = self.getXposArrayRef();
		      auto buf_info =
		          py::buffer_info(posArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {posArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getYposArray",
	      [](const GCDetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* posArr = self.getYposArrayRef();
		      auto buf_info =
		          py::buffer_info(posArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {posArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getZposArray",
	      [](const GCDetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* posArr = self.getZposArrayRef();
		      auto buf_info =
		          py::buffer_info(posArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {posArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getXorientArray",
	      [](const GCDetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* orientArr = self.getXorientArrayRef();
		      auto buf_info =
		          py::buffer_info(orientArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {orientArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getYorientArray",
	      [](const GCDetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* orientArr = self.getYorientArrayRef();
		      auto buf_info =
		          py::buffer_info(orientArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {orientArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getZorientArray",
	      [](const GCDetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* orientArr = self.getZorientArrayRef();
		      auto buf_info =
		          py::buffer_info(orientArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {orientArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
}
#endif
