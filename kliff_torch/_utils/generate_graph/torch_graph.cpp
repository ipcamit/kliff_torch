#include <torch/script.h>
#include <torch/extension.h>

#include <vector>
#include <string>

// #include "helper.hpp"
#include "neighbor_list.h"
#include "periodic_table.hpp"
#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;

struct GraphData{
  GraphData(){};
  int n_layers;
  std::vector<torch::Tensor> edge_index ;
  torch::Tensor pos;
  torch::Tensor energy;
  torch::Tensor forces;
  std::vector<int>images;
  torch::Tensor elements;
  torch::Tensor contributions;
};

// Quick and dirty
// TODO try and see ways to reduce memory allocation
// For some reason it looks like original neighbor lists
// were following singleton pattern, but it not clear why.
// Mostly because the time taken to initialize the NeigList object
// Shall be minimal as opposed to calculations. And in any case data
// is copied to python as return_value_policy was not defined explicitly.
// <*It might be useful in getting distributed graphs.*> As once calculated
// neighbors will be reused. But as on python side the here will be a live
// refernce, not sure if "nodelete" is needed.

void  graph_set_to_graph_array(std::vector<std::set<std::tuple<long, long>>> &
			       unrolled_graph, int64_t ** graph_edge_indices_out) {
    int i = 0;
    for (auto const edge_index_set: unrolled_graph) {
        int j = 0;
        int graph_size = static_cast<int>(edge_index_set.size());
        graph_edge_indices_out[i] = new long[graph_size * 2];
        for (auto bond_pair: edge_index_set) {
            graph_edge_indices_out[i][j] = std::get<0>(bond_pair);
            graph_edge_indices_out[i][j + graph_size] = std::get<1>(bond_pair);
            j++;
        }
        i++;
    }
}

GraphData get_complete_graph(int n_graph_layers, double cutoff,
			     std::vector<std::string>& element_list,
			     py::array_t<double,
			     py::array::c_style | py::array::forcecast>& coords,
			     py::array_t<double>& cell,
			     py::array_t<int>& pbc){
  int n_atoms = element_list.size();
  double infl_dist = cutoff * n_graph_layers;
  std::vector<int> species_code(n_atoms);
  for (auto elem: element_list){
    species_code.push_back(get_z(elem)); 
  }

  int Npad;
  std::vector<double> pad_coords;
  std::vector<int> pad_species;
  std::vector<int> pad_image;

  NeighList *nl;
  nbl_initialize(&nl);
  nbl_create_paddings(n_atoms,
		      cutoff,
		      cell.data(),
		      pbc.data(),
		      coords.data(),
		      species_code.data(),
		      Npad,
		      pad_coords,
		      pad_species,
		      pad_image);
  // std::cout << "Here:  " << pad_species.size() << "  " << Npad <<"\n";
  // std::cout << pad_coords.size() << "\n";
  int n_coords = n_atoms * 3;
  int padded_coord_size = n_coords + Npad*3;
  
  // std::cout << padded_coord_size <<"\n";

  double * padded_coords = new double[padded_coord_size];
  int * need_neighbors = new int[n_atoms + Npad];
  
  auto r = coords.unchecked<2>();
  int pos_ij = 0;
  for(int i = 0; i < n_atoms; i++){
    for(int j = 0; j < 3; j++){
      pos_ij = i * 3 + j;
      padded_coords[pos_ij] = r(i,j);
    }
    need_neighbors[i] = 1;
  }
  for(int i = 0; i < Npad; i++){
    for(int j = 0; j< 3; j++){
      pos_ij = (n_atoms + i) * 3 +  j;
      padded_coords[pos_ij] = pad_coords[i*3 + j];
    }
    need_neighbors[n_atoms + i] = 1;
  }

  for (int  i = 0; i < n_atoms + Npad; i++){
    for (int j = 0; j < 3 ; j++) {
      // std::cout << padded_coords[i * 3 + j] << "  ";
    }
    // std::cout << "\n";
  }
  
  //cast to const
  double* const & const_padded_coords = padded_coords;
  int* const & const_need_neighbors = need_neighbors;

  // std::cout << "Here3\n";
  nbl_build(nl,
            n_atoms + Npad,
            const_padded_coords,
            infl_dist,
            1,
            &cutoff,
            const_need_neighbors);
  // std::cout << "Here4\n";


  // Build complete graph
  // TODO distributed graph generation (basically unroll the loop)
  int number_of_neighbors;
  int const * neighbors;
  int neighbor_list_index = 0;

  std::tuple<int, int> bond_pair, rev_bond_pair;
  std::vector<std::set<std::tuple<long, long> > > unrolled_graph(n_graph_layers);
  std::vector<int> next_list, prev_list;

  for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
    prev_list.push_back(atom_i);
    for (int i = 0; i < n_graph_layers; i++) {
      // std::set<std::tuple<int, int> > conv_layer;
      if (!prev_list.empty()){
	// this condition is needed for edge cases where the selected atom has no neighbors
	// I dont think it will be ever encountered in real problems so not sure if this is
	// nessecary. TODO See if I can remove it safely.
	do {
	  int curr_atom = prev_list.back();
	  prev_list.pop_back();

	  nbl_get_neigh(nl,
			1,
			&cutoff,
			neighbor_list_index,
			curr_atom,
			&number_of_neighbors,
			&neighbors);
	
	  for (int j = 0; j < number_of_neighbors; j++) {
	    bond_pair = std::make_tuple(curr_atom, neighbors[j]);
	    rev_bond_pair = std::make_tuple(neighbors[j], curr_atom);
	    unrolled_graph[i].insert(bond_pair);
	    unrolled_graph[i].insert(rev_bond_pair);
	    next_list.push_back(neighbors[j]);
	  }
	  // neighbor list pointer just points to nl object list, so not needed to be freed
	} while (!prev_list.empty());
	// std::cout << "List prev:" << prev_list.size() <<": List next:" << next_list.size() <<"\n";
	prev_list.swap(next_list);
	//unrolled_graph[i].insert(conv_layer.begin(), conv_layer.end());
      }
    }
    prev_list.clear();
  }
  int64_t ** graph_edge_indices = new int64_t *[n_graph_layers];
  graph_set_to_graph_array(unrolled_graph, graph_edge_indices);

  GraphData gs;
  gs.n_layers = n_graph_layers;
  auto int_options = torch::TensorOptions().dtype(torch::kI64).device(torch::kCPU).requires_grad(false);
  auto double_options = torch::TensorOptions()
                         .dtype(torch::kF64)
                         .device(torch::kCPU)
                         .requires_grad(true);
  for(int i = 0; i < n_graph_layers; i++){
    auto edge_index_i = torch::from_blob(graph_edge_indices[i],
					 {2, unrolled_graph[i].size()},
					 int_options).clone();
    gs.edge_index.push_back(edge_index_i);
  }
  gs.pos = torch::from_blob(padded_coords, {n_atoms + Npad, 3}, double_options).clone();

  species_code.reserve(pad_species.size());
  for (auto padding_species: pad_species){
    species_code.push_back(padding_species);
  }
  gs.elements = torch::from_blob(species_code.data(),{species_code.size()},int_options).clone();

  gs.images = pad_image;

  for (int i = 0; i < n_atoms; i++){
    need_neighbors[i] = 0;
  }
  gs.contributions = torch::from_blob(need_neighbors, {n_atoms + Npad}, int_options).clone();
  
  delete[] padded_coords;
  delete[] need_neighbors;
  for (int i = 0; i < n_graph_layers; i++){
    delete[] graph_edge_indices[i];
  }
  delete[] graph_edge_indices;
  nbl_clean(&nl);
  return gs;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  py::class_<GraphData>(m, "GraphData")
    .def(py::init<>())
    .def_readwrite("edge_index",&GraphData::edge_index)
    .def_readwrite("pos", &GraphData::pos)
    .def_readwrite("n_layers", &GraphData::n_layers)
    .def_readwrite("energy", &GraphData::energy)
    .def_readwrite("forces", &GraphData::forces)
    .def_readwrite("images", &GraphData::images)
    .def_readwrite("elements", &GraphData::elements)
    .def_readwrite("contributions", &GraphData::contributions);
  m.def("get_complete_graph", &get_complete_graph, "gets complete graphs");
    }
