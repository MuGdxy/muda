#pragma once
#include <muda/muda_def.h>
#include <muda/ext/geo/distance/distance_type.h>
#include <muda/ext/geo/distance/point_triangle.h>
#include <muda/ext/geo/distance/point_edge.h>
#include <muda/ext/geo/distance/point_point.h>
#include <muda/ext/geo/distance/edge_edge.h>

namespace muda::distance
{
template <class T>
MUDA_GENERIC void point_point_distance_unclassified(const Eigen::Vector<T, 3>& p0,
                                                    const Eigen::Vector<T, 3>& p1,
                                                    T& dist2);

template <class T>
MUDA_GENERIC void point_triangle_distance_unclassified(const Eigen::Vector<T, 3>& p,
                                                       const Eigen::Vector<T, 3>& t0,
                                                       const Eigen::Vector<T, 3>& t1,
                                                       const Eigen::Vector<T, 3>& t2,
                                                       T& dist2);

template <class T>
MUDA_GENERIC void edge_edge_distance_unclassified(const Eigen::Vector<T, 3>& ea0,
                                                  const Eigen::Vector<T, 3>& ea1,
                                                  const Eigen::Vector<T, 3>& eb0,
                                                  const Eigen::Vector<T, 3>& eb1,
                                                  T& dist2);

// http://geomalgorithms.com/a02-_lines.html
template <class T>
MUDA_GENERIC void point_edge_distance_unclassified(const Eigen::Vector<T, 3>& p,
                                                   const Eigen::Vector<T, 3>& e0,
                                                   const Eigen::Vector<T, 3>& e1,
                                                   T& dist2);

}  // namespace muda::distance

#include "details/distance_unclassified.inl"
