#include <CGAL/config.h>
#include <CGAL/Epick_d.h>
#include <CGAL/Delaunay_triangulation.h>
#include <CGAL/algorithm.h>
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <array>
#include <iomanip>
#include "ply.h"

namespace _impl {
	template<typename T> struct get_static_length {};

	template<typename T, size_t Len>
	struct get_static_length<const T(&)[Len]> {
		static constexpr size_t value = Len;
	};

	template<typename T, size_t Len>
	struct get_static_length<const std::array<T, Len>&> {
		static constexpr size_t value = Len;
	};

	template<typename T, typename F, size_t Ind0, size_t ... Inds>
	std::ostream& join_impl(std::ostream& os, char delimiter, const T& cont, F&& f, std::index_sequence<Ind0, Inds...>) {
		os << f(cont[Ind0]);
		return (..., (os << delimiter << f(cont[Inds])));
	}

	struct join_helper {
		template<typename T>
		constexpr T&& operator()(T&& arg) const noexcept { return static_cast<T&&>(arg); }
	};
}

template<typename T>
std::ostream& join(std::ostream& os, char delimiter, const T& cont) {
	return _impl::join_impl(os, delimiter, cont, _impl::join_helper{}, std::make_index_sequence<_impl::get_static_length<const T&>::value>());
}

template<typename T, typename F>
std::ostream& join(std::ostream& os, char delimiter, const T& cont, F&& f) {
	return _impl::join_impl(os, delimiter, cont, f, std::make_index_sequence<_impl::get_static_length<const T&>::value>());
}

template<>
std::ostream& join<Eigen::Matrix3d>(std::ostream& os, char delimiter, const Eigen::Matrix3d& mat) {
	return _impl::join_impl(os, delimiter, mat.data(), _impl::join_helper{}, std::make_index_sequence<9>());
}

template<>
std::ostream& join<Eigen::Matrix4d>(std::ostream& os, char delimiter, const Eigen::Matrix4d& mat) {
	return _impl::join_impl(os, delimiter, mat.data(), _impl::join_helper{}, std::make_index_sequence<16>());
}

template<typename T>
double dot3(const T& lhs, const T& rhs) {
	return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

template<typename T>
double dot4(const T& lhs, const T& rhs) {
	return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2] + lhs[3] * rhs[3];
}

template<typename T>
double dot6(const T& lhs, const T& rhs) {
	return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2] + lhs[3] * rhs[3] + lhs[4] * rhs[4] + lhs[5] * rhs[5];
}

template<typename T>
double distance6(const T& lhs, const T& rhs) {
	double diff[6];
	for (int i = 0; i < 6; ++i) {
		diff[i] = lhs[i] * rhs[i];
	}
	return dot6(diff, diff);
}

std::array<double, 6> lerp_pose(const std::array<double, 6>& lp, const std::array<double, 6>& rp, double alpha) {
	std::array<double, 6> lerped;
	for (int i = 0; i < 6; ++i) {
		lerped[i] = lp[i] * (1 - alpha) + rp[i] * alpha;
	}
	return lerped;
}

std::array<double, 6> pose2vec(const Eigen::Matrix4d& pose) {
	Eigen::Vector3d translation = pose.block(0, 3, 3, 1);
	Eigen::Matrix3d rot = pose.block(0, 0, 3, 3);
	Eigen::Quaterniond quaternion(rot);
	
	const double omega = acos(quaternion.w());
	const double frac = omega < 1e-6 ? 1.0 : (omega / sin(omega));
	return {
		quaternion.x() * frac,
		quaternion.y() * frac,
		quaternion.z() * frac,
		translation.x(),
		translation.y(),
		translation.z()
	};
}

Eigen::Matrix4d vec2pose(const std::array<double, 6>& vec) {
	const double omega = sqrt(dot3(vec, vec));
	const double frac = omega < 1e-6 ? 1 : sin(omega) / omega;
	double q[4];
	q[0] = cos(omega);
	q[1] = vec[0] * frac;
	q[2] = vec[1] * frac;
	q[3] = vec[2] * frac;
	const double norm = dot4(q, q);
	for (int j = 0; j < 4; ++j) q[j] /= norm;

	Eigen::Matrix4d pose = Eigen::Matrix4d::Zero();
	Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
	pose.block(0, 0, 3, 3) = quat.toRotationMatrix();
	pose.block(0, 3, 3, 1) = Eigen::Vector3d(vec.data() + 3);
	return pose;
}



template<size_t P>
class BspineCurve {
	
	mutable std::vector<std::array<double, P>> m_dp;
	mutable size_t m_i0;
	mutable double m_x;

	size_t m_N;
	std::vector<double> m_u;

	double B(size_t i, size_t p) const {
		if (i + p < m_i0 || i > m_i0) return 0.0;
		if (p == 0) return 1.0;
		double& v = m_dp[i][p - 1];
		if (v == 0.0) {
			double b1 = B(i, p - 1);
			double b2 = B(i + 1, p - 1);
			double p1 = m_u[i + p] == m_u[i] ? 0.0 : ((m_x - m_u[i]) / (m_u[i + p] - m_u[i]) * b1);
			double p2 = m_u[i + p + 1] == m_u[i + 1] ? 0.0 : ((m_u[i + p + 1] - m_x) / (m_u[i + p + 1] - m_u[i + 1]) * b2);
			v = p1 + p2;
		}
		return v;
	}

public:
	BspineCurve(size_t N) : m_N(N), m_x(0.0), m_i0(0), m_u(N + P + 1) {
		if (N > P + 1) {
			for (size_t i = m_N; i < m_N + P + 1; ++i) m_u[i] = 1.0;
			for (size_t i = 1; i < m_N - P; ++i) m_u[P + i] = static_cast<double>(i) / (m_N - P);
			m_dp.resize(m_N + P + 1);
		}
	}

	std::vector<double> solve(double x) const {
		if (m_N <= P + 1) return {};

		// [P + 1] [N - P - 1] [P + 1]
		// divide into N - P blocks
		// B_{P,0} ... B_{N-1,0}

		m_x = x;
		m_i0 = P + static_cast<size_t>(x * (m_N - P));
		
		for (size_t i = 0; i < m_dp.size(); ++i) m_dp[i].fill(0);

		std::vector<double> result(m_N);
		for (size_t i = 0; i < m_N; ++i) result[i] = B(i, P);
		return result;
	}

	bool valid() const {
		return m_N > P + 1;
	}
};


class SimplexGraph {
private:
	std::unordered_map<int, int> m_remap;
	std::vector<int> m_nodes;
	Eigen::MatrixXi m_edges, m_paths;
	Eigen::MatrixXd m_distances;

	inline void create_if_not_exists(int i) {
		if (!m_remap.count(i)) {
			int last = static_cast<int>(m_nodes.size());
			m_remap[i] = last;
			m_nodes.push_back(i);
		}
	}

	void get_path(std::vector<int>& paths, int from, int to) const {
		int pre = m_paths(from, to);
		if (pre == -1) return;
		get_path(paths, from, pre);
		paths.push_back(m_nodes[pre]);
		get_path(paths, pre, to);
	}


public:
	
	template<typename D>
	SimplexGraph(const std::vector<std::vector<int>>& facets, D&& distance_function) {
		for (const auto& facet : facets) {
			for (int v : facet) create_if_not_exists(v);
		}
		const size_t N = m_nodes.size();
		m_edges = Eigen::MatrixXi::Zero(N, N);
		m_paths = Eigen::MatrixXi::Constant(N, N, -1);
		m_distances = Eigen::MatrixXd::Constant(N, N, std::numeric_limits<double>::max() * 0.4);
		for (size_t i = 0; i < N; ++i) m_distances(i, i) = 0;

		for (const auto& facet : facets) {
			for (size_t i = 0; i < facet.size(); ++i) {
				const int ii = m_remap[facet[i]];
				for (size_t j = i + 1; j < facet.size(); ++j) {
					const int jj = m_remap[facet[j]];
					if (m_edges(ii, jj) == 0) {
						m_edges(ii, jj) = m_edges(jj, ii) = 1;
						m_distances(ii, jj) = m_distances(jj, ii) = distance_function(ii, jj);
					}
				}
			}
		}

		for (size_t k = 0; k < N; ++k) {
			for (size_t i = 0; i < N; ++i) {
				for (size_t j = 0; j < N; ++j) {
					double trial_distance = m_distances(i, k) + m_distances(k, j);
					if (m_distances(i, j) > trial_distance) {
						m_distances(i, j) = trial_distance;
						m_paths(i, j) = static_cast<int>(k);
					}
				}
			}
		}
	}

	
	std::vector<int> get_longest_path() const {
		const size_t N = m_nodes.size();
		size_t max_i = 0, max_j = 1;
		double max_d = 0;
		for (size_t i = 0; i < N; ++i) {
			for (size_t j = i + 1; j < N; ++j) {
				if (m_edges(i, j) != 0 && m_distances(i, j) > max_d) {
					max_d = m_distances(i, j);
					max_i = i;
					max_j = j;
				}
			}
		}
		std::vector<int> result;
		result.reserve(N);
		if (m_edges(max_i, max_j) != 0) {
			result.push_back(m_nodes[max_i]);
			get_path(result, static_cast<int>(max_i), static_cast<int>(max_j));
			result.push_back(m_nodes[max_j]);
		}
		return result;
	}

	friend std::ostream& operator <<(std::ostream& os, const SimplexGraph& g) {
		const size_t N = g.m_nodes.size();
		for (size_t i = 0; i < N; ++i) {
			for (size_t j = i + 1; j < N; ++j) {
				if (g.m_edges(i, j) != 0) os << i << ',' << j << ':' << g.m_distances(i, j) << std::endl;
			}
		}
		return os << std::endl;
	}
	
};


template<int D, typename Scalar = double>
class Lerp {
	static_assert(D > 1, "Dim must be greater than 1");

	using DT = CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>;
	using Point = typename DT::Point;
	using PIter = typename std::vector<Point>::const_iterator;

	std::vector<Point> m_points;
	mutable std::unordered_map<typename DT::Vertex_const_handle, int> m_handles;
	DT m_dt{ D };
	bool m_valid;

	static constexpr size_t factorial() {
		size_t factorial = 1;
		for (size_t i = 1; i <= D; i++) {
			factorial *= i;
		}
		return factorial;
	}

	Scalar compute_volume(const Eigen::Matrix<Scalar, D, D + 1>& points) const {
		Eigen::Matrix<Scalar, D, D> m = points.block<D, D>(0, 1);
		for (size_t j = 0; j < D; j++) {
			m.col(j) -= points.col(0);
		}
		Scalar det = m.determinant();
		return det / factorial();
	}

public:
	Lerp(const Lerp&) = delete;

	template<typename Vd, std::enable_if_t<sizeof(Vd) == D * sizeof(Scalar), int> I = 0>
	Lerp(const std::vector<Vd>& coordinates): m_valid(false) {
		m_points.reserve(coordinates.size());
		for (const Vd& v : coordinates) {
			m_points.emplace_back(std::begin(v), std::begin(v) + D);
		}
		if (!m_points.empty()) {
			auto hint = m_dt.insert(m_points[0]);
			m_handles[hint] = 0;
			for (size_t i = 1; i < m_points.size(); ++i) {
				hint = m_dt.insert(m_points[i], hint);
				m_handles[hint] = static_cast<int>(i);
			}
			m_valid = m_dt.is_valid() && m_dt.current_dimension() == D;
		}
	}

    int dim() const {
        return m_dt.current_dimension();
    }

	bool is_valid() const {
		return m_valid;
	}

	std::ostream& dump(std::ostream& os) const {
		for (auto cit = m_dt.full_cells_begin(); cit != m_dt.full_cells_end(); ++cit) {
			if (m_dt.is_infinite(cit)) continue;
			for (auto it = cit->vertices_begin(); it != cit->vertices_end(); ++it) {
				os << (m_handles.count(*it) ? m_handles[*it] : -1) << ' ';
			}
			os << std::endl;
		}
		return os;
	}

	template<typename Vd>
	std::enable_if_t<sizeof(Vd) == D * sizeof(Scalar), std::optional<std::pair<std::array<int, D + 1>, std::array<Scalar, D + 1>>>> interp(const Vd& point) const {
		std::array<int, D + 1> idxs = {};
		std::array<Scalar, D + 1> weights = {};
		Eigen::Matrix<Scalar, D, D + 1> points;


		if (!m_valid) return std::nullopt;
		Point p{ std::begin(point), std::begin(point) + D };

		auto ch = m_dt.locate(p);
		size_t i = 0;
		for (auto it = ch->vertices_begin(); it != ch->vertices_end(); ++it) {
			if (!m_handles.count(*it)) return std::nullopt;
			idxs[i] = m_handles[*it];
			points.col(i) = Eigen::Array<Scalar, 1, D>(m_points[idxs[i]].data());
			++i;
		}

		
		Scalar whole_volume = compute_volume(points);
		if (whole_volume == 0.0) return std::nullopt;
		for (i = 0; i < D + 1; i++) {
			points.col(i) = Eigen::Array<Scalar, 1, D>(p.data());
			Scalar sub_volume = compute_volume(points);
			weights[i] = sub_volume / whole_volume;
			points.col(i) = Eigen::Array<Scalar, 1, D>(m_points[idxs[i]].data());
		}
		return std::make_pair(idxs, weights);
	}

	std::vector<std::array<double, 6>> sample_traj(int max_step) const {
		std::vector<int> vertices;
		std::vector<std::array<double, 6>> traj;
		for (auto cit = m_dt.full_cells_begin(); cit != m_dt.full_cells_end(); ++cit) {
			if (m_dt.is_infinite(cit)) continue;
			for (auto it = cit->vertices_begin(); it != cit->vertices_end(); ++it) {
				if (!m_handles.count(*it)) return traj;
				vertices.push_back(m_handles[*it]);
			}
			break;
		}
		if (vertices.empty()) return traj;
		std::array<double, 6> center = { 0,0,0,0,0,0 };
		std::array<double, 6> endpoint = {};
		for (int ind : vertices) {
			for (int i = 0; i < 6; ++i) {
				center[i] += m_points[ind][i];
			}
		}
		for (int i = 0; i < 6; ++i) {
			center[i] /= vertices.size();
			endpoint[i] += m_points[0][i];
		}
		for (int step = 1; step <= max_step; ++step) {
			const double alpha = static_cast<double>(step) / max_step;
			traj.emplace_back(lerp_pose(center, endpoint, alpha));
		}
		return traj;
	}

	std::vector<std::array<double, 6>> get_roam_traj(int max_step) const {
		std::vector<std::vector<int>> facets;
		std::vector<std::array<double, 6>> traj;

		for (auto cit = m_dt.full_cells_begin(); cit != m_dt.full_cells_end(); ++cit) {
			if (m_dt.is_infinite(cit)) continue;
			int inf_idx = -1;
			for (int i = 0; i < 7; ++i) {
				auto n = cit->neighbor(i);
				if (m_dt.is_infinite(n)) {
					if (inf_idx >= 0) inf_idx = 7;
					else inf_idx = i;
				}
			}
			if (inf_idx < 0) continue;
			std::vector<int> full_connected;
			for (int i = 0; i < 7; ++i) {
				if (inf_idx == i) continue;
				auto v = cit->vertex(i);
				if (!m_handles.count(v)) return traj;
				full_connected.push_back(m_handles[v]);
			}
			facets.push_back(full_connected);
		}

		// build graph to calc longest traj

		// TODO: add circle test

		SimplexGraph g(facets, [&](size_t i, size_t j) {
			return distance6(m_points[i], m_points[j]);
			});
		
		const auto& traj_idxs = g.get_longest_path();
		
		// TODO: B spine

		BspineCurve<2> spine(traj_idxs.size());
		if (!spine.valid()) return traj;

		for (int step = 0; step < max_step; ++step) {
			const double alpha = static_cast<double>(step) / (max_step - 1);
			const auto& weights = spine.solve(alpha);
			std::array<double, 6> pose = { 0,0,0,0,0,0 };
			for (size_t i = 0; i < traj_idxs.size(); ++i) {
				if (weights[i] == 0) continue;
				for (size_t j = 0; j < 6; ++j) {
					pose[j] += weights[i] * m_points[traj_idxs[i]][j];
				}
			}
			traj.emplace_back(pose);
		}

		return traj;
	}
    
};

template<int D, typename T> Lerp(const std::vector<std::array<T, D>>&)->Lerp<D, T>;


struct Image {
	int id;
	std::string path;
	Eigen::Matrix4d extrinsic;
	Eigen::Matrix4d extrinsic_inv;
	Eigen::Matrix3d intrinsic;
	double W, H;

	Image(int id, const std::string& path, const double(&quaternion)[4], const double(&translation)[3], const double(&intrinsic)[4]) :
		id(id),
		path(path),
		extrinsic(Eigen::Matrix4d::Zero()),
		intrinsic(Eigen::Matrix3d::Zero()),
		W(intrinsic[2] * 2),
		H(intrinsic[3] * 2)
	{
		const Eigen::Quaterniond q(quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]);
		extrinsic.block(0, 0, 3, 3) = q.toRotationMatrix();
		extrinsic.block(0, 3, 3, 1) = q * -Eigen::Vector3d(translation);
		extrinsic.col(1) *= -1;
		extrinsic.col(2) *= -1;
		extrinsic(3, 3) = 1;
		extrinsic_inv = extrinsic.inverse().eval();
		this->intrinsic(0, 0) = intrinsic[1];
		this->intrinsic(1, 1) = intrinsic[1];
		this->intrinsic(0, 2) = intrinsic[2];
		this->intrinsic(1, 2) = intrinsic[3];
		this->intrinsic(2, 2) = 1;
	}


	void texture_map(const Eigen::MatrixX3d& vertices, Eigen::Block<Eigen::MatrixXd, -1, 1, true>&& u_ref, Eigen::Block<Eigen::MatrixXd, -1, 1, true>&& v_ref) const {
		size_t nrows = vertices.rows();
		Eigen::Matrix4Xd xyz(4, nrows);
		xyz.block(0, 0, 3, nrows) = vertices.transpose();
		xyz.block(3, 0, 1, nrows) = Eigen::VectorXd::Ones(nrows).transpose();
		auto p = (intrinsic * (extrinsic_inv * xyz).block(0, 0, 3, nrows)).transpose().array();
		u_ref = 1.0 - (p.col(0) / p.col(2)) / W;
		v_ref = 1.0 - (p.col(1) / p.col(2)) / H;
	}
};

std::vector<std::vector<double>> texture_mapping(const std::vector<std::array<double, 3>>& positions, const std::vector<Image>& image_infos, const std::vector<int>& image_indexes) {
	Eigen::MatrixX3d vertices(positions.size(), 3);
	Eigen::MatrixXd projected(positions.size(), image_indexes.size() * 2);

	for (size_t i = 0; i < positions.size(); ++i) {
		vertices.row(i) = Eigen::Array3d(positions[i].data()).transpose();
	}

	for (size_t i = 0; i < image_indexes.size(); ++i) {
		image_infos[image_indexes[i]].texture_map(vertices, projected.col(i * 2), projected.col(i * 2 + 1));
	}

	std::vector<std::vector<double>> result(positions.size(), std::vector<double>(image_indexes.size() * 2));
	
	for (size_t i = 0; i < positions.size(); ++i) {
		for (size_t j = 0; j < image_indexes.size() * 2; ++j) {
			result[i][j] = projected(i, j);
		}
	}

	return result;
}




int main()
{
	std::vector<std::array<double, 6>> camera_data;
	std::vector<std::array<double, 6>> roaming_trajectories;
	std::vector<std::pair<std::array<int, 7>, std::array<double, 7>>> frames;
	std::vector<Image> image_infos;
	

	{
		std::string line;

		{
			std::ifstream ifs("model.json", std::ios::in);
			std::getline(ifs, line);
			std::getline(ifs, line);
		}

		{
			std::istringstream iss(line);
			while (true) {
				std::string path;
				path.reserve(32);
				int image_id;
				double quaternion[4];
				double translation[3];
				double intrinsic[4];
				iss >> image_id;
				if (iss.fail() || iss.eof()) break;
				iss.ignore(1024, ',');
				for (double& x : quaternion) {
					iss >> x;
					iss.ignore(1024, ',');
				}
				for (double& x : translation) {
					iss >> x;
					iss.ignore(1024, ',');
				}
				for (double& x : intrinsic) {
					iss >> x;
					iss.ignore(1024, ',');
				}
				iss.ignore(1024, '"');
				while (true) {
					char c = iss.get();
					if (c == '"') break;
					path += c;
				}
				iss.ignore(1024, ',');
				if (iss.fail() || iss.eof()) break;

				image_infos.emplace_back(image_id, path, quaternion, translation, intrinsic);
				camera_data.emplace_back(pose2vec(image_infos.back().extrinsic));
			}
		}

	}

	if (0) {
		size_t len = camera_data.size();
		const auto& lp = camera_data[len / 2 - 1];
		const auto& rp = camera_data[len / 2];
		const int max_step = 50;
		for (int step = 1; step <= max_step; ++step) {
			const double alpha = static_cast<double>(step) / max_step;
			roaming_trajectories.emplace_back(lerp_pose(lp, rp, alpha));
		}
	}


	std::vector<int> remap_list;
	std::unordered_map<int, int> remap_map;

	{
		Lerp<6> lerp{ camera_data };

		if (!lerp.is_valid()) {
			std::cerr << "Error! Invalid camera data" << std::endl;
			return 1;
		}

		// lerp.dump(std::cout);
		// std::cout << std::endl;

		roaming_trajectories = lerp.get_roam_traj(50);

		for (const auto& traj : roaming_trajectories) {
			auto result = lerp.interp(traj);
			if (result == std::nullopt) {
				continue;
				std::cerr << "Error! Break with framesize=" << frames.size() << std::endl;
				break;
			}
			for (int i : (*result).first) {
				if (!remap_map.count(i)) {
					remap_map[i] = static_cast<int>(remap_list.size());
					remap_list.push_back(i);
				}
			}
			frames.emplace_back(*result);
		}
	}

	
	

	{
		std::ofstream ofs("vis.json", std::ios::out);

		ofs << "{\"frames\":[" << std::endl;

		for (size_t i = 0; i < frames.size(); ++i) {
			const auto& [idxs, weights] = frames[i];
			

			if (i > 0) ofs << ',' << std::endl;
			ofs << "{\"extrinsic\":[";
			join(ofs << std::fixed << std::setprecision(5), ',', vec2pose(roaming_trajectories[i]));
			ofs << "],\"ids\":[";
			join(ofs, ',', idxs, [&](int i) { return remap_map[i]; });
			ofs << "],\"weights\":[";
			join(ofs << std::fixed << std::setprecision(5), ',', weights);
			ofs << "]}";
		}

		ofs << std::endl << "],\"uvs\":[" << std::endl;

		{
			std::vector<std::array<double, 3>> vertices;
			std::vector<std::array<size_t, 3>> faces;

			{
				std::string mesh_path = "dt.ply";
				happly::PLYData ply_in(mesh_path);
				vertices = ply_in.getVertexPositions();
				for (const auto& face : ply_in.getFaceIndices()) {
					if (face.size() == 3) faces.push_back({ face[0], face[1], face[2] });
				}
			}

			auto uvs = texture_mapping(vertices, image_infos, remap_list);
			for (size_t i = 0; i < uvs.size(); ++i) {
				if (i > 0) ofs << ',' << std::endl;
				ofs << '[';
				for (size_t j = 0; j < uvs[i].size(); ++j) {
					if (j > 0) ofs << ',';
					ofs << std::fixed << std::setprecision(5) << uvs[i][j];
				}
				ofs << ']';
			}

			ofs << std::endl << "],\"vertices\":[" << std::endl;
			for (size_t i = 0; i < vertices.size(); ++i) {
				if (i > 0) ofs << ',';
				ofs << '[';
				join(ofs << std::fixed << std::setprecision(5), ',', vertices[i]);
				ofs << ']';
			}

			ofs << std::endl << "],\"faces\":[" << std::endl;
			for (size_t i = 0; i < faces.size(); ++i) {
				if (i > 0) ofs << ',';
				ofs << '[';
				join(ofs, ',', faces[i]);
				ofs << ']';
			}
		}

		ofs << std::endl << "],\"paths\":[" << std::endl;

		for (size_t i = 0; i < remap_list.size(); ++i) {
			if (i > 0) ofs << ',' << std::endl;
			ofs << '"' << image_infos[remap_list[i]].path << '"';
		}

		ofs << std::endl << "],\"poses\":[" << std::endl;

		for (size_t i = 0; i < remap_list.size(); ++i) {
			if (i > 0) ofs << ',' << std::endl;
			ofs << '[';
			join(ofs << std::fixed << std::setprecision(5), ',', image_infos[remap_list[i]].extrinsic);
			ofs << ']';
		}

		ofs << std::endl << "],\"Ks\":[" << std::endl;

		for (size_t i = 0; i < remap_list.size(); ++i) {
			if (i > 0) ofs << ',' << std::endl;
			ofs << '[';
			join(ofs << std::fixed << std::setprecision(5), ',', image_infos[remap_list[i]].intrinsic);
			ofs << ']';
		}

		ofs << std::endl << "]}" << std::endl;
		ofs.close();
	}

	

	

	return 0;
}