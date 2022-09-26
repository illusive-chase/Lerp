#include <CGAL/config.h>
#include <CGAL/Epick_d.h>
#include <CGAL/Delaunay_triangulation.h>
#include <CGAL/algorithm.h>
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <array>


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
    
};

template<int D, typename T> Lerp(const std::vector<std::array<T, D>>&)->Lerp<D, T>;



int main()
{
	std::vector<std::array<double, 2>> points = {
		{0,1},
		{0,2},
		{1,1}
	};

	Lerp lerp{ points };

	if (!lerp.is_valid()) return 1;

	lerp.dump(std::cout);
	std::cout << std::endl;

	auto result = lerp.interp(std::array<double, 2>{ 1.5, 1.5 });
	if (result == std::nullopt) return -1;
	auto&& [idxs, weights] = *result;

	for (int idx : idxs) {
		std::cout << idx << ' ';
	}
	std::cout << std::endl;

	for (double weight : weights) {
		std::cout << weight << ' ';
	}
	std::cout << std::endl;

	return 0;
}