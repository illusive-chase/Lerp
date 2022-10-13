#include "lerp.hh"
#include <CGAL/config.h>
#include <CGAL/Epick_d.h>
#include <CGAL/Delaunay_triangulation.h>
#include <CGAL/algorithm.h>
#include <Eigen/Dense>
#include <optional>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <array>
#include <utility>

template<typename T>
double distance(const T& lhs, const T& rhs, size_t D) {
    double sum = 0.0;
    for (int i = 0; i < static_cast<int>(D); ++i) {
        double diff = lhs[i] - rhs[i];
        sum += diff * diff;
    }
    return sum;
}

template<size_t D>
std::array<double, D> lerp_pose(const std::array<double, D>& lp, const std::array<double, D>& rp, double alpha) {
    std::array<double, D> lerped;
    for (size_t i = 0; i < D; ++i) {
        lerped[i] = lp[i] * (1 - alpha) + rp[i] * alpha;
    }
    return lerped;
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
            double p1 = m_u[i + p] == m_u[i] ? b1 : ((m_x - m_u[i]) / (m_u[i + p] - m_u[i]) * b1);
            double p2 = m_u[i + p + 1] == m_u[i + 1] ? b2 : ((m_u[i + p + 1] - m_x) / (m_u[i + p + 1] - m_u[i + 1]) * b2);
            v = p1 + p2;
        }
        return v;
    }

public:
    BspineCurve(size_t N) : m_i0(0), m_x(0.0), m_N(N), m_u(N + P + 1) {
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

        std::vector<double> result(m_N);

        m_x = x;
        m_i0 = P + static_cast<size_t>(x * (m_N - P));
        if (m_i0 >= m_N) {
            result.back() = 1.0;
            return result;
        }

        for (size_t i = 0; i < m_dp.size(); ++i) m_dp[i].fill(0);

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
        if (N > 1) {
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
    }


    std::vector<int> get_longest_path() const {
        const size_t N = m_nodes.size();
        if (N < 2) return {};
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


template<size_t D, typename Scalar = double>
class Lerp {
    static_assert(D > 1, "Dim must be greater than 1");

    using DT = CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>;
    using Point = typename DT::Point;
    using PIter = typename std::vector<Point>::const_iterator;

    std::vector<Point> m_points;
    std::array<Scalar, D> m_center;
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
        Eigen::Matrix<Scalar, D, D> m = points.block(0, 1, D, D);
        for (size_t j = 0; j < D; j++) {
            m.col(j) -= points.col(0);
        }
        Scalar det = m.determinant();
        return det / factorial();
    }

public:
    Lerp(const Lerp&) = delete;

    template<typename Vd, std::enable_if_t<sizeof(Vd) == D * sizeof(Scalar), int> I = 0>
    Lerp(const std::vector<Vd>& coordinates) : m_valid(false) {
        m_points.reserve(coordinates.size());
        for (size_t i = 0; i < D; ++i) m_center[i] = 0;
        for (const Vd& v : coordinates) {
            for (size_t i = 0; i < D; ++i) m_center[i] += v[i];
            m_points.emplace_back(std::begin(v), std::begin(v) + D);
        }
        if (!m_points.empty()) {
            for (size_t i = 0; i < D; ++i) m_center[i] /= coordinates.size();
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
        std::array<int, D + 1> idxs;
        std::array<Scalar, D + 1> weights;
        Eigen::Matrix<Scalar, D, D + 1> points;
        std::array<Scalar, D> fixed_point;
        for (size_t i = 0; i < D; ++i) fixed_point[i] = point[i] * static_cast<Scalar>(0.999) + m_center[i] * static_cast<Scalar>(0.001);

        if (!m_valid) return std::nullopt;
        Point p{ std::begin(fixed_point), std::begin(fixed_point) + D };

        auto ch = m_dt.locate(p);
        size_t i = 0;
        for (auto it = ch->vertices_begin(); it != ch->vertices_end(); ++it) {
            if (!m_handles.count(*it)) return std::nullopt;
            idxs[i] = m_handles[*it];
            points.col(i) = Eigen::Array<Scalar, 1, D>(m_points[idxs[i]].data());
            ++i;
        }

        if (i != D + 1) return std::nullopt;

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

    std::vector<std::array<double, D>> sample_traj(size_t max_step) const {
        std::vector<int> vertices;
        std::vector<std::array<double, D>> traj;
        for (auto cit = m_dt.full_cells_begin(); cit != m_dt.full_cells_end(); ++cit) {
            if (m_dt.is_infinite(cit)) continue;
            for (auto it = cit->vertices_begin(); it != cit->vertices_end(); ++it) {
                if (!m_handles.count(*it)) return traj;
                vertices.push_back(m_handles[*it]);
            }
            break;
        }
        if (vertices.empty()) return traj;
        std::array<double, D> center;
        std::array<double, D> endpoint;
        for (size_t i = 0; i < D; ++i) center[i] = endpoint[i] = 0;
        for (int ind : vertices) {
            for (size_t i = 0; i < D; ++i) {
                center[i] += m_points[ind][i];
            }
        }
        for (size_t i = 0; i < D; ++i) {
            center[i] /= vertices.size();
            endpoint[i] += m_points[0][i];
        }
        for (size_t step = 1; step <= max_step; ++step) {
            const double alpha = static_cast<double>(step) / max_step;
            traj.emplace_back(lerp_pose(center, endpoint, alpha));
        }
        return traj;
    }

    std::vector<std::array<double, D>> get_roam_traj(size_t max_step) const {
        std::vector<std::vector<int>> facets;
        std::vector<std::array<double, D>> traj;

        for (auto cit = m_dt.full_cells_begin(); cit != m_dt.full_cells_end(); ++cit) {
            if (m_dt.is_infinite(cit)) continue;
            int inf_idx = -1;
            for (int i = 0; i < static_cast<int>(D + 1); ++i) {
                auto n = cit->neighbor(i);
                if (m_dt.is_infinite(n)) {
                    if (inf_idx >= 0) inf_idx = static_cast<int>(D + 1);
                    else inf_idx = i;
                }
            }
            if (inf_idx < 0) continue;
            std::vector<int> full_connected;
            for (int i = 0; i < static_cast<int>(D + 1); ++i) {
                if (inf_idx == i) continue;
                auto v = cit->vertex(i);
                if (!m_handles.count(v)) return traj;
                full_connected.push_back(m_handles[v]);
            }
            facets.push_back(full_connected);
        }

        // build graph to calc longest traj

        SimplexGraph g(facets, [&](size_t i, size_t j) {
            return distance(m_points[i], m_points[j], D);
            });

        const auto& traj_idxs = g.get_longest_path();

        BspineCurve<2> spine(traj_idxs.size());
        if (!spine.valid() || max_step == 0) return traj;

        for (size_t step = 0; step < max_step; ++step) {
            const double alpha = static_cast<double>(step) / (max_step - 1);
            const auto& weights = spine.solve(alpha);
            std::array<double, D> pose;
            for (size_t i = 0; i < D; ++i) pose[i] = 0;
            for (size_t i = 0; i < traj_idxs.size(); ++i) {
                if (weights[i] == 0) continue;
                for (size_t j = 0; j < D; ++j) {
                    pose[j] += weights[i] * m_points[traj_idxs[i]][static_cast<int>(j)];
                }
            }
            traj.emplace_back(pose);
        }

        return traj;
    }

};

template<size_t D, typename T> Lerp(const std::vector<std::array<T, D>>&)->Lerp<D, T>;


namespace lerp {

    std::vector<std::tuple<std::array<double, 6>, std::array<int, 7>, std::array<double, 7>>> get_frames(size_t num_frames, const std::vector<std::array<double, 6>>& camera_data) {

        Lerp lerp{ camera_data };
        std::vector<std::tuple<std::array<double, 6>, std::array<int, 7>, std::array<double, 7>>> frames;

        if (!lerp.is_valid()) {
            std::cerr << "Error! Invalid camera data" << std::endl;
            return frames;
        }

        const auto& trajs{ lerp.get_roam_traj(num_frames) };

        if (trajs.empty()) {
            std::cerr << "Error! Empty trajectories" << std::endl;
            return frames;
        }

        for (const auto& traj : trajs) {
            auto result = lerp.interp(traj);
            if (result == std::nullopt) {
                std::cerr << "Warning! Skip when framesize=" << frames.size() << std::endl;
                continue;
            }
            auto&& [idxs, weights] = std::move(*result);
            frames.emplace_back(std::make_tuple(traj, idxs, weights));
        }
        return frames;
    }

}